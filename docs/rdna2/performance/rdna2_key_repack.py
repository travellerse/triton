#!/usr/bin/env python3
"""Can the key be repacked into the fast GEMM layout cheaply enough to be worth it?

Measured facts this builds on (rdna2_key_layout_ab.py, production shapes q=5120 k=10287 H=32 D=128):
  * key laid out token-contiguous (packed [BH, D, N])  -> 74.3 ms/call, QK^T 24.0 ms
  * key as production builds it (D-contiguous, tokens strided by 4096) -> 102.6 ms, QK^T 55.7 ms
  * `.contiguous()` on the production view, timed inside the call -> 100.1 ms, i.e. the transpose
    copy itself costs ~26 ms and consumes nearly the whole 28 ms GEMM win.

So the opportunity is real but gated on repacking cost. Copying 84 MB read + 84 MB write should take
~0.6 ms at this card's bandwidth; torch's elementwise path is 40x off that because it is a transposed
gather. A tiled Triton transpose goes through LDS and can be coalesced on both sides. This script
measures three arms end to end, with every byte of work inside the timed region:

  baseline      production layout, nothing else
  torch_repack  production view .contiguous() then the packed GEMM chain
  triton_repack tiled Triton transpose to the packed layout then the packed GEMM chain

Correctness is checked against the baseline output for each arm. Run under the admission wrapper.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/code/triton/python")
import torch
import triton
import triton.language as tl

B, H, D = 1, 32, 128
Q_TOK, K_TOK, QCS = 5120, 10287, 1024
FLOPS = 2 * 2 * H * Q_TOK * K_TOK * D
SCALE = D ** -0.5


@triton.jit
def repack(K, Out, N, sb, sh, sn, sd, DN: tl.constexpr, NH: tl.constexpr,
           BD: tl.constexpr, BN: tl.constexpr):
    """Tiled transpose straight from the projection's [B, N, H, D] buffer -> packed [BH, D, N].

    Reading the original buffer (not a pre-reshaped copy) keeps the source reorganisation
    inside the measured arm, which is the whole point of the comparison.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // NH
    h = bh % NH
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = pid_d * BD + tl.arange(0, BD)
    src = K + b * sb + h * sh + offs_n[:, None] * sn + offs_d[None, :] * sd
    tile = tl.load(src, mask=offs_n[:, None] < N, other=0.0)
    dst = Out + bh * (DN * N) + offs_d[:, None] * N + offs_n[None, :]
    tl.store(dst, tl.trans(tile), mask=offs_n[None, :] < N)


def timed(fn, inner=1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / inner


def clock_ramp(seconds=8.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a) * 1e-4
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def chain(q_h, key_t, v_h, out):
    for c0 in range(0, Q_TOK, QCS):
        c1 = min(c0 + QCS, Q_TOK)
        scores = torch.baddbmm(
            torch.empty(H, c1 - c0, K_TOK, device="cuda", dtype=torch.float16),
            q_h[:, c0:c1], key_t, alpha=SCALE, beta=0)
        torch.softmax(scores, dim=-1, out=scores)
        torch.bmm(scores, v_h, out=out[:, c0:c1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--BD", type=int, default=32)
    ap.add_argument("--BN", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    clock_ramp()
    g = torch.Generator(device="cuda").manual_seed(31)
    q_src = torch.randn(B, Q_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    k_src = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    v_src = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)

    kh = k_src.reshape(B, K_TOK, H, D)
    key_strided = kh.permute(0, 2, 3, 1).reshape(B * H, D, K_TOK)          # what ships
    q_h = q_src.reshape(B, Q_TOK, H, D).permute(0, 2, 1, 3).reshape(B * H, Q_TOK, D).contiguous()
    v_h = v_src.reshape(B, K_TOK, H, D).permute(0, 2, 1, 3).reshape(B * H, K_TOK, D).contiguous()
    # source view for the Triton repack: [BH, N, D] with the projection's strides
    packed = torch.empty(B * H, D, K_TOK, device="cuda", dtype=torch.float16)
    out = torch.empty(H, Q_TOK, D, device="cuda", dtype=torch.float16)

    def triton_repack():
        grid = (triton.cdiv(K_TOK, args.BN), D // args.BD, B * H)
        # k_src is the raw [B, N, H, D] projection output: batch, head, token, dim strides
        repack[grid](k_src, packed, K_TOK,
                     sb=K_TOK * H * D, sh=D, sn=H * D, sd=1,
                     DN=D, NH=H, BD=args.BD, BN=args.BN, num_warps=4)
        return packed

    def arm_baseline():
        chain(q_h, key_strided, v_h, out)

    def arm_torch():
        chain(q_h, key_strided.contiguous(), v_h, out)

    def arm_triton():
        chain(q_h, triton_repack(), v_h, out)

    arms = [("baseline", arm_baseline), ("torch_repack", arm_torch),
            ("triton_repack", arm_triton)]

    rep = {"shapes": {"H": H, "D": D, "q_tokens": Q_TOK, "k_tokens": K_TOK, "QCS": QCS,
                      "BD": args.BD, "BN": args.BN},
           "flops": FLOPS, "step_calls": 32, "step_ms_reference": 9616.8,
           "identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                        "triton": triton.__version__},
           "arms": {}}

    # correctness first: each arm's output vs the baseline's
    results = {}
    for name, fn in arms:
        out.zero_()
        fn()
        torch.cuda.synchronize()
        results[name] = out.clone()
    for name, fn in arms:
        rep["arms"].setdefault(name, {})
        rep["arms"][name]["max_abs_vs_baseline"] = float(
            (results[name].float() - results["baseline"].float()).abs().max())

    order = [a for pair in (arms, arms, arms) for a in pair]
    for name, fn in order:
        for _ in range(args.warmup):
            timed(fn)
        s = sorted(timed(fn) for _ in range(args.reps))
        med = s[args.reps // 2]
        a = rep["arms"][name]
        a.setdefault("medians_ms", []).append(round(med, 3))
        a.setdefault("samples_ms", []).append([round(x, 3) for x in s])
        print(f"{name:14s} median {med:8.3f} ms  {FLOPS / (med / 1000) / 1e12:6.2f} TFLOPs",
              flush=True)

    base = max(rep["arms"]["baseline"]["medians_ms"])
    for name, _ in arms:
        if "medians_ms" not in rep["arms"][name]:
            continue
        med = max(rep["arms"][name]["medians_ms"])
        rep["arms"][name]["median_ms"] = round(med, 3)
        rep["arms"][name]["tflops"] = round(FLOPS / (med / 1000) / 1e12, 2)
        rep["arms"][name]["call_saved_ms"] = round(base - med, 3)
        rep["arms"][name]["step_saved_ms"] = round((base - med) * 32, 1)
        rep["arms"][name]["step_pct"] = round((base - med) * 32 / 9616.8 * 100, 2)
        print(f"{name:14s} saved {rep['arms'][name]['call_saved_ms']:7.3f} ms/call -> "
              f"{rep['arms'][name]['step_saved_ms']:8.1f} ms/step "
              f"({rep['arms'][name]['step_pct']}% of step)")

    json.dump(rep, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
