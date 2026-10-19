#!/usr/bin/env python3
"""Why does the packed QK^T cost 24 ms isolated and 38.8 ms inside the real call?

Three candidate causes, tested against each other with everything else fixed (same packed key,
same query, same shapes, same chunking):

  alloc_per_chunk   what production does: torch.empty of a fresh (32,1024,10287) fp16 buffer,
                    675 MB, for every chunk
  alloc_reused      same call writing into one buffer allocated once
  alpha_is_one      same as alloc_per_chunk but with the 1/sqrt(D) scale dropped
  plain_mm          mm instead of baddbmm
  repack_adjacent   repack the key immediately before the GEMM, so the packed buffer is
                    written and then read back-to-back (cache state, not allocation)

If the difference is allocation, reusing the score buffer is a small, safe change worth ~15 ms
per call. If it is cache state, it is not fixable at the call site and should be stated as such.
Run under admission.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/code/triton/python")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
import torch
import triton
import triton.language as tl

H, D = 32, 128
Q_TOK, K_TOK, QCS = 5120, 10287, 1024
CHUNKS = -(-Q_TOK // QCS)
SCALE = D ** -0.5
FLOPS_QK = 2.0 * H * Q_TOK * K_TOK * D


def clock_ramp(seconds=8.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a) * 1e-4
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


@triton.jit
def repack(K, Out, N, s0, s1, s2, sob, BD: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = pid_d * BD + tl.arange(0, BD)
    tile = tl.load(K + bh * s0 + offs_d[None, :] * s1 + offs_n[:, None] * s2,
                   mask=offs_n[:, None] < N, other=0.0)
    tl.store(Out + bh * sob + offs_d[:, None] * N + offs_n[None, :],
             tl.trans(tile), mask=offs_n[None, :] < N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    clock_ramp()
    g = torch.Generator(device="cuda").manual_seed(17)
    q = torch.randn(H, Q_TOK, D, generator=g, device="cuda", dtype=torch.float16)
    k_strided = torch.randn(H, K_TOK, D, generator=g, device="cuda", dtype=torch.float16).transpose(1, 2)
    key_packed = k_strided.contiguous()
    scratch = torch.empty(H, QCS, K_TOK, device="cuda", dtype=torch.float16)

    def gemms(out_buf, fresh, alpha):
        def run():
            for c0 in range(0, Q_TOK, QCS):
                buf = (torch.empty(H, QCS, K_TOK, device="cuda", dtype=torch.float16)
                       if fresh else out_buf)
                torch.baddbmm(buf, q[:, c0:c0 + QCS], key_packed, alpha=alpha, beta=0)
        return run

    def repack_then_gemms():
        packed2 = torch.empty_like(key_packed)
        repack[(triton.cdiv(K_TOK, 128), D // 32, H)](
            k_strided, packed2, K_TOK, k_strided.stride(0), k_strided.stride(1),
            k_strided.stride(2), packed2.stride(0), BD=32, BN=128, num_warps=4)
        for c0 in range(0, Q_TOK, QCS):
            torch.baddbmm(torch.empty(H, QCS, K_TOK, device="cuda", dtype=torch.float16),
                          q[:, c0:c0 + QCS], packed2, alpha=SCALE, beta=0)

    arms = {
        "alloc_per_chunk": gemms(None, True, SCALE),
        "alloc_reused": gemms(scratch, False, SCALE),
        "alpha_is_one": gemms(None, True, 1.0),
        "alpha_is_one_reused": gemms(scratch, False, 1.0),
        "repack_adjacent": repack_then_gemms,
    }

    rep = {"shapes": {"H": H, "D": D, "q_tokens": Q_TOK, "k_tokens": K_TOK, "chunk": QCS,
                      "chunks": CHUNKS},
           "qk_flops": FLOPS_QK,
           "identity": {"gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                        "triton": triton.__version__},
           "arms": {}}
    order = ["alloc_per_chunk", "alloc_reused", "alpha_is_one", "alpha_is_one_reused",
             "repack_adjacent", "alloc_per_chunk", "alloc_reused", "alloc_per_chunk"]
    for name in order:
        fn = arms[name]
        for _ in range(args.warmup):
            timed(fn)
        s = sorted(timed(fn) for _ in range(args.reps))
        med = s[args.reps // 2]
        a = rep["arms"].setdefault(name, {"medians_ms": []})
        a["medians_ms"].append(round(med, 3))
        print(f"{name:22s} median {med:8.3f} ms  QK^T {FLOPS_QK / (med / 1000) / 1e12:6.2f} TFLOPs",
              flush=True)

    for name, a in rep["arms"].items():
        a["best_median_ms"] = min(a["medians_ms"])
        a["tflops_at_best"] = round(FLOPS_QK / (a["best_median_ms"] / 1000) / 1e12, 2)
    base = rep["arms"]["alloc_per_chunk"]["best_median_ms"]
    for name, a in rep["arms"].items():
        a["gain_vs_alloc_per_chunk_ms"] = round(base - a["best_median_ms"], 3)
        a["step_gain_ms_x32calls"] = round((base - a["best_median_ms"]) * 32, 1)
    rep["aa_band_ms"] = max(rep["arms"]["alloc_per_chunk"]["medians_ms"]) - \
        min(rep["arms"]["alloc_per_chunk"]["medians_ms"])
    json.dump(rep, open(args.out, "w"), indent=1)
    print("A/A band on the repeated arm:", round(rep["aa_band_ms"], 3), "ms")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
