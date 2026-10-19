"""A/B whether a spill-free tile beats a spilling one on gfx1030 (on-device).

Background: docs/rdna2/performance/rdna2-fast-attn-is-register-bound.md established
statically that RDNA2's FMA dot path spills VGPRs at almost every attention-sized tile, and
that the earlier flash-attention config sweep on this host used *only* spilling configs
(128x64x4, 128x128x4, 64x128x4, 128x64x8, 256x64x8, 64x64x4). So the register-budget
hypothesis was never tested. This script tests it: for a matched pair of tiles that differ
mainly in spill count, measure the FA inner-loop time.

Protocol (the same one used by the earlier measured wins):
  * 8 s clock ramp before any timed bracket (the card idles at 500 MHz),
  * ~BATCH launches inside each timed bracket, so per-launch overhead does not dominate,
  * median over reps, arms interleaved, baseline arm repeated to size the A/A band,
  * correctness: each config must reproduce the reference attention within tolerance,
  * peak VRAM reported per config, because a spill-free tile that needs more scratch is
    the failure mode worth seeing.

Run ONLY through the admission wrapper:
  python3 docs/rdna2/admission/gpu_admission.py --acquire run -- \
      python3 docs/rdna2/performance/rdna2_spill_ab.py --configs 64,64,128,4:32,32,128,4
"""

import argparse
import json
import statistics
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def fa_fwd(Q, K, V, Out, SQ, SK, sm_scale,
           BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)

    q = tl.load(Q + bh * SQ * HD + offs_m[:, None] * HD + offs_d[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)

    for start_n in range(0, SK, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(K + bh * SQ * HD + offs_n[None, :] * HD + offs_d[:, None])
        s = tl.dot(q, k) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + bh * SQ * HD + offs_n[:, None] * HD + offs_d[None, :])
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(Out + bh * SQ * HD + offs_m[:, None] * HD + offs_d[None, :],
             acc.to(tl.float16))


def reference(q, k, v):
    """Per-head fp32 softmax(QK^T)V.

    Head-at-a-time on purpose: a whole-batch fp32 score matrix for 16 heads x 4096 tokens
    is ~2 GiB, which would make the measurement reserve more VRAM than the kernel itself
    needs and could evict the user's workload.
    """
    out = torch.empty(q.shape, device=q.device, dtype=torch.float16)
    scale = q.shape[-1] ** -0.5
    for h in range(q.shape[0]):
        s = (q[h].float() @ k[h].float().transpose(0, 1)) * scale
        p = torch.softmax(s, dim=-1)
        out[h] = (p @ v[h].float()).half()
        del s, p
    return out


def timed(fn, batch, reps, ramp_s):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    time.sleep(ramp_s)
    out = []
    for _ in range(reps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(batch):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / batch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True,
                    help="colon-separated BM,BN,HD,W lists; first is the baseline arm")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--ramp", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no device"
    torch.cuda.empty_cache()
    S, H, D = args.tokens, args.heads, None
    cfgs = [tuple(int(x) for x in c.split(",")) for c in args.configs.split(":")]

    qkv = {}
    for (bm, bn, hd, w) in cfgs:
        if hd not in qkv:
            torch.manual_seed(0)
            qkv[hd] = tuple(torch.randn(H, S, hd, device="cuda", dtype=torch.float16)
                            for _ in range(3))

    results = {}
    for idx, (bm, bn, hd, w) in enumerate(cfgs):
        q, k, v = qkv[hd]
        out = torch.empty(H, S, hd, device="cuda", dtype=torch.float16)
        grid = (triton.cdiv(S, bm), H)
        sm = 1.4426950408889634 * (hd ** -0.5)

        def launch():
            fa_fwd[grid](q, k, v, out, S, S, sm, BM=bm, BN=bn, HD=hd,
                         num_warps=w, num_stages=1)

        launch()
        torch.cuda.synchronize()
        ref = reference(q, k, v)
        err = (out.float() - ref.float()).abs().max().item()
        rel = err / ref.float().abs().max().item()
        torch.cuda.reset_peak_memory_stats()
        ms = timed(launch, args.batch, args.reps, args.ramp)
        med = statistics.median(ms)
        results[f"{idx}:{bm}x{bn}x{hd}w{w}"] = {
            "median_ms": med, "all_ms": ms, "max_abs_err": err, "rel_err": rel,
            "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
            "arm_index": idx,
        }
        print(f"arm {idx}: {bm:>4}x{bn:<4} hd={hd:<4} w={w}  median {med:7.3f} ms  "
              f"min {min(ms):7.3f}  rel_err {rel:.2e}  "
              f"peak {torch.cuda.max_memory_allocated()/2**20:7.1f} MiB", flush=True)

    names = list(results)
    base = results[names[0]]
    # The caller repeats the first config at the end to size the A/A band; without that
    # band no ratio below can be called an improvement.
    aa_spread = None
    if len(names) > 1 and cfgs[0] == cfgs[-1]:
        a, b = results[names[0]]["median_ms"], results[names[-1]]["median_ms"]
        aa_spread = abs(a - b) / min(a, b)
        print(f"\nA/A band (arm 0 repeated last): {a:.3f} vs {b:.3f} ms -> "
              f"{aa_spread * 100:.2f}% spread", flush=True)
    print("\n== ratios vs first arm ==")
    for n in names:
        r = results[n]
        line = f"  {n:22s} {r['median_ms']:7.3f} ms  x{base['median_ms']/r['median_ms']:.3f}"
        print(line, flush=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results, "aa_spread": aa_spread,
                   "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(0)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
