"""Does an RDNA2-tuned Triton flash-attention beat the path ComfyUI uses today?

The measured register-budget result (rdna2-spill-cost-measured.md) says RDNA2 attention wants
small tiles at the *smallest warp count that keeps per-lane work*, not the spill-free point.
This takes that conclusion to the production geometry profiled earlier:

    q_tokens=5120, kv_tokens=10287, heads=32, head_dim=128, fp16, mask=None

(the Qwen-Image-2.1 DiT call with a prefix-cached text block, so kv_tokens != q_tokens and
the key tail needs masking).

Arms, all measured in one admitted run so they share a clock state:

  * subquad:<qchunk> -- the chunked baddbmm path ComfyUI's attention_sub_quad actually
    executes (one query chunk at a time, the whole key set per chunk). The chunk ladder in
    attention.py:453 picks the size from free VRAM, so it is passed explicitly here for
    determinism. An arm that cannot allocate is recorded as OOM instead of crashing the run:
    at this geometry the *unchunked* torch path needs 6.28 GiB on a 12 GiB card, which is
    exactly why ComfyUI chunks at all.
  * Triton FA at several (BM, BN, warps) using the tile sizes the spill table recommends.

Every arm is checked against a per-head fp32 reference before it is timed.

Run ONLY through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --acquire --budget-mib 4000 --tag faprod run -- \
      python3 -u docs/rdna2/performance/rdna2_fa_prod_ab.py --out /tmp/fa_prod.json
"""

import argparse
import json
import math
import statistics
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def fa_prod(Q, K, V, Out, QN, KN, scale,
            BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)

    q = tl.load(Q + bh * QN * HD + offs_m[:, None] * HD + offs_d[None, :],
                mask=offs_m[:, None] < QN, other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)

    for start_n in range(0, KN, BN):
        offs_n = start_n + tl.arange(0, BN)
        n_ok = offs_n < KN
        k = tl.load(K + bh * KN * HD + offs_n[None, :] * HD + offs_d[:, None],
                    mask=n_ok[None, :], other=0.0)
        s = tl.dot(q, k) * scale
        s = tl.where(n_ok[None, :], s, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + bh * KN * HD + offs_n[:, None] * HD + offs_d[None, :],
                    mask=n_ok[:, None], other=0.0)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(Out + bh * QN * HD + offs_m[:, None] * HD + offs_d[None, :],
             acc.to(tl.float16), mask=offs_m[:, None] < QN)


def subquad(q, k, v, out, qchunk, mode="upcast"):
    """softmax(QK^T)V one query chunk at a time, whole key set per chunk, i.e. the shape of
    ComfyUI's attention_sub_quad when a chunk size is affordable.

    mode selects how the softmax materializes, because that step is bandwidth-bound (the
    production profile put it at 24 ms of the ~71 ms call) and the fused-GEMM route is not:
      upcast  -- softmax(att.float()).half()  (fp32 score copy, what ComfyUI does when
                 attn_precision is fp32; it is also the reference-accurate case)
      fp16    -- softmax(att) in fp16 (ATen still accumulates in fp32)
      inplace -- att.softmax_(-1), no separate probability tensor
    """
    H, QN, HD = q.shape
    KN = k.shape[1]
    scale = 1.0 / math.sqrt(HD)
    kt = k.transpose(-1, -2)
    for s0 in range(0, QN, qchunk):
        s1 = min(s0 + qchunk, QN)
        qc = q[:, s0:s1]
        att = torch.bmm(qc, kt)
        att *= scale
        if mode == "upcast":
            p = torch.softmax(att.float(), dim=-1).half()
        elif mode == "fp16":
            p = torch.softmax(att, dim=-1)
        else:
            p = att.softmax_(-1)
        out[:, s0:s1] = torch.bmm(p, v)
        del att, p


def reference(q, k, v):
    """fp32 softmax(QK^T)V head-at-a-time; q [H,QN,HD], k/v [H,KN,HD]."""
    out = torch.empty(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for h in range(q.shape[0]):
        s = (q[h].float() @ k[h].float().transpose(0, 1)) * scale
        p = torch.softmax(s, dim=-1)
        out[h] = (p @ v[h].float()).half()
        del s, p
    return out


def bracket(fn, batch, reps, ramp):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    time.sleep(ramp)
    out = []
    for _ in range(reps):
        st = torch.cuda.Event(True)
        en = torch.cuda.Event(True)
        st.record()
        for _ in range(batch):
            fn()
        en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en) / batch)
    return out


def rel_err(a, ref):
    d = (a.float() - ref.float()).abs().max().item()
    return d / ref.float().abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-tokens", type=int, default=5120)
    ap.add_argument("--kv-tokens", type=int, default=10287)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--arms", default="subquad_fp16:2048,subquad_inplace:2048,"
                                     "subquad_upcast:2048,subquad_fp16:1024,"
                                     "subquad_inplace:1024,32x32x4,subquad_fp16:2048")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--ramp", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no device"
    H, QN, KN, HD = args.heads, args.q_tokens, args.kv_tokens, args.head_dim
    torch.manual_seed(0)
    q = torch.randn(H, QN, HD, device="cuda", dtype=torch.float16)
    k = torch.randn(H, KN, HD, device="cuda", dtype=torch.float16)
    v = torch.randn(H, KN, HD, device="cuda", dtype=torch.float16)
    ref = reference(q, k, v)
    flops = 4.0 * H * QN * KN * HD
    free, total = torch.cuda.mem_get_info()
    print(f"H={H} QN={QN} KN={KN} HD={HD}  {flops/1e9:.1f} GFLOP/call  "
          f"ref_max={ref.float().abs().max().item():.3f}  "
          f"vram free={free/2**20:.0f} MiB total={total/2**20:.0f} MiB", flush=True)

    results = {}
    arms = args.arms.split(",")
    for idx, arm in enumerate(arms):
        out = torch.zeros(H, QN, HD, device="cuda", dtype=torch.float16)

        if arm.startswith("subquad"):
            body = arm.split(":", 1)
            qchunk = int(body[1])
            mode = body[0].split("_", 1)[1] if "_" in body[0] else "fp16"

            def run(qc=qchunk, md=mode, o=out):
                subquad(q, k, v, o, qc, md)
        else:
            bm, bn, w = (int(x) for x in arm.split("x"))
            grid = (triton.cdiv(QN, bm), H)
            sc = 1.4426950408889634 / math.sqrt(HD)

            def run(bm=bm, bn=bn, w=w, grid=grid, sc=sc, o=out):
                fa_prod[grid](q, k, v, o, QN, KN, sc, BM=bm, BN=bn, HD=HD,
                              num_warps=w, num_stages=1)

        try:
            run()
            torch.cuda.synchronize()
            err = rel_err(out, ref)
            torch.cuda.reset_peak_memory_stats()
            ms = bracket(run, args.batch, args.reps, args.ramp)
            med = statistics.median(ms)
            results[f"{idx}:{arm}"] = {"median_ms": med, "all_ms": ms, "rel_err": err,
                                       "peak_mib": torch.cuda.max_memory_allocated() / 2**20}
            print(f"arm {idx}: {arm:14s} median {med:8.3f} ms  min {min(ms):8.3f}  "
                  f"{flops/med/1e9:6.2f} TFLOPs  rel_err {err:.2e}  "
                  f"peak {results[f'{idx}:{arm}']['peak_mib']:7.1f} MiB", flush=True)
        except torch.OutOfMemoryError as e:
            results[f"{idx}:{arm}"] = {"oom": str(e).splitlines()[0][:200]}
            print(f"arm {idx}: {arm:14s} OOM (not runnable at this geometry)", flush=True)
            torch.cuda.empty_cache()
        del out

    keys = list(results)
    first, last = keys[0].split(":", 1)[1], keys[-1].split(":", 1)[1]
    if first == last and "median_ms" in results[keys[0]] and "median_ms" in results[keys[-1]]:
        a = results[keys[0]]["median_ms"]
        b = results[keys[-1]]["median_ms"]
        print(f"\nA/A band: {a:.3f} vs {b:.3f} ms -> {abs(a-b)/min(a,b)*100:.2f}%", flush=True)
    timed = {k: v["median_ms"] for k, v in results.items() if "median_ms" in v}
    if timed:
        best = min(timed, key=lambda kk: (timed[kk], kk))
        print(f"fastest arm: {best} at {timed[best]:.3f} ms "
              f"({flops/timed[best]/1e9:.2f} TFLOPs)", flush=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "gflop_per_call": flops / 1e9,
                   "vram_free_mib_at_start": free / 2**20,
                   "results": results, "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(0)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
