# Softmax materialization on RDNA2: production already sits at the good point, and the fp32 option is a trap

Status: **MEASURED** (one admitted run, tag `smode2`, A/A **0.46%**), plus a **NEGATIVE/NO
MATERIAL IMPROVEMENT** finding about what is left to tune. Date: 2026-10-19.
Harness: [rdna2_fa_prod_ab.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_fa_prod_ab.py)
Artifacts: [fa_smode2.log](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/fa_smode2.log),
[fa_smode2.json](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/fa_smode2.json).

## Why this was the right target after the fused-FA regression

`rdna2-fused-fa-loses-on-rdna2.md` showed that fusing cannot beat the library GEMMs on RDNA2.
The remaining attention cost that is *not* a GEMM is the softmax, which is pure bandwidth: the
production profile put it at 24 ms of a ~71 ms call. This run measures how the cost depends on
how the softmax materializes, at the production geometry
(`heads=32, q=5120, kv=10287, head_dim=128`, fp16, `mask=None`, 862.9 GFLOP/call).

## Results

| arm | median ms | TFLOPs | max rel err | peak VRAM |
| --- | --- | --- | --- | --- |
| fp16 softmax, q=2048 (baseline, repeated last: 92.14) | 92.57 | 9.32 | 1.99e-3 | 2901 MiB |
| **in-place softmax, q=2048** | 92.17 | 9.36 | 1.99e-3 | **1615 MiB** (−44%) |
| **fp32-upcast softmax, q=2048** | **178.57** | 4.83 | 1.99e-3 | **6743 MiB** |
| fp16 softmax, q=1024 | 93.56 | 9.22 | 1.99e-3 | 1607 MiB |
| **in-place softmax, q=1024** | 92.77 | 9.30 | 1.99e-3 | **964 MiB** (−67%) |
| Triton fused FA, 32×32, w4 (for reference) | 182.18 | 4.74 | 4.97e-4 | 313 MiB |

Three readings, in order of usefulness:

1. **The fp32-softmax option costs 1.93× and 4.2× the memory** (178.6 vs 92.2 ms; 6743 vs
   1615 MiB). That is the `attn_precision`/`upcast_attention` path
   (`attention.py:430`, `get_attn_precision`), and on this card the extra cost is not the
   arithmetic, it is the traffic: materializing the score matrix in fp32 and writing an fp16
   copy back adds ~5.4 GB of VRAM movement per call. With 32 attention calls per step, that is
   a ~2.8 s/step penalty — i.e. choosing "high" attention precision on RDNA2 is roughly as
   expensive as replacing the whole attention with a fused Triton kernel (182 ms × 32 ≈ 5.8 s,
   same order). **Recommendation: never enable fp32 attention precision on RDNA2; if accuracy
   is the goal, use the fused kernel instead, which is both the same speed *and* 4× more
   accurate** (rel err 4.97e-4 vs 1.99e-3) — that is the one honest selling point the fused
   route has.
2. **In-place softmax is free on time and cuts peak VRAM by 44%** (92.17 vs 92.57 ms, inside
   the 0.46% A/A band, so "no material time change"; −1286 MiB is real). Combined with
   `query_chunk_size=1024`, peak drops to 964 MiB (−67% vs the fp16/q=2048 baseline) for
   +0.2% time.
3. **Production is already at that point.** `comfy/ldm/qwen_image21/model.py:239/250` calls
   `optimized_attention(..., query_chunk_size=1024, inplace_softmax=True,
   fused_attention=qwen21_fused_attention)`. So there is **no remaining win to harvest here** —
   this run validates the existing configuration against the alternatives rather than
   proposing a new one, and confirms the fused-attention hook is being offered a target it must
   beat (92 ms/call, 964 MiB), which today's Triton FA does not.

## What is left in the attention call

The only measured lever with headroom is still **operand layout**, not fusion and not softmax:
`rdna2-key-repack-improvement.md` measured 111.0 → 78.5 ms/call (1.417×) by packing K and making
Q contiguous, gated behind `Q21_TRITON_REPACK_KEY` and still default-off because the whole-step
image gate is BLOCKED. Against the production numbers in this table, that candidate is ~15% of
the call; the fused-attention route is a dead end on this hardware at `head_dim=128`.
