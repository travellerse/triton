# Fused Q6_K dequant matvec on RDNA2: 4.0–7.0x over dequant + matmul at M=1

Status: **MEASURED IMPROVEMENT** on native gfx1031 (RX 6700 XT). Date: 2026-10-20.
Kernel: [q6k_triton_matvec.py](file:///home/trave/code/triton/docs/rdna2/performance/q6k_triton_matvec.py).
Companion: [rdna2-q5k-fused-matvec.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-q5k-fused-matvec.md).

## What it does

The same fusion as the Q5_K kernel, for the format of all 128 attention projections and 16 of the
down projections: decode each 210-byte Q6_K block in registers, multiply it by the activation block
and accumulate in f32, with the shipped kernel's two fp16 roundings. The materialised fp16 weight is
never written or read.

## Results (M=1, BN=4, 4 warps, median of 7 × 50 launches, same process)

| shape | fused | dequant + matmul | speedup | A/A | fused throughput |
| --- | --- | --- | --- | --- | --- |
| 4096×4096 | **0.063 ms** | 0.270 ms | **4.28×** | 0.74% | 266.5 Gweight/s |
| 4096×12288 | **0.172 ms** | 1.894 ms | **11.04×** | 0.03% | 293.4 Gweight/s |
| 24576×4096 | **0.333 ms** | 3.877 ms | **11.65×** | 0.20% | 302.5 Gweight/s |

The matmul arm is `x @ W.T` (the `F.linear` orientation); the first version used `W @ x`, which is
several times slower at these shapes and flattered the fused kernel, so these are the corrected
numbers.

BN=4 instead of 8 is decisive for this format: 0.063 vs 0.176 ms at 4096² and 0.332 vs 0.924 at
24576×4096 (2.8×), with the A/A band tightening from 3.9% to 0.43% at the largest shape. At
~270–300 Gweight/s the kernel is close to the card's bandwidth for the 0.82 B/weight format
(≈220–250 GB/s of weight traffic against a 384 GB/s spec).

## Correctness

Fused vs the shipped dequant followed by an f32 contraction: max relative error 9.4×10⁻⁶ at
64×1024 and 256×4096, and 8.2×10⁻⁴ at 4096×4096 (max absolute 4.7×10⁻² — the reference contracts
fp16 products whose q range is ±32). The decode is the kernel previously validated bit-exact
against the plugin's `dequantize_blocks_Q6_K`.

## Small-M reuse

`mv6k_m` reuses each decoded weight block across M activation rows with a `[M, BN, 256]` f32
accumulator. Values agree with the reference to `rel_to_max` 1.6–1.9×10⁻⁶ for M=1..8 at 4096².
With the corrected `x @ W.T` baseline:

| shape | M | fused | dequant + matmul | speedup | A/A |
| --- | --- | --- | --- | --- | --- |
| 4096×4096 | 1 | 0.063 ms | 0.271 ms | 4.30× | 0.49% |
| 4096×4096 | 2 | 0.079 ms | 0.272 ms | 3.44× | 0.28% |
| 4096×4096 | 4 | 0.082 ms | 0.271 ms | 3.30× | 0.16% |
| 4096×4096 | 8 | 0.132 ms | 0.271 ms | 2.05× | 0.36% |
| 24576×4096 | 1 | 0.335 ms | 3.874 ms | 11.57× | 0.20% |
| 24576×4096 | 2 | 0.452 ms | 3.854 ms | 8.53× | 4.33% |
| 24576×4096 | 4 | 0.441 ms | 3.867 ms | 8.76× | 1.32% |
| 24576×4096 | 8 | 0.740 ms | 3.850 ms | 5.20× | 0.69% |

At the narrow shape the plain GEMM is cheap once M>1 and the margin falls to 2.05× by M=8; at the
wide shape the weight read dominates, so the fused arm keeps at least a 5.2× lead. The M=2 row at
24576×4096 has the largest A/A band (4.33%), still far below its effect.

## What it would be worth

If each Q6_K tensor is used once per decode token, the 128 4096×4096 projections save
`128 × (0.270 − 0.063) ≈ 26 ms` and the 16 4096×12288 down projections `16 × (1.894 − 0.172) ≈
28 ms` versus dequant + matmul. That is arithmetic from the per-shape measurements, not a measured
whole-step number; no end-to-end decode timing was run.

## Limits

* **M is limited to 1, 2, 4, 8.** `mv6k_m` is measured for those; large-M prefill needs a
  real GEMM tiling and re-reads the quantised weights per M panel.
* BN=4 and 4 warps are the tuned choice; the sweep covered BN 4/8/16/32 and 4/8 warps at
  4096², not deeper block-level pipelines.
* The comparison arm is the Triton dequant kernel plus torch matmul, not the plugin's full dispatch.
