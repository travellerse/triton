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

## What it would be worth

If each Q6_K tensor is used once per decode token, the 128 4096×4096 projections save
`128 × (0.270 − 0.063) ≈ 26 ms` and the 16 4096×12288 down projections `16 × (1.894 − 0.172) ≈
28 ms` versus dequant + matmul. That is arithmetic from the per-shape measurements, not a measured
whole-step number; no end-to-end decode timing was run.

## Limits

* **M=1 only.** Prefill uses a GEMM and would need each decoded block reused across M rows.
* BN=4 and 4 warps are the tuned choice; the sweep covered BN 4/8/16/32 and 4/8 warps at
  4096², not deeper block-level pipelines.
* The comparison arm is the Triton dequant kernel plus torch matmul, not the plugin's full dispatch.
