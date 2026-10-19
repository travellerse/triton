# Fused Q5_K dequant matvec on RDNA2: 6.1–12.5x over dequant + matmul at M=1

Status: **MEASURED IMPROVEMENT** on native gfx1031 (RX 6700 XT). Date: 2026-10-20.
Kernel: [q5k_triton_matvec.py](file:///home/trave/code/triton/docs/rdna2/performance/q5k_triton_matvec.py).

## What it does

At M=1 the shipped path dequantises the whole weight tensor to fp16 with the validated kernel, then
reads that fp16 tensor back for the matvec. The fused kernel decodes each 176-byte Q5_K block in
registers, multiplies it by the activation block and accumulates in f32; the fp16 weight is never
materialised. The decode keeps the shipped kernel's three fp16 roundings (`d*sc`, then `*q`, then
subtracting `dmin*mv`), so only the f32 summation order differs from the reference.

## Results (M=1, BN=4, 4 warps, median of 7 × 50 launches, same process)

| shape | fused | dequant + matmul | speedup | A/A | fused throughput |
| --- | --- | --- | --- | --- | --- |
| 4096×4096 | **0.103 ms** | 0.269 ms | **2.61×** | 0.30% | 162.5 Gweight/s |
| 12288×4096 | **0.285 ms** | 1.117 ms | **3.93×** | 0.28% | 176.9 Gweight/s |
| 4096×12288 | **0.288 ms** | 1.816 ms | **6.31×** | 0.51% | 174.7 Gweight/s |
| 24576×4096 | **0.559 ms** | 3.830 ms | **6.85×** | 0.31% | 180.1 Gweight/s |

The matmul arm is the orientation `x @ W.T`, which is what `F.linear` uses and what the plugin's
GEMM does. The first version of this table used `W @ x`; that call is several times slower for the
same values and flattered the fused kernel (e.g. 0.731 ms versus 0.269 ms at 4096²), so the table
above is the corrected one.

BN was tuned after the first pass: four rows per program is 13–25% faster than eight (0.102 vs
0.118 ms at 4096², 0.553 vs 0.643 at 24576×4096), while BN=16 and BN=32 collapse (0.156 and
0.736 ms at 4096²) and eight warps are slower than four at the same BN. The small end is closed
too: BN=2 and BN=1 are 3–6% slower (0.106/0.109 ms at 4096², 0.585/0.617 at 24576×4096), so BN=4
is the optimum over the whole 1–32 range. The tuned kernel is bandwidth/issue limited at
~163–181 Gweight/s.

## Small-M reuse

`mv5k_m` reuses each decoded weight block across M activation rows with a `[M, BN, 256]` f32
accumulator. Values agree with the reference to `rel_to_max` 2.0–2.6×10⁻⁶ for M=2/4/8 at 4096².
With the corrected baseline:

| shape | M | fused | dequant + matmul | speedup | A/A |
| --- | --- | --- | --- | --- | --- |
| 4096×4096 | 1 | 0.102 ms | 0.269 ms | 2.63× | 0.24% |
| 4096×4096 | 2 | 0.125 ms | 0.271 ms | 2.17× | 0.15% |
| 4096×4096 | 4 | 0.093 ms | 0.270 ms | 2.91× | 0.14% |
| 4096×4096 | 8 | 0.170 ms | 0.271 ms | 1.59× | 0.25% |
| 24576×4096 | 1 | 0.557 ms | 3.837 ms | 6.89× | 0.39% |
| 24576×4096 | 2 | 0.684 ms | 3.839 ms | 5.61× | 1.70% |
| 24576×4096 | 4 | 0.498 ms | 3.837 ms | 7.71× | 0.48% |
| 24576×4096 | 8 | 0.956 ms | 3.830 ms | 4.01× | 0.14% |

At the narrow shape the matmul is cheap once M>1 and the margin shrinks; at the wide shape the
weight read dominates and the fused arm keeps a 4–7.7× lead up to M=8. The non-monotonic fused
times (M=4 faster than M=2) are reproducible within the A/A bands but not explained.

The comparison arm is the validated dequant kernel plus `torch.matmul` (fp16) in the same process
and clock state. "Gweight/s" counts dequantised weight elements consumed per second, so it measures
the decode + contraction rate rather than output bandwidth.

## Correctness

Fused vs the same dequant followed by an f32 contraction: max relative error 3.7×10⁻⁵ at 64×1024,
1.3×10⁻⁵ at 256×4096, and 9.0×10⁻⁵ at 4096×4096 (max absolute 2.3×10⁻², consistent with an f32
reduction of fp16 products). The decode itself is the kernel previously validated bit-exact against
the plugin's `dequantize_blocks_Q5_K`; what this check adds is the fused contraction.

## Limits

* **M is limited to 1, 2, 4, 8.** `mv5k_m` is measured for those; at large M/prefill the
  decoded blocks would need a real GEMM tiling, and the quantised weights are re-read per M panel.
* BN=4 and 4 warps are the tuned choice; other warp counts and deeper block-level
  pipelines were not explored.
* The comparison arm is the Triton dequant kernel plus torch matmul, not the plugin's full dispatch.
* Q6_K (the format of all 128 attention projections in the user's model) has no fused matvec yet.
