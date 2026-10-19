# Fused Q5_K dequant matvec on RDNA2: 6.1–12.5x over dequant + matmul at M=1

Status: **MEASURED IMPROVEMENT** on native gfx1031 (RX 6700 XT). Date: 2026-10-20.
Kernel: [q5k_triton_matvec.py](file:///home/trave/code/triton/docs/rdna2/performance/q5k_triton_matvec.py).

## What it does

At M=1 the shipped path dequantises the whole weight tensor to fp16 with the validated kernel, then
reads that fp16 tensor back for the matvec. The fused kernel decodes each 176-byte Q5_K block in
registers, multiplies it by the activation block and accumulates in f32; the fp16 weight is never
materialised. The decode keeps the shipped kernel's three fp16 roundings (`d*sc`, then `*q`, then
subtracting `dmin*mv`), so only the f32 summation order differs from the reference.

## Results (M=1, BN=8, 4 warps, median of 7 × 50 launches, same process)

| shape | fused | dequant + matmul | speedup | A/A | fused throughput |
| --- | --- | --- | --- | --- | --- |
| 4096×4096 | **0.118 ms** | 0.728 ms | **6.17×** | 0.41% | 142.2 Gweight/s |
| 12288×4096 | **0.319 ms** | 3.992 ms | **12.50×** | 0.44% | 157.6 Gweight/s |
| 4096×12288 | **0.333 ms** | 2.031 ms | **6.09×** | 0.50% | 151.0 Gweight/s |
| 24576×4096 | **0.643 ms** | 6.273 ms | **9.76×** | 1.28% | 156.6 Gweight/s |

The comparison arm is the validated dequant kernel plus `torch.matmul` (fp16) in the same process
and clock state. "Gweight/s" counts dequantised weight elements consumed per second, so it measures
the decode + contraction rate rather than output bandwidth.

## Correctness

Fused vs the same dequant followed by an f32 contraction: max relative error 3.7×10⁻⁵ at 64×1024,
1.3×10⁻⁵ at 256×4096, and 9.0×10⁻⁵ at 4096×4096 (max absolute 2.3×10⁻², consistent with an f32
reduction of fp16 products). The decode itself is the kernel previously validated bit-exact against
the plugin's `dequantize_blocks_Q5_K`; what this check adds is the fused contraction.

## Limits

* **M=1 only.** This is the decode-shaped case. A prefill GEMM would have to reuse each decoded
  weight block across M rows; that variant is not implemented, and at large M the quantised weights
  would be re-read per M panel.
* BN=8 and 4 warps are untuned; no tile sweep was run for this kernel.
* The comparison arm is the Triton dequant kernel plus torch matmul, not the plugin's full dispatch.
* Q6_K (the format of all 128 attention projections in the user's model) has no fused matvec yet.
