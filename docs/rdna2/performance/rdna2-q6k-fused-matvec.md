# Fused Q6_K dequant matvec on RDNA2: 4.0–7.0x over dequant + matmul at M=1

Status: **MEASURED IMPROVEMENT** on native gfx1031 (RX 6700 XT). Date: 2026-10-20.
Kernel: [q6k_triton_matvec.py](file:///home/trave/code/triton/docs/rdna2/performance/q6k_triton_matvec.py).
Companion: [rdna2-q5k-fused-matvec.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-q5k-fused-matvec.md).

## What it does

The same fusion as the Q5_K kernel, for the format of all 128 attention projections and 16 of the
down projections: decode each 210-byte Q6_K block in registers, multiply it by the activation block
and accumulate in f32, with the shipped kernel's two fp16 roundings. The materialised fp16 weight is
never written or read.

## Results (M=1, BN=8, 4 warps, median of 7 × 50 launches, same process)

| shape | fused | dequant + matmul | speedup | A/A | fused throughput |
| --- | --- | --- | --- | --- | --- |
| 4096×4096 | **0.179 ms** | 0.728 ms | **4.07×** | 1.72% | 93.9 Gweight/s |
| 4096×12288 | **0.518 ms** | 2.064 ms | **3.98×** | 0.39% | 97.1 Gweight/s |
| 24576×4096 | **0.904 ms** | 6.283 ms | **6.95×** | 3.25% | 111.4 Gweight/s |

The Q6_K decode is heavier per weight than Q5_K (five bit-field extractions versus four plus the
5th-bit gather), which is why the throughput is ~95–111 Gweight/s instead of ~150. The 24576×4096
row's A/A band is 3.25%, larger than the other rows but still well below the effect.

## Correctness

Fused vs the shipped dequant followed by an f32 contraction: max relative error 9.4×10⁻⁶ at
64×1024 and 256×4096, and 8.2×10⁻⁴ at 4096×4096 (max absolute 4.7×10⁻² — the reference contracts
fp16 products whose q range is ±32). The decode is the kernel previously validated bit-exact
against the plugin's `dequantize_blocks_Q6_K`.

## What it would be worth

If each Q6_K tensor is used once per decode token, the 128 4096×4096 projections save
`128 × (0.728 − 0.179) ≈ 70 ms` and the 16 4096×12288 down projections `16 × (2.064 − 0.518) ≈
25 ms` versus dequant + matmul. That is arithmetic from the per-shape measurements, not a measured
whole-step number; no end-to-end decode timing was run.

## Limits

* **M=1 only.** Prefill uses a GEMM and would need each decoded block reused across M rows.
* BN=8 and 4 warps are untuned for this kernel.
* The comparison arm is the Triton dequant kernel plus torch matmul, not the plugin's full dispatch.
