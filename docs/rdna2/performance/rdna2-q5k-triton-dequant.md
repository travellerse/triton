# Fused Triton Q5_K dequant: 10.7-16.9x over the torch path, bit-exact

Date 2026-10-19. Admitted runs `p15-q5k-exact*`, `p16-q5k-bench` (`EXCLUSIVE_COOPERATIVE_LOCK`),
gfx1030 emitted target, workload Triton 3.7.1. Kernel + validation + bench:
`docs/rdna2/performance/q5k_triton_dequant.py`; raw record
`docs/rdna2/performance/artifacts/q5k_bench.json`. Sized first in
`rdna2-dequant-baseline.md` (torch path: 26-30 GB/s vs 206-441 GB/s for a copy of the same bytes).

## Result

Best tile is `BPP=4, num_warps=4`; min-of-7 after an 8 s clock ramp.

| weight | torch | Triton | speedup | Triton rate |
|---|---|---|---|---|
| gate_up 24576×4096 | 10.298 ms | **0.965 ms** | 10.7× | 280 GB/s |
| down 4096×12288 | 5.163 ms | **0.468 ms** | 11.0× | 289 GB/s |
| q/k/v/out 4096×4096 | 1.504 ms | **0.089 ms** | **16.9×** | 506 GB/s |

The kernel now runs at or above the copy reference measured earlier, so the conversion is
bandwidth-bound rather than instruction-bound — which is the point at which further tuning stops
paying. Attributed per step at production call counts (32 blocks × gate_up + down, 4 × 32
projections): **≈629 ms/step ≈ 6.5%** recoverable, on top of and independent from the attention
operand-layout win.

## Exactness

`0/16` failures. Bit-exact against `dequantize_blocks_Q5_K` for **fp16 and bf16**, over block counts
`1, 3, 5, 8, 9, 17, 64, 100, 256, 1023, 4096, 4097` — deliberately including counts that are not
multiples of the tile, so the tail mask is exercised, and both in "realistic finite" mode (fp16
`d`/`dmin` drawn from a normal distribution) and hostile random-bit mode where `d`/`dmin` become NaN
or Inf.

Two things that had to be got right, both of which would have silently changed the model's weights:

* **Torch rounds three times, not once.** Its CUDA kernels compute `half` arithmetic in fp32 (opmath)
  but materialise every tensor op in the output dtype, so `d*sc`, `d*q` and the subtraction each round
  to fp16/bf16. A single-round fused expression gives a *different, plausible-looking* result. The
  kernel reproduces the three-step rounding sequence explicitly.
* **NaN-vs-NaN equality is false.** `torch.equal` reports a mismatch where both sides produced NaN,
  which made the first run look like a real defect. The comparison is now NaN-position-aware, and the
  bf16 case is a genuine dtype bug it caught (the kernel had hardcoded fp16 rounding); GGUF always
  stores `d`/`dmin` as fp16 and *then* widens, which is now what the kernel does.

## Status

Validated as a kernel, **not yet wired into the GGUF plugin's `cast_bias_weight` path**. Before that
counts as shipped it must be measured inside the real call with the A/B/A protocol, with peak VRAM
tracked (memory pressure is precisely what killed weight caching in OPT-036), and the plugin's own
dtype/block-size combinations covered rather than assumed.
