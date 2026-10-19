# GGUF Q5_K dequantisation is 8-15x off the memory system, not bandwidth-bound

Date 2026-10-19, admitted run (`EXCLUSIVE_COOPERATIVE_LOCK`), gfx1030 emitted target, workload
interpreter. Reproduce: `docs/rdna2/performance/rdna2_dequant_baseline.py`; raw record
`docs/rdna2/performance/artifacts/dequant_baseline.json`.

OPT-036 measured dequant at ~943 ms/step and rejected *caching* the result (the fp16 weights need
13.96 GB against 12.27 GB of VRAM). That leaves the per-step cost standing, but nobody had asked
whether it is already at the hardware limit. It is not.

Timing the plugin's own `dequantize_blocks_Q5_K` at real DiT weight shapes, beside two references on
the identical byte counts — a full fp16 copy (in+out traffic) and a pure fp16 fill (write only):

| weight | dequant | copy of same bytes | fill | dequant rate | copy rate | gap |
|---|---|---|---|---|---|---|
| gate_up 24576×4096 | 10.193 ms | 1.288 ms | 0.613 ms | **26.5 GB/s** | 210.0 GB/s | **7.9×** |
| down 4096×12288 | 5.138 ms | 0.655 ms | 0.155 ms | 26.3 GB/s | 206.4 GB/s | 7.8× |
| q/k/v/out 4096×4096 | 1.499 ms | 0.102 ms | 0.069 ms | 30.1 GB/s | 441.1 GB/s | **14.7×** |

Cost attributable per step, counting 32 blocks × (gate_up + down) plus 4 projections each:
**682.5 ms = 7.1% of the 9616.8 ms step** — and that is only the DiT's `Linear` weights, so it is
consistent with the 943 ms that OPT-036 attributed including other tensors.

## What this means

The torch implementation expands each block through a chain of reshapes, broadcasts, per-nibble
shifts and intermediate fp32/fp16 tensors, so the real work is many small strided elementwise
kernels rather than one pass over memory. Its 26-30 GB/s is roughly an order of magnitude below what
a single fused pass over the same bytes demonstrably achieves (206-441 GB/s here), so the ceiling for
a fused Triton dequant is set by the copy arm, not by the current code.

If a fused kernel reaches even the copy rate, the addressable saving is the difference between the
two columns: ~8.9 ms + ~4.5 ms + ~1.4 ms per call at the shapes above → on the order of
**500-600 ms/step (5-6%)**, comparable in size to the attention operand-layout win and independent of
it. That is what makes this worth a kernel rather than a tuning pass.

## Bar for accepting such a kernel

1. **Bit-exact** output against `dequantize_blocks_Q5_K` for fp16 (and the same for bf16), across
   several block counts including non-multiples of the tile, since the torch path's arithmetic is
   fixed and any difference would silently change the model's weights.
2. Correct for the real GGUF buffer, i.e. reading the block as a flat uint8 stream with the
   `d/dmin/scales/qh/qs` sub-fields at their byte offsets — not a pre-split tensor.
3. Measured inside the actual `cast_bias_weight` path at production shapes, with the same A/B/A and
   A/A band method used for the attention work, and peak-VRAM recorded: the whole reason caching was
   rejected is memory pressure, so a faster kernel must not raise the transient.
4. Rejected if it does not beat the torch path by a clear margin at *every* shape it replaces.
