# MXFP8 GEMM on RDNA2: the scaled-dot result tiling was a GPU hang

Status: **MEASURED IMPROVEMENT** on native gfx1031 (RX 6700 XT). Date: 2026-10-20.
Benchmark: [rdna2_mxfp8_gemm_bench.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_mxfp8_gemm_bench.py).

## What was wrong

`DotScaledOp` was converted with `GenericOpPattern`, so its result kept the default 1x1
`sizePerThread` tile, with a comment assuming a later pass would replace it. Matrix-unit targets do
overwrite the encoding, but the AMD software decomposition used by RDNA2 keeps it. On the FMA path
that tile makes each thread own 8 A rows instead of 2 at 32x32x64; the 64x64x128 K-loop carried
16,304 B of scratch per lane. At 4096³ the resulting kernel did not just run slowly — it coincided
with `gfx_0.0.0`/`gfx_0.1.0` ring timeouts and resets at 02:20 local, while the process still printed
2136 ms per launch.

`TritonScaledDotPattern` now shares `TritonDotPattern`'s result tiling (2x2 at >=4 elements per
thread, 4x4 at >=16). A 2x2-only cap was measured and rejected: it takes the loop from 32,313 to
41,811 instructions.

## Method

`e4m3 × e4m3` MXFP8 GEMM, 64x64x128 tiles, `num_stages=2`, scales `2^0`, random u8 operands.
Grid `(M/64, N/64)`; eight warmup launches, then ten timed launches, median reported. All runs went
through the GPU admission wrapper on native gfx1031 (emitted target `gfx1031`).

## Results

| shape | warps | median | effective TFLOPs | note |
| --- | --- | --- | --- | --- |
| 4096³ | 4 | **140.49 ms** | 0.98 | same config as the pre-fix arm |
| 4096³ | 8 | **101.83 ms** | **1.35** | best measured |
| 2048³ | 8 | 13.08 ms | 1.31 | |
| 1024³ | 8 | 1.85 ms | 1.16 | first run after the fix, no hang |
| 4096³ | 4 | 2136 ms (pre-fix) | 0.06 | plus gfx ring timeouts/resets |

The pre-fix arm is **15.2x slower** than the post-fix arm at the same shape and warp count, and it
destabilised the GPU; the post-fix runs produced no new ring timeouts or resets (checked in the
kernel log afterwards). Ten-launch spreads were ~1–2%, far inside the size of the effect.

## Tile sweep

With the tiling fix in place, the remaining choice is the tile itself. Nine combinations were
static-checked on gfx1031 first (all <= 5.5 KB of scratch per lane, versus 16 KB for the old 1x1
tile), then timed through the same harness at two sizes (median of 5, spread ~0.3–1%):

| config (BM×BN×BK, warps, stages) | 1024³ | 4096³ |
| --- | --- | --- |
| 64×64×128 w8 s2 | 1.873 ms (1.15) | 101.66 ms (1.35) |
| 64×64×64 w8 s2 | **1.708 ms (1.26)** | 101.26 ms (1.36) |
| 64×64×64 w8 s3 | 1.868 ms (1.15) | 106.27 ms (1.29) |
| 128×64×64 w8 s2 | 1.792 ms (1.20) | **94.99 ms (1.45)** |
| 64×128×64 w8 s2 | 2.000 ms (1.07) | 111.24 ms (1.24) |
| 128×128×64 w8 s2 | 2.601 ms (0.83) | 140.15 ms (0.98) |
| 64×64×128 w8 s3 | 2.343 ms (0.92) | 110.60 ms (1.24) |
| 64×64×256 w8 s2 | 3.219 ms (0.67) | 151.71 ms (0.91) |
| 32×64×128 w4 s2 | 2.446 ms (0.88) | 138.99 ms (0.99) |

No single tile wins both sizes: at 1024³ the 64×64×64 config is best, while at 4096³ widening M to
128 wins by 6.6% because the loaded B tile is reused across more output rows. BK=128 and deeper
pipelines both lose at the large size. The gap between the top two at 4096³ is far above the
per-config repeat spread, so 128×64×64 is a real 4096³ recommendation, not a tie.

## What this is not

* The TFLOPs are **effective** rates for dequantise-in-register FP8 → FMA with f32 operands, not
  tensor-core peak. RDNA2 has no FP8 hardware, so the software upcast is part of the work.
* MXFP8 is not part of the Qwen-Image GGUF workload; this is about the scaled-dot feature running
  instead of hanging, not about the user's step time.
* The pre-fix timing comes from a run that triggered ring resets, so treat the 15.2x as the size of
  the difference between a pathological and a healthy kernel rather than a clean benchmark ratio.
  The absolute post-fix numbers are the reliable ones.

Numerics for the same decomposition (all formats, NaN/inf, packed-nibble order, M-packed) are
measured separately by `conformance/rdna2_fp8_nan_probe.py` on the same device.
