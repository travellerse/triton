# P3 — what the restored RDNA2 family actually buys (measured A/B)

Date 2026-10-19. RX 6700 XT, `warp_size=32`, `HSA_OVERRIDE_GFX_VERSION=10.3.0` so the emitted
target is `gfx1030`; fork Triton 3.9.0 built from `work/rdna2-gfx1031-m0`. Every run was admitted
with `--acquire` and `state=EXCLUSIVE_COOPERATIVE_LOCK` (`/tmp/rdna2-admission/audit/*-p3-*.json`).

Restoring `ISAFamily::RDNA2` switches on two codegen levers that had never been timed:

* **P4** — `isRDNA()` becomes true, so cross-lane shuffles take the DPP butterfly instead of
  `ds_bpermute` (`Utility.cpp:172`), and `ScheduleLoops` stops filtering small vectors.
* **P5** — the vec-128 LDS tile `getSharedLdStTiles()` returns `{1,2,20}` instead of the scalar
  fallback.

## Method

`rdna2_perf_ab.py`: 8 s clock ramp (the card idles at 500 MHz), one deterministic input draw,
**50 launches inside each timed bracket** so a sample is kernel time rather than launch latency,
median of 7 samples, and the baseline arm run **three times** to measure the A/A band. Arms are
separate builds: HEAD (P5 present) vs HEAD with only the `case ISAFamily::RDNA2:` line removed,
each rebuilt via ccache at `-j6`.

## Results (ms per launch, median)

| workload | P5 present | P5 absent | delta | A/A spread |
|---|---|---|---|---|
| `trans_1024x1024_f32` | 0.021 | 0.021 | 0.000 | 0.000 |
| `trans_2048x512_f32` | 0.022 | 0.022 | 0.000 | 0.000 |
| `rms_5120x1024_f16` | 0.029 | 0.029 | 0.000 | 0.000 |
| `rms_5120x4096_f16` | 0.088 | 0.090 | +0.002 | 0.002 |
| `softmax_5120x1024_f16` | 0.032 | 0.032 | 0.000 | 0.000 |
| `dot_64x64x1024_f16_1tile` | 0.049 | 0.049 | 0.000 | 0.000 |

**Every delta is at or below the arm's own A/A spread.**

## Graded conclusions

* **P5 IR change: CONFIRMED.** Cross-warp `convert_layout` with 128-bit register groups on both
  sides goes from 2393 to 2556 lowering lines for `gfx-arch=gfx1030/gfx1031`, and the output becomes
  byte-identical to `gfx1100`/`gfx1170`; the no-tile family (`gfx90a`) stays at 2394. Measured by
  temporarily removing the case and rebuilding, not inferred.
* **P5 runtime effect on these shapes: NO MEASURABLE DIFFERENCE.** These workloads are
  bandwidth-bound; the tile changes LDS address arithmetic, which they do not bottleneck on.
  P5 is therefore kept as **parity restoration** with pre-removal upstream and with RDNA3, and must
  **not** be cited as a speedup.
* **P4 was not isolated.** No workload here is shuffle-bound, so this run neither confirms nor refutes
  a DPP-shuffle gain. Its correctness side is covered separately: 17 reduce cases and 3 `trans`+LDS
  round trips pass bit-clean on real hardware (`docs/rdna2/conformance/`).
* **No spills on these kernels**: `global_scratch_size = 0` in every artifact's metadata. Note
  `k_trans` requests `shared = 65536` — the entire 64 KiB LDS — which is the constraint any
  attention-style kernel has to be designed against on this part.

## Next performance target, from measurement rather than assumption

The dot arm above is one 64×64 tile per program and is occupancy-bound, so it does not yet quantify
the FMA `dot` ceiling. The known gap is the workload's fused attention: **`n_regs=256`,
`n_spills=711`, 8192 `v_dot2c_f32_f16`, 5.12 TFLOPs vs 17.59 TFLOPs for the same shapes through
rocBLAS** (`evidence/reports/42`), and RDNA2 has no WMMA/MFMA to fall back on. Since LLVM already
selects `v_dot2c_f32_f16`, the lever is register blocking in the FMA dot path so the tile fits the
wave32 VGPR budget without spilling — the success bar being spills materially down **and** sustained
TFLOPs approaching the rocBLAS figure, not a smaller instruction count.
