# Measured: VGPR spills cost real attention time on RDNA2 — but "spill-free" is not the target

Status: **MEASURED IMPROVEMENT** on the attention inner loop, with a falsification of my own
static hypothesis. Device: native gfx1031 (RX 6700 XT) under the host's
`HSA_OVERRIDE_GFX_VERSION=10.3.0`, admitted through the cooperative lock
(`state=EXCLUSIVE_COOPERATIVE_LOCK`, tags `spillab`, `spillab3`).
Date: 2026-10-19. Harness: [rdna2_spill_ab.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_spill_ab.py).

## What was measured

An online-softmax flash-attention forward step (one program per query block, K/V streamed
in `BLOCK_N` chunks, fp16 in / fp32 accumulate, `exp2`-based softmax), `S=4096`,
`heads=16`, `head_dim ∈ {64,128}`. `head_dim=128` is the production shape: the Qwen-Image
DiT attention is `dim 4096 = 32 × 128` (see `rdna2-gemm-layout-map.md`).

Protocol: 3 warm launches, 8 s clock ramp, then 7 brackets of 50 back-to-back launches with
CUDA events, median of brackets; the baseline arm is repeated as the last arm to size the
A/A band. Correctness per arm against a per-head fp32 reference. The emitted target is
`amdgpu10.30-amd-amdhsa-unknown-gfx1030`, i.e. the measurement is labelled gfx1030 under the
host's override even though `rocm-smi` reports the card as gfx1031; both are the same
GFX10.3 ISA family, which is what the register file and the FMA dot path depend on.

**A/A band: 42.754 vs 42.808 ms for the identical baseline arm at the start and end of the
run → 0.13%.** Anything above ~0.5% in the tables below is therefore material.

## Result 1 — spills are expensive, monotonically, along the tile axis

`head_dim=128`, `num_warps=4` fixed; spill counts are `.vgpr_spill_count` from the emitted
`amdhsa` metadata of the *same kernel that was timed*:

| tile BM×BN | vgpr spills | median ms | GFLOP/s | ratio vs baseline | max rel err |
| --- | --- | --- | --- | --- | --- |
| 64×64 (baseline) | 947 | 42.754 | 3.22 T | 1.000 | 7.41e-4 |
| 64×64, warps=8 | 356 | 34.970 | 3.93 T | **1.224** | 7.41e-4 |
| 32×32 | 222 | 28.888 | 4.76 T | **1.482** | 7.41e-4 |
| 64×64, `head_dim=64` | 395 | 17.071 | 4.03 T | (2.508, different shape) | 4.64e-4 |
| 32×32, `head_dim=64` | 9 | 16.143 | 4.26 T | (2.652, different shape) | 4.64e-4 |

Work is constant within the `head_dim=128` group (4·H·S²·D = 137.44 GFLOP), so ms ratios are
direct throughput ratios. The best measured point, 4.76 TFLOPs, is **80% of the 5.96 TFLOPs
PV-shaped dot ceiling** measured in `p4-dot-ceiling-rdna2.md` and 31% of the 15.19 TFLOPs
square-GEMM ceiling. The baseline (3.22) sits at 54% of that same ceiling.

The identical `rel_err` in every arm (7.41e-4 at D=128, 4.64e-4 at D=64) is the evidence that
all arms compute the same function — this is a codegen/occupancy effect, not a different
math. Peak VRAM was flat at 136 MiB across arms, so the win is not bought with memory.

**This resolves the earlier FA sweep's negative result.** That sweep (`p1` era,
`docs/rdna2/performance/`) used configs `128,64,4; 128,128,4; 64,128,4; 128,64,8; 256,64,8;
64,64,4` — every one of them is in the heavy-spill regime (compare the 947/395 rows above),
so it never contained a low-spill arm and could not have found this. My recorded
"NO MATERIAL IMPROVEMENT" for FA config choice was an honest measurement of a
hypothesis that had not actually been tested.

## Result 2 — the spill-free config lost to the low-spill one (my static guess was wrong)

The static table said the only clean island at `head_dim=64` was `32×32` with `num_warps=8`
(`vgpr_spill_count = 0`, `next_free_vgpr = 190`). Measured:

| config, `head_dim=64` | spills | median ms |
| --- | --- | --- |
| 32×32, warps=4 | 9 | **16.143** |
| 32×32, warps=8 | 0 | 29.140 (1.81× **slower**) |

Zero spills is not the objective. The tile and the traffic are identical in those two rows —
only the warp count differs — so the cost is how the work is spread: 8 warps leaves ~4 fp16
elements per lane per operand, which shrinks per-lane LDS vectors and per-thread ILP while
doubling the number of wavefronts that must share the same LDS and barriers. I have **not**
established which of those mechanisms dominates; what is established is the measurement:
removing the last 9 spills costs 1.81×.

The usable rule is therefore **minimize spills subject to keeping enough per-lane work**:
among the arms measured, `32×32 × num_warps=4` at `head_dim=128` (222 spills) is the
practical optimum, not the spill-free `num_warps=8` point, and not the big tiles.

## Result 3 — it also explains the tile-spread observation in `p4-dot-ceiling-rdna2.md`

That report recorded an 8.8× spread over the *same* dot on this card (1.73 → 15.19 TFLOPs)
and specifically that "for the PV-like shape the popular big tile choice BM128/BN128 is
catastrophic", without a mechanism. The mechanism is register pressure: in this very kernel
`64×64` at `head_dim=128` needs 947 spilled VGPRs, `128×64` at the same shape needs 2254,
and the single-dot sweep reaches 1870 for `128×128 × D=128 × w4`
(`artifacts/dot_spill_map_2026-10-19.txt`), always with `next_free_vgpr` pinned at the 256
per-thread ceiling. Big tiles on RDNA2 are not merely "less efficient" — they move register
traffic to VRAM inside the inner loop.

## Result 4 — reproduced on a second admitted run (`spillab3`)

Same kernel, same shapes, six arms with the baseline repeated last, `head_dim=128`
(137.44 GFLOP constant). A/A band this time **0.22%** (42.662 vs 42.754 ms), and the
baseline median itself matches run 1's within 0.2% (42.754 vs 42.662).

| arm | tile, warps | vgpr spills | median ms | TFLOPs | ratio vs baseline |
| --- | --- | --- | --- | --- | --- |
| 0 | 64×64, w4 | 947 | 42.662 | 3.22 | 1.000 |
| 1 | 32×64, w4 | 373 | 35.687 | 3.85 | **1.195** |
| 2 | 32×32, w4 | 222 | 28.534 | **4.82** | **1.495** |
| 3 | 64×64, w8 | 356 | 34.846 | 3.94 | 1.224 |
| 4 | 32×32, w8 | **0** (next_free_vgpr 254) | 51.005 | 2.69 | **0.836 — regression** |
| 5 | 64×64, w4 (repeat) | 947 | 42.754 | 3.22 | 0.998 |

So the two axes behave differently and both effects are real and reproducible:

* fewer spills at fixed warps → faster, monotone (947 → 373 → 222 gives 1.000 → 1.195 → 1.495),
* more warps at a *large* tile → faster (arm 3, ×1.224),
* more warps at an already-*small* tile → 17% **slower** (arm 4).

The best point found so far, 4.82 TFLOPs, is **81% of the measured 5.96 TFLOPs PV-shaped
ceiling** for this card.

## Still open

* Reproducibility is established (two admitted runs, `spillab` and `spillab3`, baselines
  42.754 / 42.662 ms). What has **not** been swept is the region between `32×32` and `32×64`
  at `warps=2`, and `48`-class tiles are unavailable to Triton's power-of-two blocks.
* The production call is *not* this kernel: Qwen-Image-2.1 attention runs `baddbmm`-based
  SDPA-style paths and Triton only where ComfyUI dispatches to it, so this establishes the
  RDNA2 attention *design envelope*, not a whole-workload speedup. Wiring a Triton FA into
  ComfyUI and measuring the step is the next step, and it needs the geometry gate
  (task: image gate + VRAM accounting).
* Whether a formulation with fewer live values (e.g. keeping the scores tile in fp16, or
  splitting `head_dim` across the K loop) can push past 4.82 TFLOPs is untested; `num_stages`
  was already ruled out statically.
* How much of the pressure is the softmax bookkeeping rather than the two dots: the sweep in
  `rdna2_fa_spill_map.py` shows the same tiles needing far more registers with the online
  softmax than a bare dot does (`128×64, D=64, w4`: 1018 spills here vs 300 for the plain dot),
  so the running `m`/`l` and the extra `p` tile are a real part of the cost — but that sweep
  was stopped partway (partial output kept as
  `artifacts/fa_form_spill_map_gfx1030_partial.txt`) once the timed kernel's own table became
  the number that mattered.
