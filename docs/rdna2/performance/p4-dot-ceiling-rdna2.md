# P4 — measured ceiling of Triton's FMA `dot` on RDNA2

Date 2026-10-19. gfx1030 emitted target under `HSA_OVERRIDE_GFX_VERSION=10.3.0`, fork Triton 3.9.0,
`warp_size=32`, admitted runs (`EXCLUSIVE_COOPERATIVE_LOCK`). fp16 in, fp32 accumulate.
Reproduce with `docs/rdna2/performance/rdna2_dot_ceiling.py`; raw table in
`docs/rdna2/performance/artifacts/dot_ceiling2.json`.

The fused-attention kernel in the real workload reaches **5.12 TFLOPs** while rocBLAS reaches
**17.59 TFLOPs** on the same dots, and RDNA2 has no WMMA/MFMA. Before touching either the kernel or
the compiler, this measures what `tl.dot` can do at all on this hardware.

## Best per shape

| shape | M×N×K | best tile (BM/BN/BK/warps) | TFLOPs | worst tile |
|---|---|---|---|---|
| `sq_2048` | 2048³ | 64/128/64/w4 | **15.19** | 64/64/32/w4 → 9.91 |
| `sq_1024` | 1024³ | 64/128/64/w4 | **12.72** | 64/64/32/w4 → 8.41 |
| `qk_like` | 1024×8192×128 | 128/128/32/w4 | **13.77** | 128/128/64/w4 → 7.65 |
| `pv_like` | 1024×128×8192 | 64/64/64/w4 | **5.96** | 128/128/32/w4 → 1.73 |

## What this settles

1. **The compiler's dot lowering is not the ceiling.** Triton reaches 15.19 TFLOPs, ≈86% of the
   17.59 TFLOPs rocBLAS achieves on this card. So rewriting the FMA dot lowering is not the first
   move; getting the kernel's tile shapes right is.
2. **The attention shapes are not equally reachable.** QK^T-like (short-K, wide-N) hits 13.77 TFLOPs,
   but PV-like (long-K, N=128) tops out at **5.96**. The workload's fused kernel at 5.12 TFLOPs is
   therefore already close to the *PV-shaped* limit, which is why spill reduction alone never moved it
   much — the contraction shape, not the register pressure, is the binding constraint.
3. **Tile choice dominates: 8.8× spread** (1.73 → 15.19) over the same op on the same hardware.
   For the PV-like shape the popular "big tile" choice BM128/BN128 is catastrophic (1.73 TFLOPs,
   3.4× worse than the best tile) — a trap any attention re-tuning must avoid.

## Correctness note, recorded so it is not misread

An earlier run reported `WRONG max_abs_err≈300` for several small-tile configs. That was **a bug in
this probe's grouped tile ordering** (`gsz` could go ≤0, then `pid % gsz` mapped tiles outside the
output, leaving unwritten rows), not an RDNA2 numerics defect. After fixing it to the canonical
grouped-order formula, all 40 configs across all four shapes are numerically correct against a fp32
`torch` reference. Recorded because the false signal was generated on RDNA2 hardware and could
otherwise be quoted as evidence of a silent-wrong.

## Consequence for the fast-attention work

Target the *shape*, not the instruction: make the PV contraction into something the hardware can do
at ≥10 TFLOPs (e.g. keep the long contraction inside the block while widening the per-program output,
or split the contraction so each program's dot is wide rather than narrow), while keeping the QK^T
side at its 13.8 TFLOP shape. The measured bar for a new kernel is aggregate > the current 116 ms
per-block call from the P1 study and per-dot TFLOPs consistent with the table above.
