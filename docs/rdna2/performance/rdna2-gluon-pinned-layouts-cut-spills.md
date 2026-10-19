# Hand-pinned Gluon layouts spill less than Triton's auto-layout on RDNA2 (static, up to −28%)

Status: **CONFIRMED statically** at matched math and matched emitted dot counts; **NOT ESTABLISHED**
as a speed improvement — on this card spill count and speed already disagreed once before.
Date: 2026-10-19. Tool:
[performance/rdna2_gluon_fa_spill_map.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_gluon_fa_spill_map.py)
(`--threads 8,4` and `--threads 4,8`), arch `gfx1030`, zero GPU launches.

## Why this question

RDNA2 attention is register-bound ([rdna2-fast-attn-is-register-bound.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-fast-attn-is-register-bound.md)),
and the device measurement that followed it showed the relationship is not monotone: cutting spills
along the tile axis sped things up (1.495×), yet the statically *spill-free* config was 17% **slower**
than the spilling baseline ([rdna2-spill-cost-measured.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-spill-cost-measured.md)).
So a lower spill count is a *candidate selector*, never a ranking — which is exactly how it is used here.
Gluon is now known to compile for gfx1030
([gluon-can-express-rdna2-without-matrix-units.md](file:///home/trave/code/triton/docs/rdna2/gluon-can-express-rdna2-without-matrix-units.md)),
so the question is whether choosing the layout by hand changes register pressure at all.

## Method and its validity checks

Both arms are the same attention inner step (QK^T → `exp2(s − 8.0)` → PV, SK=1024). Neither does row
reductions, because `gl.max`/`gl.sum` along an axis do not build through the offline `GluonASTSource`
path on *any* AMD arch (reproduced on gfx1100 as well as gfx1030) — a frontend limitation, not an
RDNA2 one. Every row prints each arm's emitted `v_dot*` count, and in all 20 rows the counts are
identical, so the arms really did the same work. The comparison is only meaningful with that check:
the first version of this tool compared a reduction-free Gluon kernel against a Triton kernel that
*did* online softmax, and reported "Gluon better" at one tile purely because it was doing less math.

## Result: the pinned layout wins, but only the right one

`vgpr_spill_count`, gfx1030, same tile/HD/warps on each row:

| BM×BN | HD | W | auto (Triton) | pinned `threads=[8,4]` | pinned `threads=[4,8]` |
| --- | --- | --- | --- | --- | --- |
| 32×32 | 64 | 1 | 382 | 475 | **316** |
| 32×32 | 128 | 1 | 557 | 1841 | 985 |
| 64×32 | 64 | 1 | 1065 | **902** | **981** ✱ |
| 64×32 | 64 | 2 | 387 | 412 | **311** |
| 64×32 | 128 | 1 | 2957 | 3060 | **2646** |
| 64×32 | 128 | 2 | 1124 | 1566 | **1010** |
| 64×64 | 64 | 1 | 2203 | 2685 | 2212 |
| 64×64 | 64 | 2 | 918 | 1436 | **773** |
| 64×64 | 128 | 1 | 5674 | 6573 | **4193** (−26%) |
| 64×64 | 128 | 2 | 2708 | 3879 | **1951** (−28%) |

Tally: `[8,4]` → 1 win / 9 losses; `[4,8]` → **8 wins / 2 losses**, largest at the tiles that matter
for `head_dim=128` (64×64: −26% and −28%).

✱ Note `64×32/64/1`: both pinned variants beat auto there, so that row is not the reason for the
conclusion.

**A single arm nearly produced the opposite paper.** With one hand-picked layout the honest reading
was "Triton's auto-layout wins 9 of 10 — Gluon layout control is not the lever". Changing
`threads_per_warp` flips it. The finding is therefore about *layout choice*, not about Gluon per se,
and the two arms that differ only in `[8,4]` vs `[4,8]` bracket how much that choice moves register
pressure on this hardware.

## The interesting consequence for Triton itself

If a fixed `threads_per_warp=[4,8]` beats what the compiler's own layout selection picks for RDNA2 at
identical math, that is a statement about the selection heuristic, not just about Gluon authoring.
The follow-up, which does not need Gluon at all, is whether the same register pressure can be had
from the Triton frontend by changing what the pass is allowed to choose on RDNA2 — i.e. a
`TritonGPU`-side tuning item rather than a hand-authored kernel. That has not been investigated.

## Queued, and blocked on the device

A stuck `rocprofv3` session of mine still holds the card (I do not kill processes), so the runs this
result asks for are queued, not performed:

1. `64×64 / HD=128 / W=2`, pinned `[4,8]` vs auto, on device, batched launches with the A/A band —
   does −28% spills become anything, given that the spill-free config was previously *slower*?
2. A wider pinned sweep (order, `size_per_thread`, warps along the other axis) to see whether `[4,8]`
   is the best of a family or a coincidence of two probes.
3. The reduction blocker: `gl.max`/`gl.sum` failing to build through `GluonASTSource` on AMD is worth
   a minimal-repro issue regardless of RDNA2 — it currently makes full FA steps unwritable in Gluon
   through the offline path on every AMD target.
