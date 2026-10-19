# Differential compile sweep: RDNA2 lowering matches RDNA3/RDNA4 across an op surface, with one cost left

Status: instrument **CONFIRMED** and committed; the one open row is **quantified but unmeasured on
hardware** (device work is blocked). Date: 2026-10-19.
Tool: [conformance/rdna2_differential_compile_sweep.py](file:///home/trave/code/triton/docs/rdna2/conformance/rdna2_differential_compile_sweep.py)
— 49 programs × 3 arches (`gfx1030`, `gfx1100`, `gfx1200`), compiled with an explicit `GPUTarget`,
zero device contact.

## What it is

The on-device probe covers behaviour; the lit sweep covers the compiler's own tests. This covers the
middle: the same source compiled for RDNA2 and compared *structurally* against RDNA3 and RDNA4.
Three arches rather than two, on purpose — a two-way diff cannot tell "RDNA2 is broken" from "this
generation spells it differently", and an earlier run of this tool proved that trap live: it flagged
15 rows until the vocabulary was normalised, all of it noise.

Normalisation, each rule justified rather than tuned until the output looked clean:
`v_dual_X` → two `v_X` (RDNA3 dual-issue), `_e32/_e64/_sdwa` suffixes dropped, `dwordx4↔b128` and
friends collapsed (RDNA3 renamed the width mnemonics), scheduler-only ops (`s_clause`,
`s_delay_alu`, `s_waitcnt_depctr`) excluded, and `_b16` channel forms folded into their `b32`
equivalents — the ISA guide confirms RDNA2 has `V_LSHLREV_B16`/`V_LSHRREV_B16` but no
`V_CNDMASK_B16`/`V_AND_B16`/`V_OR_B16`/`V_MOV_B16`, so RDNA2's b32 form is the same algorithm with
a measurable op count, which the tool reports as "emulation cost" instead of a missing
instruction.

## Result

`MATCH=38  ISA-vocabulary=5  arch-conditional-by-design=5  needs-a-look=1  compile-failures=0`

The 38 include every binary/unary op (f32/f16/bf16), all seven cast pairs, the three reductions,
cumsum, reshape, and the bit-twiddling case. The 5 "arch-conditional" rows are the three `dot_*`
cases (RDNA2 has no WMMA/MFMA at all — and per the standing rule this file does not compare an FMA
dot against a WMMA dot as if it measured speed) and the bf16 max/min/where forms.

## The one open row, and what it actually says

`atomics_mixed` — one kernel with `tl.atomic_add` (f32), `tl.atomic_max` (f32) and
`tl.atomic_and` (i32 through a bitcast), on separate pointers:

| arch | emitted atomics |
| --- | --- |
| gfx1030 | `global_atomic_cmpswap` × **8**, no buffer atomics at all |
| gfx1100 | `buffer_atomic_add_f32` ×2, `buffer_atomic_and_b32` ×2, `global_atomic_cmpswap_b32` ×4 |
| gfx1200 | `buffer_atomic_add_f32` ×2, `buffer_atomic_and_b32` ×2, `global_atomic_max_i32`, `global_atomic_min_u32` |

Read against the ISA opcode list in
[buffer-atomic-capability-audit-rdna2.md](file:///home/trave/code/triton/docs/rdna2/buffer-atomic-capability-audit-rdna2.md),
this splits cleanly and the split is not the one I expected:

* **f32 add is correct behaviour.** RDNA2 has no `BUFFER_ATOMIC_FADD` opcode at all, so the CAS
  sequence is the only option; nothing to fix.
* **`and` and `max` are a real, quantified cost.** `BUFFER_ATOMIC_AND` (57), `SMIN/UMIN` (53/54)
  and `SMAX/UMAX` (55/56) all exist on RDNA2, and RDNA3/RDNA4 use the buffer forms. `supportsBufferAtomicRMW()`
  excludes the family, so RDNA2 pays a compare-and-swap loop where the hardware has a single
  instruction — here 8 atomics instead of 4.

That is the M1 gate decision, now with an instruction-level cost attached rather than a hypothesis.
The fix stays where M1 left it: the integer subset (`and/or/xor/umin/umax`, i32/i64) was built and
pinned in an isolated worktree, restricted to unmasked accesses because GFX10's behaviour for a
predicated-off buffer atomic is undocumented and the wrong answer is a silent out-of-bounds write,
and it was never merged because it needs device validation.

## Blocked, and what it unblocks

`state=BUSY`, cooperative lock held by pid 260702 with KFD pid 260737 — a stuck `rocprofv3` session
of mine that ignores SIGTERM, so `timeout` cannot reclaim it and I do not kill the user's processes.
Queued behind that: the integer-buffer-atomic A/B (correctness first, then the atomics-throughput
effect this table predicts), the pinned-vs-auto layout A/B (10–37% spill differences, tile-dependent),
the cache-policy effect-size run, and the Q6_K dequant bit-exactness gate.
