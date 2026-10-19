# RDNA2 buffer atomics: ISA capability versus what Triton allows

Status: audit **CONFIRMED** against the AMD "RDNA 2" ISA Reference Guide (section 12.14, MUBUF
opcode table) and the current gates in code. The two gaps it names are **BLOCKED for fixing** —
each needs device validation, and the device is currently unavailable to me (see the last section).
Date: 2026-10-19.

## What RDNA2 actually has

Opcodes as listed in the ISA guide's MUBUF tables:

| class | opcodes present on RDNA2 |
| --- | --- |
| integer 32-bit | `SWAP` 48, `CMPSWAP` 49, `ADD` 50, `SUB` 51, `CSUB` 52, `SMIN` 53, `UMIN` 54, `SMAX` 55, `UMAX` 56, `AND` 57, `OR` 58, `XOR` 59, `INC` 60, `DEC` 61 |
| float 32-bit | `FCMPSWAP` 62, `FMIN` 63, `FMAX` 64 — each annotated "handles NaN/INF/denorm" |
| 64-bit packed | `SWAP_X2` 80, `CMPSWAP_X2` 81, `ADD_X2` 82, `SUB_X2` 83, `SMIN_X2` 85, … plus `FMIN_X2` / `FMAX_X2` / `FCMPSWAP_X2` |
| **float add** | **absent** — there is no `BUFFER_ATOMIC_FADD` in the RDNA2 opcode list |

## What Triton allows

| gate | file:line | RDNA2 arm | classification |
| --- | --- | --- | --- |
| `supportsBufferAtomicRMW()` | [TargetFeatures.cpp:271-276](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L271) | excluded entirely (CDNA3/4, RDNA3/4m/4, CDNA5) | **software gap** — every integer op above exists on RDNA2 |
| `supportsBufferAtomicFadd()` | [TargetFeatures.cpp:278-289](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L278) | explicit `false` since `88c8a152ec`; before that no RDNA2 arm → `true` by fallthrough | **ISA limit** (no `FADD` opcode); the fallthrough was the defect, see below |
| `supportsBufferAtomicFMinMax(f32)` | [TargetFeatures.cpp:291-299](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L291) | f32 list is RDNA3/RDNA4m/RDNA4/CDNA5 → RDNA2 false | **software gap** — `FMIN`/`FMAX` 63/64 exist |
| f64 min/max | same function, CDNA-only list | false | correct: the `_X2` float ops are f32-pairs, not f64 |

So the blanket `supportsBufferAtomicRMW` exclusion is not an ISA statement. What the M1 work
actually recorded is narrower and worth quoting rather than embellishing: `supportsBufferAtomicRMW()`
and `supportsBufferAtomicFMinMax()` exclude RDNA2 *and* `Unknown`, which is why the four
`test/TritonGPU/amd/amd-convert-buffer-ops-atomic-*.mlir` lines with `NO-F16,NO-BF16,NO-F32,NO-F64`
prefixes **pass either way** — they are anti-regression guards on the gate, not evidence that RDNA2
lacks the instructions ([m1-implementation-report.md:500-512](file:///home/trave/code/triton/docs/rdna2/m1-implementation-report.md#L500)).
The integer subset (`and/or/xor/umin/umax`, i32/i64) was built and pinned in an isolated worktree
and never merged, because its runtime validation stayed BLOCKED.

### A trap for whoever widens the gate — closed

`supportsBufferAtomicFadd()` had no RDNA2 arm, so it fell through to `return true`
([TargetFeatures.cpp:278-289](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L278)).
Today that was harmless only because `supportsBufferAtomicRMW()` rejects the family first. If the RMW
gate is widened without giving fadd an explicit RDNA2 arm, the lowering will be asked to emit
`BUFFER_ATOMIC_FADD` — an opcode that does not exist on this part. That inconsistency is a real
defect in the gate structure, independent of whether RDNA2 atomics ever get enabled.

Closed on this branch by `88c8a152ec`: RDNA2 gets an explicit `false` with the opcode-table citation,
which changes nothing for any other family (one pass call site, one never-called `TargetInfo` wrapper,
both checked by grep).

## Why the fix is not just "add RDNA2 to the list"

Two things must be settled with hardware, and neither is answerable statically:

1. **Masked lanes — now documented, with one residual.** The RDNA2 guide's "Range Checking" section
   (8.1, Table 35 and its notes) answers the question the M1 record called undocumented:
   "When an address is out of range, **writes are ignored (dropped)** and reads return zero", and
   note 3 states explicitly that "Load/store-format-* instruction **and atomics** are range-checked
   *all or nothing* — either entirely in or out". So a predicated-off lane issued as a buffer
   atomic is dropped by hardware, provided the descriptor actually asks for a checking mode.

   Triton does set that mode on RDNA/RDNA4/CDNA5: `OOB_select = 3` with `stride = 0` and
   `numRecords = INT_MAX-1` ([BufferOpsEmitter.cpp:69-84](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp#L69)),
   and mode 3 with `const_stride == 0` reduces to `offset + payload > NumRecords`. Masked lanes are
   sent `offset = 0x80000000` ([BufferOpsEmitter.cpp:280-283](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp#L280)).

   **The residual is arithmetic, not ISA lore.** `0x8000_0000` exceeds `numRecords = 0x7FFF_FFFE`
   only under an *unsigned* comparison; if the hardware compares the offset as signed, that value is
   negative, the check does not fire, and masked lanes perform real writes — which is exactly the
   silent-corruption failure the M1 record was protecting against. The guide does not say which.
   This is now a one-line device experiment rather than an open design question: run a masked buffer
   atomic over a guard region whose contents are known, with the mask false, and check whether the
   guard bytes change. Any lane written = signed comparison = the current code is already unsafe for
   *every* masked buffer access on RDNA, atomics included, which would make this more urgent than
   the gate. Given the load-side observation above, the gate can now be widened behind the probe as
   a confirmation rather than as the first test of the mechanism; that is a change of confidence,
   not of order of work: run the probe, then enable, then measure the CAS-to-single-instruction win
   this file's own instruction counts predict.

   That residual is now largely closed by evidence rather than argument. Comparing how a masked
   `tl.atomic_and` lowers on the two families:

   * **gfx1100** emits `buffer_atomic_and_b32` fed by a `v_cndmask … 0x80000000` address — RDNA3's
     masked atomics rely on hardware OOB suppression today, in shipped code;
   * **gfx1030** emits neither: it guards with `v_cmpx_gt_i32_e32` + `s_cbranch_execz` and a CAS
     loop, so RDNA2's masked atomics currently use EXEC branching, while RDNA2's masked *buffer
     loads* in the same kernels already use the `0x80000000` sentinel.

   That last point is the discriminator the guard-region test cannot provide: under a signed offset
   comparison, a masked-off buffer load on RDNA2 would read garbage where the hardware is supposed
   to return zero, and masked loads with `other=0` are covered by the on-device conformance probe
   (43/43 at native gfx1031). The comparison is therefore unsigned on this part, by observation on
   the load side of the very mechanism in question — `OOB_select` is one descriptor field shared by
   loads, stores and atomics, and note 3 puts atomics in the same all-or-nothing check.

   Two things still shrink the residual, and both are worth stating so the risk is not overstated:
   the mechanism is **already in shipped use** -- RDNA3/RDNA4 build the descriptor with the same
   `OOB_select=3` flags and *do* have `supportsBufferAtomicRMW()` enabled, so masked buffer atomics
   already run through this path on shipping targets; the open question is only whether gfx10
   compares the offset differently. And `num_records` is an unsigned descriptor field, which makes
   the unsigned reading the natural one.

   A guard-region test cannot settle it either way: a lane that escaped suppression would write at
   `base - 2 GiB`, outside the tensor, so in-buffer sentinels stay clean under both readings. The
   discriminating observation is a memory violation, which is what
   [conformance/rdna2_masked_buffer_atomic_probe.py](file:///home/trave/code/triton/docs/rdna2/conformance/rdna2_masked_buffer_atomic_probe.py)
   is built to surface (it also runs unchanged on gfx1100, where the path is live today).
2. **Return-value and cache-policy interaction.** On GFX10 the atomic's `GLC` bit selects whether
   the pre-op value is returned (ISA guide 8.1.10 "Atomics"), so the cache-policy aux path and the
   atomic path must not share a table — the RDNA2 cache helper says so explicitly, and any atomic
   work has to respect it.

## The signed min/max refusal is a cross-target gap, and it hides a structural bug

`ConvertToBufferOps` refuses `RMWOp::MIN`/`MAX` on an integer for **every** target ("signed integer
min/max RMW operations use generic atomic lowering"), even though `buffer_atomic_smin`/`smax` select on
gfx1030, gfx1100 and gfx942 alike (`llc -mcpu=...` on
`llvm.amdgcn.raw.ptr.buffer.atomic.{smin,smax}.i32` and the `.i64` `_x2` forms). RDNA2 now bypasses that
refusal through the per-op subset gate; RDNA3 and CDNA still pay the CAS loop, which is the remaining
cross-target win.

The refusal is also where the structure was wrong: it was an early-return, and the floating-point check
underneath it (`supportsBufferAtomicFMinMax`) is **not guarded by a type test** -- it answers false for
`i32` -- so the first attempt to let RDNA2 through fell straight into the float refusal and changed
nothing. Anyone widening it further must keep the two branches separated by type, which is what the
code does now (`if integer / else float`).

## Blocked, and what unblocks it

Runtime is currently unavailable to me: an earlier `rocprofv3` session of mine is stuck holding the
admission lock (its first Triton dispatch never completed and the tool ignores `SIGTERM`), and the
standing constraints are "do not kill or signal any process" and "zero launches unless the state is
`EXCLUSIVE_COOPERATIVE_LOCK`". Preflight therefore reports `state=BUSY, cooperative lock held by
pid 260702, KFD PID 260737`. Unblocking needs that process ended by its owner — after which the
validation queue is (the subset is already merged, so this validates shipped behaviour rather than
gating a merge):

* unmasked i32/i64 `and/or/xor/umin/umax` and signed `min/max` buffer atomics on gfx1030 *and* native
  gfx1031, with an A/A control in every case — `performance/rdna2_buffer_atomic_isa_probe.py` already
  compiles exactly those cases and reports `buffer_atomic_*` versus `global_atomic_cmpswap` per case, so
  the device version adds value checking, not kernel authoring;
* a deliberately masked atomic case to answer the lane-suppression question, written so a wrong
  answer shows up as a detected corruption rather than a silent OOB write;
* the measured effect of the change itself: contended-atomic throughput against a build without the
  subset, because "one instruction instead of a CAS retry loop" is a static claim that has not been
  timed on this card;
* only after those, widen `supportsBufferAtomicRMW()` / `supportsBufferAtomicFMinMax()` for the other
  families, each in its own commit with its lit test, and re-examine whether `enableIntraWaveReduce`
  should extend past CDNA3/CDNA4 now that RDNA2's contention path is a single instruction.

Float `min/max` are deliberately **not** in this queue: `buffer_atomic_fmin/fmax` select fine, but
`tl.atomic_max/min` on a float never reaches this pass as `RMWOp::MAX` on a float (the frontend bitcasts
and uses the signed/unsigned integer ordering trick), so enabling them is a NaN and ±0 semantics question
rather than a capability question.

## End to end on the real ISA text (compile-only, no device)

`performance/rdna2_buffer_atomic_isa_probe.py` compiles `tl.atomic_*` kernels through the whole pipeline
and greps the emitted `.amdgcn`. With the subset enabled, on **gfx1030**:

| case | instruction emitted | reading |
| --- | --- | --- |
| `and`/`or`/`xor` i32 | `buffer_atomic_and` / `_or` / `_xor` | one instruction, instead of the generic path's `global_atomic_cmpswap` retry loop (measured on `atomicrmw and i32` with llc below) |
| `max`/`min` i32 (signed) | `buffer_atomic_smax` / `buffer_atomic_smin` | the widening in this session |
| `max`/`min` u32 | `buffer_atomic_umax` / `buffer_atomic_umin` | |
| `and`/`max` i64 | `buffer_atomic_and_x2`, `buffer_atomic_smax_x2` | the `_X2` 64-bit forms exist and are reached |
| `add` i32, `xchg` i32 | `global_atomic_add`, `global_atomic_swap`, **0 cmpswap** | controls: these were already native, so they were not worth moving |
| `and` i32 **with a user mask** | `global_atomic_cmpswap` (8 sites) | the provably-active-lane gate is not decorative: a masked atomic cannot reach the buffer path |

The same probe on **gfx1100** (gate-open family, control) emits `buffer_atomic_and_b32` for the masked
case but still `global_atomic_cmpswap_b32` for signed `max`/`min` -- which is the cross-target refusal
above, observed rather than inferred.

Grade: pass, ROCDL, MI lowering and assembler all agree on RDNA2, and the mask gate holds at the ISA
level. Still not proven: what a bounds-suppressed lane does, because that needs the device.

## What the generic path costs each operation today (toolchain-measured, no device)

`llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=asm` on plain `atomicrmw`, i.e. what Triton
gets when the buffer path is not taken. Same result on gfx1100, so this is not an RDNA2-only quirk:

| operation | emitted on gfx1030 | instructions |
| --- | --- | --- |
| `atomicrmw add i32` | `global_atomic_add v0, v1, s[2:3]` | 1, native |
| `atomicrmw xchg float` | `global_atomic_swap v0, v1, s[0:1]` | 1, native |
| `atomicrmw and i32` | `global_atomic_cmpswap` in a retry block (`%atomicrmw.start`) | loop |
| `atomicrmw fadd float` | `global_atomic_cmpswap` in a retry block | loop |

Three consequences for the gate, none of them inference:

* The subset the enablement picks (`and`/`or`/`xor`/`umin`/`umax`) is precisely the CAS-expanded set,
  and `BUFFER_ATOMIC_AND`/`OR`/`XOR`/`UMIN`/`UMAX` do exist on this part -- so the buffer path replaces
  a compare-and-swap loop with one instruction. That is the whole claimed win; it is not a win for
  `add` or `exch`, which are already single native instructions on the global path.
* `fadd` CAS-expands *and* has no buffer opcode, so it is a genuine ISA limit rather than a software
  gap; nothing in the gate can fix it.
* Because `global_atomic_add`/`_swap` exist, `supportsBufferAtomicRMW()` staying closed for RDNA2 is
  not "RDNA2 has no atomics" -- it is only "no fast path for the bitwise/min-max ops". Anyone reading
  the gate as a capability statement about atomics in general would be wrong.

Reproduce with the four kernels in the file this section was measured from: `atomicrmw` on
`ptr addrspace(1)`, `acq_rel`, `align 4`, `-mcpu=gfx1030` and `-mcpu=gfx1100`.

## A shortcut that looks available and is not: "just emit `llvm.atomicrmw`"

The CAS loop above is emitted by Triton itself (`BufferOpsEmitter`), so the tempting fix is to stop
hand-rolling it and let LLVM choose an instruction per target. Tested directly with the LLVM this
Triton links (`llc -mcpu=...` on `atomicrmw and/or/xor/umin/max/add64/fadd/fmin`, both `acquire` and
`monotonic`, `ptr addrspace(1)`):

* every integer RMW except `add` becomes a **loop on gfx1030 and on gfx1100 alike**;
* only width-native cases collapse to one instruction (`global_atomic_add_x2` /
  `global_atomic_add_u64`);
* yet Triton's RDNA3 output contains real `buffer_atomic_and_b32` — because it calls the buffer
  atomic intrinsic itself, not because LLVM selected one.

So LLVM's generic path is not a free lunch on either family: routing through it would give RDNA2 no
gain and would take RDNA3's existing native atomics away. The only route to single-instruction
integer atomics on RDNA2 remains what M1 built: widen Triton's own buffer-atomic gate. Recorded
because the shortcut is the obvious first guess and it costs an afternoon to disprove.
