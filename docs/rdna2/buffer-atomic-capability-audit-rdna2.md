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
| `supportsBufferAtomicFadd()` | [TargetFeatures.cpp:278-289](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L278) | no RDNA2 arm → returns `true` by fallthrough | **ISA limit** (no `FADD` opcode), but currently saved only by the RMW gate — see the trap below |
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

### A trap for whoever widens the gate

`supportsBufferAtomicFadd()` has no RDNA2 arm, so it falls through to `return true`
([TargetFeatures.cpp:278-289](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L278)).
Today that is harmless only because `supportsBufferAtomicRMW()` rejects the family first. If the RMW
gate is widened without giving fadd an explicit RDNA2 arm, the lowering will be asked to emit
`BUFFER_ATOMIC_FADD` — an opcode that does not exist on this part. That inconsistency is a real
defect in the gate structure, independent of whether RDNA2 atomics ever get enabled.

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
   the gate. No list change should be made before that answer exists.
2. **Return-value and cache-policy interaction.** On GFX10 the atomic's `GLC` bit selects whether
   the pre-op value is returned (ISA guide 8.1.10 "Atomics"), so the cache-policy aux path and the
   atomic path must not share a table — the RDNA2 cache helper says so explicitly, and any atomic
   work has to respect it.

## Blocked, and what unblocks it

Runtime is currently unavailable to me: an earlier `rocprofv3` session of mine is stuck holding the
admission lock (its first Triton dispatch never completed and the tool ignores `SIGTERM`), and the
standing constraints are "do not kill or signal any process" and "zero launches unless the state is
`EXCLUSIVE_COOPERATIVE_LOCK`". Preflight therefore reports `state=BUSY, cooperative lock held by
pid 260702, KFD PID 260737`. Unblocking needs that process ended by its owner — after which the
validation queue is:

* unmasked i32/i64 `and/or/xor/umin/umax` and f32 `min/max` buffer atomics on gfx1030 *and* native
  gfx1031, with an A/A control in every case;
* a deliberately masked atomic case to answer the lane-suppression question, written so a wrong
  answer shows up as a detected corruption rather than a silent OOB write;
* only then widen `supportsBufferAtomicRMW()` / `supportsBufferAtomicFMinMax()`, each in its own
  commit with its lit test.
