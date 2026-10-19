# RDNA2 capability-gate map: which exclusions are hardware and which are gaps

Status: **CONFIRMED** for every row marked ISA-verified; two rows are **unverified** and named as
such. No code changed — this is the compatibility half of the program: a single place that says, per
capability gate, whether RDNA2 is excluded because GFX10 lacks the instruction or because Triton
just never grew an arm for the family.
Date: 2026-10-19. Evidence: AMD "RDNA 2" ISA Reference Guide (the copy in `~/下载`), section 12.14
(MUBUF opcodes), 12.19/13.3 (DPP, DPP16/DPP8), 6.9 (DPP array), plus the gates in
[TargetFeatures.cpp](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp).

Only one predicate in the whole file names `ISAFamily::RDNA2` (the `isRDNA` helper at
[TargetFeatures.cpp:398](file:///home/trave/code/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp#L398)).
Everything else is a positive list of other families, so RDNA2 lands on `default:` and each default
has to be judged one at a time. That is what this table does.

## Excluded because GFX10 has no such instruction — correct

| gate | what RDNA2 lacks | ISA evidence |
| --- | --- | --- |
| `supportsDirectToLdsLoadBitWidth`, `supportsDirectToLdsScatter`, `supportsDirectFromLdsStoreBitWidth` | any global→LDS direct path and bulk ops | `grep -i "TO_LDS"` over the guide: no instruction matches; "bulk" occurs once, in prose. These are CDNA `global_load_to_lds` / TDM facilities |
| `supportsBufferLoadToLocal`, `requiresAliasInfoForAsyncOps`, `useAsyncMarks` | the async-copy machinery those predicates drive | same absence; the CDNA-only arms are lists of CDNA3/4/5 |
| `supportsTDM`, `supportsMultiCTALaunch`, `supportsMulticast`, `supportsClusterLoadBitWidth`, `getMaxMulticastMaskPopcount`, `supportsCvtPkScalePk8*`, `supportsHwScaled{Up,Down}cast` | tensor data motion, multicast, packed scaled converts | CDNA5-only lists; no corresponding RDNA2 opcodes |
| `supportsPermlaneSwap` | `*_SWAP` cross-lane variants | the guide lists `V_PERMLANE16_B32` and `V_PERMLANEX16_B32` and no permlane swap form |
| `supportMaximumMinimum` | — | CDNA4/RDNA4m/CDNA5 only. RDNA2 therefore takes the software NaN-handling path, and that is now *asserted* by RDNA2 RUN lines in `test/Conversion/amd/minmax.mlir`, not just implied |
| `getSharedMemorySize` → 64 KiB | — | the `default:` arm gives RDNA2 64 KiB, which is the RDNA2 per-CU LDS capacity — i.e. RDNA2 benefits from the default here rather than being penalised by it |

## Excluded, and the exclusion is right for a reason worth writing down

`supportDppBroadcast()` is false for RDNA2 (and for RDNA3/RDNA4 — it is a CDNA2-4 arm). RDNA2 *does*
have DPP, so the name alone looks like a gap. It is not: the predicate guards the last two ops of a
32-lane reduce, `row_bcast:15 row_mask:0xa` and `row_bcast:31`
([TargetInfo.cpp:563-570](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp#L563)), whose
control values are CDNA's DPP encodings. RDNA2's documented cross-row broadcast forms are
`BCASTX16`/`BCASTX32` (guide's DPP control table: `xor_mask/or_mask/and_mask` triples), and the
`else` branch already implements exactly that step with `permlanex16` + `readlane`. The RDNA2 arm of
`test/Conversion/amd/tritongpu_to_llvm_rdna.mlir` pins that fallback ladder (`rocdl.update.dpp` with
280/276/274/273, then `rocdl.permlanex16`, then `rocdl.readlane`), so this is a *chosen* lowering,
not a missing feature.

`getSharedMemoryBanks()` (`TargetInfo.cpp:132`) returns 64 only for CDNA4/CDNA5 and falls through to 32
for RDNA2, which looks wrong against the guide: it says the LDS "is configured with 64 banks" and
"128kB ... split up into 64 banks of dword-wide RAMs". The same sentence resolves it -- "These 64 banks
are further sub-divided into two sets of 32-banks each where 32 of the banks are affiliated with a pair
of SIMD32's, and the other 32 banks are affiliated with the other pair". The consumers
(`ReduceOpToLLVM.cpp:504`, `ConvertLayoutOpToLLVM.cpp:137`) use the count for bank-conflict estimation of
one wave's accesses, and a wave32 on RDNA2 is served from one 32-bank group, so 32 is the number the
analysis wants. The default also matches what was measured on this card (wave32, 64 KiB LDS; see
`initial-audit.md` §5.4) -- documented here because "default" and "correct" are easy to confuse.

## Real gaps (RDNA2 has the instruction, Triton refuses to use it)

| gate | ISA fact | consequence today |
| --- | --- | --- |
| `supportsBufferAtomicRMW()` — CDNA3/4, RDNA3/4m/4, CDNA5 | full integer set exists: `SWAP/CMPSWAP/ADD/SUB/CSUB/SMIN/UMIN/SMAX/UMAX/AND/OR/XOR/INC/DEC` (opcodes 48-61) plus 64-bit `_X2` forms | the family predicate is still closed, but the integer subset that actually pays (`and`/`or`/`xor`/`umin`/`umax` **and signed `min`/`max`**, unmasked, provably-active lanes) now takes the buffer path through a per-op gate in `ConvertToBufferOps`; the rest of the set stays generic |
| `supportsBufferAtomicFMinMax(f32)` — RDNA3/4m/4/CDNA5 | `BUFFER_ATOMIC_FMIN` 63, `FMAX` 64, `FCMPSWAP` 62 exist, and `_X2` variants too | **not enabled, on purpose**: `llc -mcpu=gfx1030` selects `buffer_atomic_fmin/fmax` fine, but `tl.atomic_max/min` on a float never arrives as `RMWOp::MAX` on a float (the frontend bitcasts and uses the signed/unsigned integer ordering trick), so switching it would be an unmeasured NaN and ±0 semantics change |
| `supportBitwidth32Elementwise()` — CDNA only | RDNA2's complete packed-op list is 20 `V_PK*` mnemonics and every one has a 16-bit element type (`V_PK_ADD_F16`, `V_PK_{MIN,MAX,FMA,FMAC,MUL}_F16`, `V_PK_{ADD,SUB,MAD,MIN,MAX}_{I16,U16}`, `V_PK_{LSHL,LSHR,ASHR}REV_B16`); there is no packed-f32 op at all | not a gap: `false` for RDNA2 is the ISA-correct answer, and `supportBitwidth16Elementwise()` returning `true` unconditionally for AMD is likewise correct |

Counterpart: `supportsBufferAtomicFadd()` now excludes RDNA2 explicitly — there is no
`BUFFER_ATOMIC_FADD` opcode at all — instead of inheriting `true` from the permissive fallthrough that
used to answer for any unnamed family. The fallthrough was harmless only because the RMW gate rejects
the family first, so this is a no-op in behaviour; it removes the inconsistency rather than a live bug.
See
[buffer-atomic-capability-audit-rdna2.md](file:///home/trave/code/triton/docs/rdna2/buffer-atomic-capability-audit-rdna2.md)
for that trap, the anti-regression status of the `atomic-*` lit lines, and the validation queue
(masked-lane suppression, and GLC meaning "return pre-op value" for atomics rather than a cache
hint, so the atomic and cache-policy tables must stay separate).

## Not verified

* `supportsWaveId()` — RDNA4/CDNA5 only. I first wrote that the guide exposes `SH_ID`/`SE_ID` fields
  and checked it: those tokens appear nowhere in the document, so it gives me no basis either way.
  The question is about cost and about what a wave id means under WGP mode, not about whether some
  register exists, and it stays unclassified.

## RDNA2's dot ALU is already being used — int8 dot is not a gap

The guide's "Feature Changes in RDNA2 Devices" section lists the packed dot instructions RDNA2 added
for inference: `V_DOT2_F32_F16`/`V_DOT2C_F32_F16`, `V_DOT2_I32_I16`/`V_DOT2_U32_U16`,
`V_DOT4_I32_I8`/`V_DOT4C_I32_I8`, `V_DOT4_U32_U8`, `V_DOT8_I32_I4`/`V_DOT8_U32_U4`. Whether Triton
uses them is answerable from emitted instructions, so the probe
(`docs/rdna2/performance/rdna2_int_dot_probe.py`) compiles a 16x16x16 dot for one warp against an
explicit `GPUTarget` and histograms the dot-family mnemonics in `.amdgcn`. No device, no submission.

| arch | i8 -> i32 | f16 -> f32 |
| --- | --- | --- |
| gfx1030 / gfx1031 | `v_dot4c_i32_i8` x32 | `v_dot2c_f32_f16` x64 |
| gfx942 | `v_dot4c_i32_i8_e32` x16 | `v_mfma_f32_16x16x16_f16` x1 |
| gfx1100 | `v_wmma_i32_16x16x16_iu8` x1 (+2 `v_permlanex16_b32`) | `v_wmma_f32_16x16x16_f16` x1 (+4 permlane) |
| gfx1200 | `v_wmma_i32_16x16x16_iu8` x1 (+1 permlane) | `v_wmma_f32_16x16x16_f16` x1 (+2 permlane) |

Reading, with the arithmetic stated so it can be checked: 16x16x16 = 4096 MACs over 32 lanes = 128
MACs per lane, so an ideal packed-int8 lowering issues 128/4 = 32 `dot4` and an ideal f16 lowering
128/2 = 64 `dot2`. **RDNA2 hits both numbers exactly.** The "ideal" count is only meaningful for this
per-lane FMA-style shape, though — CDNA and RDNA3+ hand the whole tile to one MFMA/WMMA instruction, so
their issued counts are not comparable, which is why the table shows instructions rather than a ratio.

So there is no int8/int16/f16 dot gap to close on RDNA2: the `AccelerateAMDMatmul` FMA fallback already
reaches the ISA's best MACs-per-instruction, and the measured 15.19 TFLOPs ceiling in
`performance/p4-dot-ceiling-rdna2.md` is a limit of tiles and memory movement, not of the instruction
choice. The remaining opening on this list is hardware Triton cannot express at all: `V_DOT8_I32_I4`
and `V_DOT8_U32_U4` (eight 4-bit products per instruction, directly relevant to the GGUF I4 paths) have
no `tl.dot` operand type that reaches them, even though LLVM exposes the intrinsics
(`llvm.amdgcn.sdot8`/`udot8`/`sudot4`, packed elements passed as `i32` plus an accumulator and a clamp
bit). That is a frontend-and-lowering feature, not a gate to widen, and it is the one place where an
int4 kernel could beat the dequant-then-dot4 path rather than just match it.

## `tl.dot_scaled` aborted the process on RDNA2 — closed by decomposition

`tl.dot_scaled` (MX scaled dot) was the one feature where RDNA2 fell off the pattern list entirely:
`AccelerateAMDMatmul` registered `ScaledBlockedTo*` only for CDNA4/5 and the software decomposition
only for CDNA2/3/RDNA3+, so on gfx103x no pattern matched, the op survived into
`ConvertTritonGPUToLLVM`, and
[DotOpToLLVM.cpp](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM.cpp)
hit `report_fatal_error("Unsupported DotScaleOp found when converting TritonGPU to LLVM")`. That is an
uncatchable abort, not a diagnosable error, and it made every scaled-dot kernel unusable on this
family. It was a missing arm, not a hardware limit: RDNA2 has no MFMA/WMMA, but it does not need one
for this feature, because the decomposition the other families already run in software is the same
`amdg.scaled_upcast_fp4` / `tt.fp_to_fp` plus plain `tt.dot` shape that the FMA lowering already
handles.

The RDNA2 arm now registers exactly `DecomposeAMDScaledBlocked` at the same benefit as CDNA2/3 and
RDNA3. Two pieces of evidence:

* `test/TritonGPU/amd/accelerate-amd-matmul-rdna-no-matrix.mlir` asserts both halves on gfx1030 — the
  op is gone *and* the upcast-plus-dot shape replaced it — with gfx1010's Unknown-family arm as the
  negative control and the existing RDNA3/NO-MATRIX arms covering the other families.
* The compile-only probe
  [rdna2_dot_scaled_compile_probe.py](file:///home/trave/code/triton/docs/rdna2/conformance/rdna2_dot_scaled_compile_probe.py)
  walks the whole frontend-to-LLVM pipeline against an explicit `GPUTarget` (no device, no launch).
  Every format the frontend accepts (`e2m1`, `e4m3`, `e5m2`, `bf16`, `fp16`, plus mixed
  `e4m3×e2m1` and `fp16×bf16`) compiles on gfx1030 and native gfx1031 at the probe's 32x32x64 /
  4-warp shape. The controls show zero scratch; every gfx103x MX/bf16 case spills a little even at
  this tile (about 1.1 KB of scratch per lane and ~275 folded spill stores), and plain fp16 is the
  exception (88 B, 21 stores):

| arch | mxfp8 | mxfp4 | mixed | dot path in `.amdgcn` |
| --- | --- | --- | --- | --- |
| gfx1030 / gfx1031 | 3328 instr | 2710 | 3242 | FMA (dot-family histogram empty) |
| gfx1100 | 2109 | 1404 | 1783 | `v_wmma_f32_16x16x16_bf16` x4 |
| gfx1200 | 1227 | 835 | 1060 | `v_wmma_f32_16x16x16_bf16` x4 |
| gfx942 | 1155 | 924 | 1060 | `v_mfma_f32_32x32x8_bf16` x8 |

The other formats tell the same story: e5m2, bf16 and the fp16×bf16 mix compile to the FMA dot on
gfx103x (1970–2702 instructions), while plain fp16 is the one case that reaches RDNA2's packed dot —
256 `v_dot2c_f32_f16` calls in 907 instructions. Nothing in the format set falls off the family's
decomposition, and the batched form is covered too: a 2x32x32x64 `dot_scaled` compiles on gfx1030
(7,507 instructions, 3,400 B scratch) and gfx1100 (3,925, zero scratch), so the decomposed path is
not 2D-only.

The e4m3 rows carry a NaN guard that the software path used to lack. Rebuilding 0x7F/0xFF with the
bias multiply mapped E4M3FN's NaN encoding to a finite ±480, while the backend's plain fp8→f16
software path maps the same byte to NaN; the upcast now selects a canonical NaN first. That costs
80 selects and ~14% instructions on gfx103x at this tile (7% on the mixed case), and up to ~38% on
gfx1100 where the WMMA dot leaves less room around the conversion. mxfp4 (no NaN encoding), e5m2,
bf16 and fp16 are unchanged. Correctness is pinned by the `scaled_upcast_fp8_sw_nan` chunk in
`test/Conversion/amd/tritongpu_to_llvm_rdna.mlir` (red when the select is removed). The same defect
in the plain fp8→bf16 software cast (`OcpF8ToBf16SW`, which `tl.cast(..., bf16)` takes on these
families) got the equivalent guard for both OCP formats — equality against 0x7F for E4M3FN, unsigned
>0x7C for E5M2 — plus a second equality with 0x7C that restores the signed infinity. That last half
was found on hardware: the software bit trick turned E5M2 +Inf into 65536.0. The guards are pinned
by the `fp8_e4m3_bf16_sw_nan` and `fp8_e5m2_bf16_sw_nan` chunks, and
[rdna2_fp8_nan_probe.py](file:///home/trave/code/triton/docs/rdna2/conformance/rdna2_fp8_nan_probe.py)
runs the whole set on the card (native gfx1031, RX 6700 XT): finite uniform dots for e4m3/e5m2/e2m1,
NaN operands and NaN scales, ±inf, ±448/±57344, and 448×448 through the scaled upcast — ten checks,
all pass. Grade: **MEASURED** on gfx1031.

The feature also survives the form a real MX GEMM uses — a 64x64x128 K-loop with scale loads. On
gfx1030/gfx1031 it compiles at `num_stages=2` and `3` (58,088 and 87,225 instructions for mxfp8),
with gfx1100 as the control (21,703 / 33,454). Three costs worth knowing before reaching for it:
compile time is tile-sensitive (36.6 s for 64x64x128 at stages=2 vs 3.7 s for 32x32x64), the
large-tile kernel is 5.6x the instruction count of the equivalent plain f16 dot (10,369), and
spilling is substantial — 16,304 B of scratch per lane with 4,349 folded spill stores at 4 warps,
7,692 B at 8 warps, versus 1,540 B for the plain f16 dot at the same tile; a 32x32x64 loop stays at
1,232 B. `waves_per_eu` 0/1/2 did not change the scratch size. The instruction and compile-time
numbers are partly the software upcast's cost; all of them are compile-time facts and the runtime
cost is unmeasured.

Instruction counts are totals and not comparable across families: gfx103x executes the dot as
per-lane FMA (hence only LDS traffic shows up in a dot-mnemonic histogram), while the Matrix-core
families issue one instruction for the whole tile. The control rows matter because they show the new
arm changes only the family that previously had no arm — RDNA3/RDNA4 and CDNA4 keep their
WMMA/MFMA paths unchanged.

Limits: this is compile-only, and the lit test pins the IR shape. Numerical validation of the
decomposed MX path on hardware is still owed; device work on this branch remains blocked, so it is
tracked with the other hardware-validation items rather than claimed.

## Cross-family gap that is not RDNA2's alone: `eviction_policy` is dropped

`tl.load(..., eviction_policy='evict_first'|'evict_last')` reaches Triton IR on the op (the TTGIR shows
the attribute) and no AMD lowering reads it — `EvictionPolicy` appears only in
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`, never under `third_party/amd`.
Measured, not inferred, with `performance/rdna2_eviction_policy_probe.py`, which compares *instruction
streams* (name-only metadata and `.loc` lines stripped — without that filter a byte diff reports ten
"differences" for a hint that changes nothing):

| arch | `evict_first` vs no hint | `evict_last` vs no hint | `.cg` control |
| --- | --- | --- | --- |
| gfx1030 (38 instr) | identical | identical | 8 lines differ |
| gfx1031 (38 instr) | identical | identical | 8 lines differ |
| gfx1100 (39 instr) | identical | identical | 8 lines differ |
| gfx942 (29 instr) | identical | identical | 4 lines differ |

So the hint is inert on RDNA2, RDNA3 and CDNA alike. Whether it *can* be honoured is a different question
per family, and for RDNA2 the answer is only half yes: the guide's L2 replacement fields give
`SLC=1, DLC=0` = "Stream", which ages lines out faster and is therefore the nearest encoding for
`evict_first` — and it is already exactly what `.cs` requests on this part. There is **no** "keep this
line" encoding, so `evict_last` cannot be honoured on RDNA2 at all, and an implementation that mapped it
to `SLC=0, DLC=0` would be silently wrong rather than conservative.

Not implemented, on purpose, for two reasons that are both about evidence rather than effort: making the
hint mean Stream on RDNA2 is a *policy claim* that needs a measured effect (the cache-policy timing
attempt is still queued behind the stuck device), and any change here alters CDNA and RDNA3 codegen too,
which this branch has no way to validate on hardware. The right next step is an upstream question —
"should `eviction_policy` be a no-op, an error, or a per-family mapping" — not a guess inside the pass.

The load-side row is Table 38 (`SLC=1, DLC=0` → L2 "Stream", L1 "Hit LRU"); the sentence explaining what
Stream does ("Hit leaves line in cache but do not reset age") is Table 39's, which is the store-and-atomic
table for the same two fields.

## What this changes about the plan

Nothing was found that can be fixed by editing a list. The two buffer-atomic gaps are gated behind
device validation that is currently blocked (my own stuck `rocprofv3` session still holds the
admission lock and I do not kill processes), and `supportDppBroadcast` turned out to be correct on
inspection — which is the useful negative: the family's reduce path is already on the best available
RDNA2 sequence.
