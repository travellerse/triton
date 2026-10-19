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

## What this changes about the plan

Nothing was found that can be fixed by editing a list. The two buffer-atomic gaps are gated behind
device validation that is currently blocked (my own stuck `rocprofv3` session still holds the
admission lock and I do not kill processes), and `supportDppBroadcast` turned out to be correct on
inspection — which is the useful negative: the family's reduce path is already on the best available
RDNA2 sequence.
