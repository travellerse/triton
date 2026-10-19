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
| `supportsBufferAtomicRMW()` — CDNA3/4, RDNA3/4m/4, CDNA5 | full integer set exists: `SWAP/CMPSWAP/ADD/SUB/CSUB/SMIN/UMIN/SMAX/UMAX/AND/OR/XOR/INC/DEC` (opcodes 48-61) plus 64-bit `_X2` forms | every buffer atomic falls back to the global path; the integer subset was built and pinned in an isolated worktree but never merged |
| `supportsBufferAtomicFMinMax(f32)` — RDNA3/4m/4/CDNA5 | `BUFFER_ATOMIC_FMIN` 63, `FMAX` 64, `FCMPSWAP` 62 exist, and `_X2` variants too | f32 atomic min/max cannot use buffer atomics on RDNA2 |
| `supportBitwidth32Elementwise()` — CDNA only | not checked against the ISA (see below) | elementwise ops are split to narrower vectors on RDNA2 |

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

* `supportBitwidth32Elementwise()` — I did not find an ISA statement about which element bit-widths
  the DPP/VALU path can process for elementwise (as opposed to memory) ops, so I am not claiming a
  gap. It would need either the LLVM `GCNSubtarget` equivalent (`hasReduction`, `dot1instr` etc. are
  CDNA-facing) or a measured A/B of a 32-bit-per-element vs split-vector kernel on device.
* `supportsWaveId()` — RDNA4/CDNA5 only. I first wrote that the guide exposes `SH_ID`/`SE_ID` fields
  and checked it: those tokens appear nowhere in the document, so it gives me no basis either way.
  The question is about cost and about what a wave id means under WGP mode, not about whether some
  register exists, and it stays unclassified.

## What this changes about the plan

Nothing was found that can be fixed by editing a list. The two buffer-atomic gaps are gated behind
device validation that is currently blocked (my own stuck `rocprofv3` session still holds the
admission lock and I do not kill processes), and `supportDppBroadcast` turned out to be correct on
inspection — which is the useful negative: the family's reduce path is already on the best available
RDNA2 sequence.
