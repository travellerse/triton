# Triton RDNA2 / gfx1031 Support — M0 Engineering Audit

**Scope of this round:** establish a verifiable baseline and locate the minimal
next fix. No production code in the AMD backend was modified. No dependency was
upgraded. No destructive git operation was performed.

**Target hardware:** AMD Radeon RX 6700 XT, `gfx1031` (RDNA2 / GFX10.3).
**Repository:** `/home/trave/code/triton`, branch `work/rdna2-gfx1031-m0`, base `b575c0c38d` (= `upstream/main`).
**Date:** 2026-10-12.

Evidence grades used throughout: **Cs** = confirmed by source, **Ct** = confirmed
by test (executed this round), **Or** = observed runtime behaviour, **H** =
hypothesis (reasoned, untested), **U** = unknown.

---

## 1. Executive summary

1. **Upstream's removal is 20 lines, and it has exactly one hard failure mode.**
   PR #12042 (`4c398335e0`) deleted the RDNA1/RDNA2 enum values and the two
   `major == 10` classification branches. The only surviving consequence is that
   `computeISAFamily()` now returns `ISAFamily::Unknown` for every `gfx10xx`
   arch (`third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:77-107`),
   and three LLVM-lowering passes hard-fail on `Unknown`. **A `gfx1031` compile
   today dies at `make_llir`; nothing else blocks it.** (Cs + Ct)

2. **"Restore the enum" is not sufficient, and a naive restore is actively
   unsafe.** Two independent code sites changed behaviour *around* the removal
   in ways that mean the parent commit's RDNA2 path no longer exists to be
   restored into:
   - Buffer resource-descriptor flags: RDNA2 used RDNA-style flags; `Unknown`
     now silently takes the **CDNA-style** descriptor while buffer ops remain
     enabled by default. Wrong descriptor flags are a *correctness* bug, not a
     build error. (Cs + Ct)
   - `TargetInfo::warpReduce()`: the version guard that explicitly excluded
     GFX10 from the DPP warp-reduce path was **deleted six days later** by
     `4a5147d919` (#12146, "Remove pre-CDNA2 target support"), together with
     `getIsaVersion()`, which no longer exists anywhere in the AMD backend.
     Re-admitting `gfx1031` therefore silently routes it into a reduction path
     that upstream had deliberately excluded for it. (Cs)

3. **The pre-removal "RDNA2 support" was never a real implementation.** In the
   parent commit the *entire* RDNA1/RDNA2 footprint was 4 code sites plus 2 lit
   `RUN` lines. There was never an RDNA2 MMA path (`AccelerateAMDMatmul.cpp`
   contains no `RDNA1`/`RDNA2` case in the parent either), no RDNA2 layout work,
   and no RDNA2 hardware test. Restoring it recovers *classification*, not
   capability. (Cs)

4. **`gfx1030` and `gfx1031` are the same ISA family by Triton's own rules and
   by LLVM's codegen** — verified, not assumed. Triton parses the last digit as
   hex and matches only `major==10 && minor==3`, so `gfx1030/1031/1032` all map
   identically; the pre-removal `triton-opt` emits byte-identical buffer
   descriptors for `gfx1030` and `gfx1031`; and `llc` produces identical
   assembly for both on the probed kernel. Remaining differences are SKU-level
   (CU count, bus width), which Triton does not use for instruction selection.
   (Cs + Ct) — with the caveat that only a trivial kernel was compared.

5. **The machine already runs Triton kernels on this GPU — with the old
   package.** Installed Triton 3.5.1 compiled and correctly executed masked
   elementwise add for fp32/fp16/bf16 at n=1024 on the RX 6700 XT. That proves
   the ROCm 7.2.53211 / LLVM / HIP / driver / hardware stack is functional and
   that the gap is purely in the source tree's target classification. (Or + Ct)

6. **Environment caveat that will bite every future test:** `HSA_OVERRIDE_GFX_VERSION=10.3.0`
   is already set in the user's environment. HIP consequently reports the device
   as **`gfx1030`, not `gfx1031`** (`amdgpu-arch` reports the truth). Triton's
   driver takes its arch from `gcnArchName`, so on this machine the restored
   path will be exercised as `gfx1030` unless `TRITON_OVERRIDE_ARCH=gfx1031` is
   set per-invocation. (Or)

**Recommended first fix (§11), in one sentence:** restore `RDNA2` classification
in `computeISAFamily()` **and, in the same change,** re-add `ISAFamily::RDNA2` to
the RDNA list in `BufferOpsEmitter.cpp:70` and to `isRDNA()` in
`TargetFeatures.cpp:392` — never the enum alone — then gate the `warpReduce`
DPP question behind a test instead of behind an assumption.

---

## 2. Repository and git state (Stage A)

Verified facts (full command output in `environment-baseline.txt`):

| Item | Value | Grade |
|---|---|---|
| Repo root | `/home/trave/code/triton`, is the Triton repo | Ct |
| `origin` | `git@github.com:travellerse/triton.git` (fork) | Ct |
| `upstream` | `git@github.com:triton-lang/triton.git` — already configured, URL verified, not modified | Ct |
| Pre-work branch | `dev` @ `b575c0c38d` ("[AMD] MFMA scheduler: run on gfx942… (#12229)", 2026-10-11) | Ct |
| Divergence | `dev` == `origin/dev` == `origin/main` == `upstream/main` (0 ahead / 0 behind) | Ct |
| Working tree | clean, `git status --porcelain` → 0 lines | Ct |
| Removal commit in history | `git merge-base --is-ancestor 4c398335e0 HEAD` → true | Ct |
| Both reference commits present locally | `git cat-file -t` → `commit` for both | Ct |
| No pre-existing `*rdna*` branch | `git branch -a --list '*rdna*'` → empty | Ct |
| `docs/rdna2/` | did not exist; created this round (no file overwritten) | Ct |
| Branch created | `work/rdna2-gfx1031-m0` from clean `dev` — safe because the tree was clean | Ct |

No `pull`, `rebase`, `merge`, `reset`, `stash`, `clean`, or `push` was run. No
fetch was even necessary — both reference commits were already local.

---

## 3. Verified runtime environment (Stage B)

Highlights; full transcript in [`environment-baseline.txt`](./environment-baseline.txt).

- **OS:** CachyOS (Arch-like, rolling), kernel `7.2.9-2-cachyos`, x86_64.
  CPU i5-12400F (12 threads). 31 GiB RAM with **22 GiB already used + 16 GiB
  swap used before this session** — a real constraint for the §10 rebuild.
  `/home` 783 GiB free.
- **ROCm:** `/opt/rocm`, `hipcc` → HIP `7.2.53211-3d9ef42`, AMD clang
  `22.0.0git`. `rocm-smi`, `rocminfo`, `amdgpu-arch` all present.
  System clang `23.1.1` (separate from ROCm's).
- **Python:** `3.14.7` at `/usr/bin/python`. `python3.13` exists but has no
  working torch.
- **PyTorch:** `2.14.1`, `torch.version.hip = 7.2.53211`, `cuda.is_available() → True`.
- **GPU:** `AMD Radeon RX 6700 XT`; `amdgpu-arch → gfx1031`; HIP
  `gcnArchName → gfx1030` (override); warp 32; 64 KiB LDS/CU; 3 MiB L2;
  192-bit bus; 12,868,124,672 B VRAM; `multi_processor_count = 20`.
- **Triton, installed:** `3.5.1` at `/usr/lib/python3.14/site-packages/triton/`
  → `GPUTarget(backend='hip', arch='gfx1030', warp_size=32)`.
- **Triton, source:** `/home/trave/code/triton` — **not importable in this
  session**; its only build artefact is stale (see §8).
- **Relevant env:** `HSA_OVERRIDE_GFX_VERSION=10.3.0` (pre-existing, recorded
  not set), `ROCM_PATH=/opt/rocm`, `TRITON_OVERRIDE_ARCH` unset,
  `AMDGCN_USE_BUFFER_OPS` unset (→ default enabled), no visible-device limits,
  `TRITON_CACHE_DIR` unset.
- No tool was installed, upgraded, or removed. No global env was changed.

---

## 4. Analysis of the upstream removal (Stage C)

### 4.0 The whole change

```
git show --stat 4c398335e0   ->  7 files, +9 / -20
 test/TritonGPU/amd/amd-convert-buffer-ops-atomic-fadd.mlir    |  2 +-
 test/TritonGPU/amd/amd-convert-buffer-ops-atomic-rmw.mlir     |  2 +-
 third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h |  2 --
 third_party/amd/language/hip/utils.py                         |  4 ++--
 third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp| 12 ++----------
 third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp   |  4 ++--
 third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp         |  3 +--
```

Both `+2` lines are **newly added `gfx90a` `RUN` lines** — `gfx90a` had no
coverage in these two tests in the parent — while both `gfx1030` `RUN` lines were
deleted. So the commit is not a reorder: it *added* CDNA2 coverage in the same
breath as it *removed* GFX10 coverage. Net effect on the buffer-ops atomic test
matrix: `gfx90a` gained, `gfx1030` lost. (Cs)

**Q1 — which enums were deleted?** `ISAFamily::RDNA1` and `ISAFamily::RDNA2`
from `TargetFeatures.h:19-20`. `RDNA3`, `RDNA4m`, `RDNA4`, `GFX1250`, and the
CDNA values remain. (Cs)

**Q2 — what changed in ISA identification?** In `computeISAFamily()`
(`TargetFeatures.cpp:77-107`) these two branches were removed:

```cpp
if (major == 10 && minor == 3)   // gfx103x  -> RDNA2
  return ISAFamily::RDNA2;
if (major == 10 && minor == 1)   // gfx101x  -> RDNA1
  return ISAFamily::RDNA1;
```

`parseGfxArch()` was **not** touched. It still parses `gfx1031` successfully as
`(major=10, minor=3, patch=1)`, with the last digit read as **hex**. So the arch
string is recognised; only the family assignment fails. The fallthrough is
`return ISAFamily::Unknown;` (`:107`). (Cs, Ct — verified by standalone
transcription, §5.1)

**Q3 — which predicates / paths were affected?** Four, all in one direction
("RDNA2 stops being special-cased"):

| Site | Change | Effect on gfx10xx |
|---|---|---|
| `TargetFeatures.cpp:287-300` `supportsBufferAtomicFMinMax` | RDNA1/RDNA2 dropped from the f32 list and the f64 list | was `true`, now `false` → generic (non-buffer) atomics. Same answer `Unknown` gives for other reasons; low impact. (Cs) |
| `TargetFeatures.cpp:392` `isRDNA()` | RDNA1/RDNA2 cases removed | `isRDNA(gfx1031)` `true` → `false`. **Two consumers, both behavioural** — §5.3. (Cs) |
| `BufferOpsEmitter.cpp:69-76` `createResourceDescriptor` | RDNA2 dropped from the RDNA flag list | descriptor flags `822243328` → `159744` (CDNA-style). **Correctness risk**, §5.2. (Cs + Ct) |
| `TargetInfo.cpp:762-778` `getSharedLdStTiles` | `case ISAFamily::RDNA2:` deleted from the CDNA3/RDNA3/RDNA4m group | the `{1,2,20}` 128-bit LDS tile is replaced by the `{{},{}}` default. Performance/codegen shape, not correctness. (Cs) |
| `language/hip/utils.py:42-52` `smid()` | `'gfx10' in target_arch` branch removed | `ValueError: smid is not supported on gfx1030`. **A user-visible API regression** for any code calling `tl.extra.hip.smid()`. (Cs) |
| `TargetInfo.cpp` `warpReduce` | **comment only** | no functional change *in this commit* — but see §4.4, a later commit removed the real gate. (Cs) |

**Q4 — which test coverage disappeared?** Two `RUN` lines, both `gfx1030`, both
for buffer-ops **atomics**:
- `amd-convert-buffer-ops-atomic-fadd.mlir`: `gfx1030 → NO-F16,NO-BF16,NO-F32,NO-F64`
- `amd-convert-buffer-ops-atomic-rmw.mlir`: `gfx1030 → GENERIC`

Note what these asserted: that on `gfx1030` buffer atomics are **not** used
(GENERIC / NO-*). They checked an *absence*. There was never a positive RDNA2
lowering test, never an `accelerate-amd-matmul` RDNA2 case, never a
`python/test` RDNA2 gate. Today the repo has 0 `gfx10xx` lit `RUN` lines
(arch coverage: gfx90a 2, gfx942 54, gfx950 42, gfx1100 12, gfx1150 1, gfx1170
11, gfx1200 9, gfx1250 44). (Ct)

**Q5 — what does the removal message prove, and not prove?** It proves upstream
decided the RDNA1/RDNA2 *classification* was not worth maintaining, and it is
consistent with "nobody tests this hardware upstream". It does **not** prove any
of: that RDNA2 kernels were producing wrong results; that RDNA2 could not run;
that a specific hardware limitation forced it; or that restoring the classification
would work. The commit message carries no technical justification, and the diff
is a maintenance subtraction, not a bug fix. Reading it as "RDNA2 is impossible"
would be an unfounded inference. (Cs for what it says; U for the motivation)

**Q6 — was the parent implementation actually complete? No.** Evidence:
`git grep -E "RDNA1|RDNA2" 6c14a3d -- third_party/amd test` returns **only**
these files: `TargetFeatures.h` (2), `TargetFeatures.cpp` (7),
`BufferOpsEmitter.cpp` (1), `TargetInfo.cpp` (2), plus a vendored HIP header
doc comment. That is the entire historical footprint. In particular:
- `AccelerateAMDMatmul.cpp` has **zero** RDNA1/RDNA2 references in the parent —
  `getMfmaVersion()`/`getWmmaVersion()` never handled RDNA2, so `tt.dot` on
  RDNA2 always fell through to the FMA lowering. RDNA2 matrix acceleration was
  never implemented, and (§7) cannot be implemented via WMMA because the
  hardware lacks it. (Cs + Ct)
- No Python-side arch gate ever included gfx10: `compiler.py:166-214`
  (`is_pingpong_schedule_enabled`, `is_in_thread_transpose_enabled`,
  `is_async_copy_enabled`, `is_coexec_scheduler_enabled`,
  `is_expert_scheduling_enabled`, `is_fpsan_supported`, `is_consan_supported`)
  are allowlists over gfx942/gfx950/gfx110x/gfx115x/gfx117x/gfx120x/gfx1250.
  `RDNA2` never appeared. (Cs)
- No hardware-execution test existed. The two deleted `RUN` lines are `triton-opt`
  lit tests — CPU-only codegen checks. (Cs)

So: **the parent commit is a valid *source comparison baseline*, and a
classification seed, but it is not evidence of a working RDNA2 backend.**

**Q7 — what stays missing after restoration?** (each graded)
- MMA acceleration: permanently unavailable on RDNA2 via WMMA — hardware lacks
  `v_wmma_*` (Ct, §7). `dot` will be FMA-based. **H** that this dominates any
  matmul-shaped inference workload; measurable in M2.
- Async copy / pingpong / in-thread transpose / coexec / expert scheduling /
  fpsan / consan: gated off by Python allowlists that never included gfx10 (Cs).
  Restoring them is separate feature work, not restoration.
- `smid()` on gfx10xx: raises `ValueError` (Cs). **H** that restoring it is
  undesirable — the repo's own AGENTS.md forbids hardware-ID-derived source
  scalars in user programs; recommend leaving it removed.
- Buffer atomics (`buffer_atomic_*`) for RDNA2: excluded both before and after
  (the deleted lit checks asserted exactly this). Generic atomics must work. (Cs)
- The `warpReduce` DPP exclusion mechanism **no longer exists to be restored**
  — `getIsaVersion()` is gone from the AMD backend entirely (§4.4). (Cs)

### 4.4 The second removal nobody mentioned in #12042

`4a5147d919 "[AMD] Remove pre-CDNA2 target support (#12146)"`, 2026-10-06 — six
days after the RDNA removal — deleted, from `TargetInfo.cpp`:

```cpp
-  // DPP warp reduce requires gfx90a+ (CDNA2+) or gfx11+ (RDNA3+).
-  // Pre-CDNA2 GFX9 (gfx906/gfx908) and GFX10 (RDNA1/2) are excluded.
-  auto v = getIsaVersion();
-  if (!((v.Major == 9 && (v.Minor > 0 || v.Stepping >= 0xa)) || v.Major >= 11))
-    return false;
```

and also deleted `TargetInfo::getIsaVersion()` itself. `grep -rn getIsaVersion
third_party/amd/{include,lib,backend}` now returns **nothing** (Cs).

Consequence: in today's tree, `warpReduce()` (`TargetInfo.cpp:453+`) proceeds
whenever `reduceLaneIdMask == getWarpSize()-1` and a single combiner exists, and
emits the `DPP ROW_SHR0` ladder + (because `supportDppBroadcast()` is `false`
for non-CDNA) `ROCDL::PermlaneX16Op` + `ROCDL::ReadlaneOp`. This is the RDNA3
design. **Restoring `RDNA2` classification re-admits gfx1031 into that path
without any of the historical vetting**, because the guard that used to keep it
out is gone. Whether it is correct for gfx1031 is **U** (§7). This is the single
most under-appreciated risk in the restoration and it is why §10 sequences a
reduce test early rather than late.

---

## 5. Current AMD backend structure and gap inventory (Stage D)

### 5.1 The one hard blocker, and the exact failure

Three passes gate on `Unknown` and are unconditional in the `make_llir`
pipeline (`third_party/amd/backend/compiler.py:542,553,567`):

| File:line | Guard | Message |
|---|---|---|
| `TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:90-92` | `targetInfo.getISAFamily() == Unknown` | `mod.emitError("unsupported target: '") << gfxArch << "'"` |
| `TritonAMDGPUToLLVM/Membar.cpp:26-28` | same | same |
| `TritonAMDGPUToLLVM/ConvertWarpPipeline.cpp:1040-1042` | same | same |

A standalone transcription of the current `parseGfxArch()` +
`computeISAFamily()` (`/tmp/rdna2_probe/family_probe.cpp`, compiled and run this
round) gives **(Ct)**:

```
gfx942    parse=9.4.2    family=CDNA3
gfx90a    parse=9.0.10   family=CDNA2
gfx1030   parse=10.3.0   family=Unknown     <-
gfx1031   parse=10.3.1   family=Unknown     <-
gfx1032   parse=10.3.2   family=Unknown     <-
gfx1010   parse=10.1.0   family=Unknown     <- (RDNA1)
gfx1100   parse=11.0.0   family=RDNA3
gfx1151   parse=11.5.1   family=RDNA3
gfx1170   parse=11.7.0   family=RDNA4m
gfx1200   parse=12.0.0   family=RDNA4
gfx1250   parse=12.5.0   family=GFX1250
```

**Diagnosis:** the arch string is recognised (parse succeeds) — the failure is
purely family classification, and it manifests as a hard `emitError` in
`make_llir`, i.e. after `ttir`/`ttgir` are valid and before any LLVM IR exists.
Compilation fails; it does not fail at launch, and it does not produce
wrong-but-runnable code at this stage. That matches the brief's
"architecture string recognised but no target path" category, not the
"launch-time failure" category. (Cs + Ct)

Everything upstream of `make_llir` (TritonGPU layout selection, `tt.dot` →
`ttg.convert_layout`, scheduling passes that key on `TargetFeatures` softly)
runs on `Unknown` without error, which is why the failure looks sudden.

### 5.2 Silent-wrong risk #1: buffer descriptor flags (highest severity)

`BufferEmitter::createResourceDescriptor` (`BufferOpsEmitter.cpp:69-76`):

```cpp
uint32_t flags = (7 << 12) | (4 << 15);                       // CDNA-style = 159744
if (llvm::is_contained({ISAFamily::RDNA3, ISAFamily::RDNA4m,
                        ISAFamily::RDNA4, ISAFamily::GFX1250},
                       targetInfo.getISAFamily())) {
  flags |= (1 << 24);                                          // "Reserved to 1 (RDNA)"
  uint32_t oob = 3;                                            // bits 28-29, OOB select (RDNA only)
  flags |= (oob << 28);
}                                                              // RDNA-style = 822243328
```

Measured with the **pre-removal** `triton-opt` (built ~`61c02bcdba`, which still
contains RDNA2) against `test/Conversion/amd/buffer_load_store.mlir`, via
`--convert-triton-amdgpu-to-llvm=gfx-arch=<arch>` **(Ct)**:

| `gfx-arch` | family then | rsrc flags emitted |
|---|---|---|
| `gfx1030` | RDNA2 | **822243328** |
| `gfx1031` | RDNA2 | **822243328** (identical to 1030) |
| `gfx1100` | RDNA3 | 822243328 |
| `gfx942` | CDNA3 | 159744 |

and `test/Conversion/amd/buffer_load_store.mlir:222` pins the RDNA expectation
as `// RDNA: %[[FLAGS:.*]] = llvm.mlir.constant(822243328 : i32)`.

Why this is the dangerous one: buffer ops are **on by default**
(`ConvertToBufferOps.cpp` has no family gate; knob `AMDGCN_USE_BUFFER_OPS`
defaults to enabled), and `Unknown` does **not** disable them. So a future
change that adds only the enum and nothing else produces, for gfx1031, buffer
loads/stores carrying a CDNA-format descriptor — including bits 12-14 that the
in-code comment describes as RDNA-meaningful and OOB-select bits that are
"RDNA only". Result: a kernel that compiles and launches, with undefined
out-of-bounds behaviour on RDNA2 hardware. **H** for the precise hardware
outcome (needs a real-GPU test); **Cs + Ct** for the flag divergence itself.

Caveat on the test: the `triton-opt` binary is from 2026-09-02 while the `.mlir`
inputs are from HEAD, and stderr showed `error: dialect 'tt' provides no
attribute…` at `buffer_load_store.mlir:13` for every arch (a pass-through
version mismatch, not arch-specific). Output was still produced for all four
archs and the flag constants are the direct product of the code under study, so
the comparison is sound; the *warning* means the full lowering was not
necessarily complete. Grade **Ct** with that limitation stated.

### 5.3 Behavioural risk #2: `isRDNA()` consumers

Only two consumers exist in the whole backend (Cs, exhaustive grep):

1. **`TritonAMDGPUToLLVM/Utility.cpp:172` `shuffleXor`.**
   - RDNA branch (`mask < 16`): DPP `ROW_XMASK0|mask`; (`16 ≤ mask < 32`):
     `emitPermlaneX16Xor` → `ROCDL::PermlaneX16Op`.
   - CDNA branch: DPP row/quad-perm ladder.
   - else → `b.permute` (`ds_bpermute_b32`).
   `Unknown` falls to the `else`: **still valid code, slower**. Pre-removal
   RDNA2 got the register-only DPP/permlane sequence; post-removal `Unknown`
   pays an LDS round-trip per cross-lane step, which hits reductions directly.
   Confirmed executable on gfx1031: `clang --target=amdgcn -mcpu=gfx1031`
   **selects `v_permlanex16_b32`**, while `-mcpu=gfx942` errors
   `needs target feature gfx10-insts` — so the permlane path is *available* on
   RDNA2 and the RDNA branch is not obviously wrong. (Ct) `ds_bpermute_b32` also
   compiles on gfx1031. Grade: **Cs + Ct**, perf-only impact.

2. **`TritonAMDGPUTransforms/ScheduleLoops.cpp:95-97`** —
   `filterSmallVectors = !isCDNA4() && !isRDNA() && !isGFX1250()`.
   RDNA2 → `false` before, `Unknown` → `true` now: indirect-load pipelining
   becomes more conservative for RDNA2-sized vectors. Perf only, and arguably a
   regression relative to the parent. (Cs, **H** on magnitude)

`UpdateAsyncWaitCount.cpp:425` (`!isCDNA() && !isGFX1250()` → bail) behaves
identically for `Unknown` and `RDNA2`. No change. (Cs)

### 5.4 Geometry fallbacks happen to be right

`TargetFeatures.cpp:135-144` `getWarpSize()` → CDNA2/3/4 = 64, **default 32**;
`:151-161` `getSharedMemorySize()` → **default 64 KiB**; `:163-171` partition 0;
`TargetInfo.cpp:132-139` `getSharedMemoryBanks()` → default 32. For gfx1031
those defaults are correct (wave32 measured `warp_size=32`; 64 KiB LDS measured
in §3). So `Unknown` does **not** misreport the machine model. (Cs + Or)

`getMinDotSize` / `compiler.py:160-164` returns `(1,1,1)` for all targets — the
FMA fallback is architecture-neutral, so `dot` compiles regardless. (Cs)

### 5.5 Launch path and codegen plumbing are arch-agnostic

- `driver.py:355-360` `get_current_target()`:
  `arch = knobs.runtime.override_arch or device_properties['arch']` →
  `GPUTarget("hip", arch.split(':')[0], warpSize)`. `driver.c:460` sources it
  from `props.gcnArchName`. **There is no allowlist, regex, or version check
  anywhere on the AMD Python path** that would reject gfx10xx (Cs, exhaustive
  search) — so nothing needs restoring for target *propagation*.
- `compiler.py:286-288`: `gfx_major = int(base_arch[3:-2])`,
  `warp_size = 32 if gfx_major >= 10 else 64` — keyed on the arch **string**, not
  the family, and already correct for gfx1031. (Cs)
- `compiler.py:619-631, 779-802`: `mcpu`/`processor = options.arch`,
  `target_features=''` (only `+xnack` under ASAN), `__oclc_wavefrontsize64`
  derived from `options.warp_size`. No family dependency. (Cs)
- `wgp_cu_mode` (`compiler.py:278, 671-677`) adds `+cumode` for **any** arch; the
  option's docstring says "for gfx11 and gfx120x; ignored on other targets",
  which the implementation does not enforce. **U** whether `+cumode` is
  meaningful/valid on gfx1031 — my probe was inconclusive because
  `llc -mattr=<unknown>` is silently accepted (`+bogus-feature-xyz` also
  "accepted" on all three targets), and `-mattr=help` lists features without
  enabled state. Explicitly **not** claimed either way.

### 5.6 Test-side gaps

- `python/triton/_internal_testing.py:137-167` defines `is_hip_rdna3/4m/4` and
  `is_hip_rdna() = rdna3 or rdna4m or rdna4`. **There is no `is_hip_rdna2()`**,
  so on a gfx1031/gfx1030 device `is_hip_rdna()` is `False` and `is_hip_cdna()`
  is `False` — RDNA2 is invisible to the entire Python test gating layer, and
  every `is_hip_rdna()`-guarded test silently skips. (Cs)
- Zero `gfx10xx` lit `RUN` lines remain (§4 Q4).
- `third_party/amd/python/test/*` targets gfx942/gfx950/gfx1250 only.

---

## 6. Prioritised problem list

| # | Problem | Severity | Grade | Blocking M1? |
|---|---|---|---|---|
| P1 | `computeISAFamily()` → `Unknown` for `major==10`; three passes `emitError` in `make_llir`. Nothing compiles for gfx10xx. | Blocker | Cs+Ct | Yes |
| P2 | Buffer descriptor flags: `Unknown` → CDNA-style `159744` instead of RDNA-style `822243328`, while buffer ops stay enabled by default. Silent wrong OOB semantics on real hardware. | Blocker (correctness) | Cs+Ct (flag divergence); **H** on hardware consequence | Yes — must land with P1 |
| P3 | `warpReduce` DPP/permlane/readlane ladder is no longer guarded against GFX10 (guard deleted by #12146, `getIsaVersion()` gone). Restoring P1 re-admits gfx1031 unvetted. | High | Cs on code; **U** on correctness | Yes — needs a reduce test, not an assumption |
| P4 | `isRDNA()` no longer true → `shuffleXor` degrades to `ds_bpermute`; `ScheduleLoops` filters small vectors. | Medium (perf) | Cs+Ct | No |
| P5 | `getSharedLdStTiles` loses the `{1,2,20}` 128-bit LDS tile RDNA2 shared with RDNA3. | Medium (perf/codegen shape) | Cs | No |
| P6 | No `is_hip_rdna2()` → RDNA2 cannot be gated into the Python test suite. | Medium (testability) | Cs | Yes for M1 exit criteria |
| P7 | `smid()` raises `ValueError` on gfx10xx (user-facing API regression). | Low; **recommend not restoring** (AGENTS.md forbids hardware-ID-derived scalars) | Cs | No |
| P8 | Source tree is unbuildable-as-tested: only artefact is `triton-opt` from 2026-09-02; 305 native files drifted. | Process | Ct | Yes — M1 step 0 |
| P9 | `HSA_OVERRIDE_GFX_VERSION=10.3.0` masks the real part: everything sees `gfx1030`, never `gfx1031`. | Process / measurement validity | Or | Yes — must be explicit in every test |
| P10 | RDNA2 has no WMMA/MFMA: `dot` is FMA-only. Not a defect; a hardware ceiling. | Scope definition | Ct | No — defines the perf roadmap |

---

## 7. Evidence register for the headline claims

| Claim | Evidence | Location / command | Grade | Confidence | Next verification |
|---|---|---|---|---|---|
| gfx10xx classifies as `Unknown` in current source | Standalone compile+run of transcribed logic | `/tmp/rdna2_probe/family_probe.cpp`; `TargetFeatures.cpp:77-107` | Ct | High | After a real build: lit `gfx-arch=gfx1031` should print `unsupported target: 'gfx1031'` |
| …and that is fatal, not soft | Read of all 3 guards + pipeline order | `TritonGPUToLLVM.cpp:90`, `Membar.cpp:26`, `ConvertWarpPipeline.cpp:1040`, `compiler.py:542,553,567` | Cs | High | `triton-opt` reproducer |
| RDNA2 needs RDNA-style descriptor flags | Pre-removal binary emits `822243328` for gfx1030/gfx1031; CDNA emits `159744`; FileCheck pins RDNA value | `build/.../bin/triton-opt` on `test/Conversion/amd/buffer_load_store.mlir`; expectation at `:222` | Ct (caveat §5.2) + Cs | Medium-high on the divergence; **H** on hardware effect | Lit test for gfx1030/gfx1031 + real-GPU OOB mask test |
| gfx1030 ≡ gfx1031 for Triton's purposes | Hex patch digit + `major==10 && minor==3`; identical flags; identical `llc` asm | `family_probe.cpp` output; flags table; `diff asm1030.s asm1031.s` (only `.amdgcn_target` differs) | Cs + Ct | Medium (only a 1-kernel, load/store probe) | Extend to reduce, dot, atomic lit cases |
| `v_permlanex16_b32` is available on gfx1031 | clang selects it for `-mcpu=gfx1031`; gfx942 fails on missing `gfx10-insts` | `/tmp/rdna2_probe/builtins2.cpp` | Ct | High | n/a |
| DPP warp-reduce ladder is *reachable* for gfx1031 post-restore | Guard + `getIsaVersion()` deleted by `4a5147d919`; grep finds no `getIsaVersion` in `third_party/amd` | `git show 4a5147d919 -- …/TargetInfo.cpp`; `TargetInfo.cpp:453+` | Cs | High for reachability; **U** for correctness of the emitted sequence on RDNA2 | lit reduce test for gfx1031; then GPU `tl.sum` vs torch |
| RDNA2 has no WMMA 16x16x16 | Assembler: `instruction not supported on this GPU (gfx1031)` vs gfx1100 accepting the opcode | `/tmp/rdna2_probe/isa/wmma_f16.s` × `-mcpu=gfx1031/gfx1100` | Ct (capability messages) | Medium-high (operand syntax was wrong; the *capability* vs *operand* error distinction is what carries it) | Confirm with `llvm-mc -mcpu=gfx1031` once a full LLVM tool is available |
| Triton's LLVM knows gfx1031 | `llc -mcpu=gfx1031` clean vs `gfx1099` "not a recognized processor" | `/tmp/rdna2_probe/probe.ll` | Ct | High | n/a |
| Hardware+stack runs Triton RDNA2 kernels today | fp32/fp16/bf16 masked add, compile+launch+match=True, `.amdgcn_target …-gfx1030` | `/tmp/rdna2_probe/stage_f_probe.py`, installed Triton 3.5.1 | Or | High **for the installed package**, zero for the source tree | Re-run against a built source tree |
| `+cumode` behaviour on gfx1031 | probe invalidated (unknown `-mattr` silently accepted) | — | U | — | Needs `triton-opt`/full LLVM, not `-mattr` guessing |
| RDNA2 kernel perf vs CDNA/`wgp_cu_mode` effects | no benchmark run | — | U | — | M2 |

Discarded evidence, recorded for honesty: `/tmp/rdna2_probe/isa*.s` results for
`ds_bpermute`, `dpp_row_b32`, `dot2_f16`, `readlane`, `permlane16_swap` all came
back "invalid instruction", which in those cases was **my operand syntax**, not
a capability statement. Those results are excluded from every conclusion above.
The `-mattr` feature probe and the `-mattr=help` enabled-state probe likewise
produced no usable signal.

---

## 8. Tests executed and not executed

**Executed this round**
1. `git` state / ancestry / divergence checks (§2). — Ct
2. Read-only environment and HIP/torch queries. — Or
3. Minimal GPU probe, installed Triton 3.5.1, masked elementwise add fp32/fp16/bf16,
   n=1024, one launch each, compared to `x+y`: **all three correct.** — Or
   *Gated on idleness:* the first sample showed the user's ComfyUI job active at
   98% GPU with 11.8/12.9 GiB VRAM (`main.py --enable-triton-backend … :8188`).
   All GPU work was deferred until `rocm-smi --showpids` reported *No KFD PIDs
   currently running* and VRAM use fell to ~1.2 GiB. Nothing was killed, and no
   stress, large-allocation, long-running, or benchmark workload was executed.
4. `llc` target-recognition and gfx1030-vs-gfx1031 codegen diff. — Ct
5. clang builtin instruction-availability probes. — Ct
6. Standalone `parseGfxArch`/`computeISAFamily` transcription test. — Ct
7. Pre-removal `triton-opt` buffer-descriptor flag A/B across 4 archs. — Ct (caveat §5.2)

**Not executed, and why**
- **lit tests against current source.** `triton-opt` in
  `build/cmake.linux-x86_64-cpython-3.14/bin` is dated 2026-09-02 (≈`61c02bcdba`),
  i.e. pre-removal; testing with it would verify code that no longer exists in the
  tree. `ninja -n triton-opt` reports `Re-running CMake…`, and
  `git diff --name-only 61c02bcdba..HEAD` shows **305** changed `.cpp/.h/.td`
  files (**120** under `third_party/amd`). That is a large rebuild, excluded by
  task constraint ("don't start a time-consuming full compile"). Nothing was
  compiled this round. Note the LLVM dependency is *not* a blocker:
  `CMakeCache.txt` points at `~/.triton/llvm/llvm-b010a18d-ubuntu-x64-1`, which
  is present on disk.
- **Source-tree GPU execution.** Requires the rebuild above; would otherwise
  have been allowed now the GPU is idle.
- **`rocminfo`**, full test suites, benchmarks, memory-pressure tests, multi-GPU
  or long-running jobs. — out of scope / declined by constraints 6 and 7.
- **Any change to Triton cache, git remotes, env vars, or installed packages.** — none made.

---

## 9. Known limitations and open questions

- **L1** — No conclusion about the *source tree* has any runtime backing this
  round. Every source-tree claim is Cs or Ct-on-isolated-logic; the only GPU
  evidence is about the installed 3.5.1 package.
- **L2** — `gfx1030` and `gfx1031` were compared on a 1-kernel load/store probe
  and on buffer descriptor flags. The full lowering was not compared. Do not
  generalise beyond "same ISA family, same flags" until a reduce/dot/atomic
  lit comparison exists.
- **L3** — Every measurement on this host is taken through
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`, so Triton sees `gfx1030`. A genuine
  `gfx1031` run needs `TRITON_OVERRIDE_ARCH=gfx1031` (Triton's own knob,
  compile-time only) — and results must be labelled with which one produced them.
- **L4** — `multi_processor_count = 20` on a 40-CU part indicates RDNA2 WGP-mode
  accounting. **H** that this affects grid sizing / persistent-kernel occupancy
  assumptions for Triton kernels, and that it interacts with `wgp_cu_mode`.
  Unmeasured; relevant to M2, not M1.
- **L5** — Whether the `DPP ROW_SHR → PermlaneX16 → Readlane` ladder is correct
  for a 32-lane RDNA2 wave is **U**. RDNA3 runs it today at wave32, which is
  suggestive but not proof (different DPP/`gfx10-insts` semantics).
- **L6** — No evidence either way on gfx1031 `atomic_*` correctness, cache
  modifiers on RDNA2 buffer ops (`amd-buffer-cache-modifiers-rdna3.mlir` exists
  only for RDNA3), fp8 conversion paths, or `local_alloc`/swizzle behaviour.
- **L7** — The motivation behind #12042 is **U**; without it we cannot distinguish
  "unmaintained" from "known broken on hardware". Treat restoration as needing
  independent correctness proof.

---

## 10. M1 — minimal restoration plan

Guiding rule: **restore classification and every consumer of it atomically, and
prove each with a test before claiming the next.** Never mark a whole family's
capabilities as supported because one arch string was accepted. Scope is
`gfx103x` (RDNA2) only; RDNA1 (`gfx101x`) is deliberately out of scope this
milestone.

**Step 0 — build the verification tool (no production change).**
`cd build/cmake.linux-x86_64-cpython-3.14 && ninja triton-opt` (after `make`
configures). Files: build tree only. Why: every step below is unverifiable
otherwise; lit tests need no GPU. Risk: long compile, ~22 GiB RAM already in use
— run it when memory headroom is real. Success: `triton-opt` newer than HEAD.
Rollback: none needed (artefacts only).

**Step 1 — reproduce the failure, and pin it with a test first.**
Add a `gfx1031` `RUN` line to an existing AMD conversion test (or run
`triton-opt --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1031` on a small input)
and confirm the expected `unsupported target: 'gfx1031'`. Files: existing test
files. Why: the brief demands tests prove the defect before code changes. Risk:
none. Success: red test that fails for exactly the P1 reason.

**Step 2 — restore classification + descriptors together (the P1+P2 pair).**
- `TargetFeatures.h:19-20`: re-add `RDNA2` (not `RDNA1`).
- `TargetFeatures.cpp:77-107`: re-add `if (major == 10 && minor == 3) return
  ISAFamily::RDNA2;`.
- `BufferOpsEmitter.cpp:70`: add `ISAFamily::RDNA2` back to the RDNA flag list.
- `TargetFeatures.cpp:392 isRDNA()`: add `case ISAFamily::RDNA2:`.
  Files above, all in one commit. Why paired: steps 1 and 2 without 3 = compiles
  with wrong descriptors (silent-wrong); without 4 = loses the RDNA shuffle path.
  Tests: `gfx1030`/`gfx1031` `RUN` lines on
  `test/Conversion/amd/buffer_load_store.mlir` expecting the `RDNA` prefix
  (`822243328`), plus re-adding the two deleted atomic `RUN` lines.
  Risk: medium — restores behaviour that has had no hardware validation for a
  year. Failure mode: buffer loads/stores produce wrong data or fault on device.
  Rollback: single commit revert.

**Step 3 — settle P3 (warp reduce) by test, not by assumption.**
Files: `TargetInfo.cpp:453+` (`warpReduce`). Options, chosen by evidence:
(a) accept the RDNA3 ladder if lit+GPU tests agree with torch; (b) if it fails,
re-introduce an explicit family predicate (`isRDNA3OrNewer()` or a family check
in `warpReduce`) to route RDNA2 to the generic shuffle reduce — restoring the
*behaviour* the deleted comment described, since the mechanism (`getIsaVersion`)
is gone. Tests: `tl.sum`/`tl.max` lit cases for `gfx1031`, then a GPU reduce test
vs torch over several shapes/dtypes. Risk: highest of M1. Rollback: revert.

**Step 4 — restore the codegen-shape parity items.**
`TargetInfo.cpp:762-778 getSharedLdStTiles`: add `case ISAFamily::RDNA2:` back to
the CDNA3/RDNA3/RDNA4m group. Files as listed. Why: byte-for-byte restoration of
parent behaviour, cheap. Tests: existing LDS-pattern lit tests with a `gfx1031`
`RUN` line. Risk: low. Rollback: revert.

**Step 5 — make RDNA2 testable in Python.**
`python/triton/_internal_testing.py`: add `is_hip_rdna2()` (`'gfx103' in
target.arch`) and include it in `is_hip_rdna()`. Why: without it the entire
Python suite skips RDNA2 and M1 has no exit criteria. Risk: enabling previously
-skipped tests may surface real failures — that is the intended outcome; triage
rather than re-disable. Rollback: revert.

**Step 6 — correctness matrix on hardware (idle window, small sizes).**
Order: masked elementwise (fp32/fp16/bf16/fp16 subnormals) → mask/other-value
semantics → reductions (step 3 follow-up) → atomics (generic, non-buffer) →
`local_alloc`/shared-memory-heavy patterns → `tl.load` cache modifiers. Each
compared against torch, small tensors, no benchmarks. Success: agreed results
with a recorded pass/fail table. Do **not** extend to "RDNA2 supported" beyond
the ops actually verified.

**Step 7 — only then, targeted instruction work.**
Decide whether `dot`-as-FMA is acceptable for the target workload (it is the only
option on RDNA2 hardware — P10) before any attempt at `v_dot2_*`/RDNA2-specific
matmul work.

Out of scope for M1: flash attention, large GEMM, MFMA-shaped kernels, any
performance claim, RDNA1, `-strict` handling, `smid()` restoration.

---

## 11. First concrete fix recommendation

**Target:** `third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp`
`computeISAFamily()` (lines 77-107), changed **together with**
`third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:69-76` and the
`isRDNA()` switch at `TargetFeatures.cpp:392`.

```cpp
// TargetFeatures.cpp, in computeISAFamily(), in the "RDNA ISA cases" block:
if (major == 10 && minor == 3)          // gfx103x = RDNA2
  return ISAFamily::RDNA2;

// BufferOpsEmitter.cpp:70 — must be in the same change:
if (llvm::is_contained({ISAFamily::RDNA2, ISAFamily::RDNA3, ISAFamily::RDNA4m,
                        ISAFamily::RDNA4, ISAFamily::GFX1250},
                       targetInfo.getISAFamily())) {

// TargetFeatures.cpp:392 isRDNA():
case ISAFamily::RDNA2:                  // restores the DPP/permlane shuffle path
```

**Why this first:** it is the only change that turns "cannot compile at all"
(P1) into anything at all, it is 4 lines, it exactly mirrors the parent commit
for the `gfx103x` family, and it is verifiable **without a GPU** via
`triton-opt` + FileCheck once Step 0's build exists. The descriptor-flag line is
included in the same change because separating them is how a 4-line "restore
support" patch becomes a silent data-corruption bug: buffer ops stay enabled on
`Unknown`, and `Unknown` produces `159744` where RDNA2 needs `822243328`.

**Deliberately not in this change:** `getSharedLdStTiles` (step 4, independent
perf shape), `smid()` (recommend leaving removed per AGENTS.md), RDNA1 (out of
scope), and `warpReduce` (step 3 — needs a test to decide, and it is the item
where blind restoration is most likely to be wrong).

**Verification for this change, in order:**
1. `lit` `gfx1030` and `gfx1031` `RUN` lines on
   `test/Conversion/amd/buffer_load_store.mlir` with the `RDNA` prefix →
   asserts `822243328` (guards P2 directly).
2. Re-added `amd-convert-buffer-ops-atomic-{fadd,rmw}.mlir` `gfx1030` `RUN`
   lines → asserts RDNA2 still gets GENERIC/NO-* atomics.
3. A new small elementwise `ttir → llvm` test at `gfx-arch=gfx1031` → confirms
   the three `Unknown` guards are cleared.
4. Then, and only then, the GPU reduce probe of step 3.

**Success criterion:** `triton-opt … gfx-arch=gfx1031` completes and emits the
RDNA-style descriptor; **failure mode to watch for:** a kernel that now compiles
and launches but whose `tl.sum` disagrees with torch (that is P3, expected and
planned for, not a regression in this change).

---

## 12. Performance roadmap, and why it matters for the actual workload

Sequenced strictly after correctness, and each step requires a measurement to
justify the next. Framed against the host's real workload (a local ComfyUI
diffusion pipeline on a 12 GiB RX 6700 XT — fp16 UNet, CPU VAE):

- **M2, baseline first.** A reproducible harness (fixed seeds, fixed shapes,
  idle-GPU windows, `TRITON_OVERRIDE_ARCH` recorded per run) covering the ops
  M1 validated. No optimisation before this exists.
- **M3, cross-lane cost (P4/P5).** `ds_bpermute` vs DPP `ROW_XMASK` and the LDS
  tile choice both land on reduction-heavy kernels, which in a UNet means group
  norm, softmax, and attention score reductions. This is where restoring
  `isRDNA()` and `getSharedLdStTiles` should pay, and it is measurable.
- **M4, `dot` is the ceiling (P10).** RDNA2 has no WMMA/MFMA, so Triton's `dot`
  is FMA-based. Options in cost order: tune tiles/`num_warps`/`BLOCK` for FMA;
  investigate `v_dot2_f32_f16`/`v_fma_f16`-shaped lowering (RDNA2 ISA support for
  `v_dot2_*` is **U** here and must be probed with `llvm-mc`/`llc` before any
  design); only then consider hand-tuned kernels. Any claim of "matmul
  acceleration on RDNA2" must be backed by measured flops, never by analogy to
  RDNA3.
- **M5, occupancy and WGP/CU mode (L4).** The 20-vs-40 "CU" question and
  `wgp_cu_mode` are 12 GiB-VRAM-relevant for diffusion: they change per-CTA LDS
  budget and register pressure, which is what actually limits large latent
  tiles. Needs measurement, not documentation.
- **M6, memory path.** Cache modifiers on RDNA2 buffer ops (**L6**, untested),
  `global`→`buffer` conversion thresholds, and the 3 MiB L2 / 192-bit bus
  behaviour for conv-style access patterns.

The honest end-state for this project is *"RDNA2/gfx103x executes a defined,
tested subset of Triton kernels correctly, with FMA-only dot"*, not *"Triton
supports RDNA2"*. The subset must be enumerated by test, because that claim is
what survives review.
