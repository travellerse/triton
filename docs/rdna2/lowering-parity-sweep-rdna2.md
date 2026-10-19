# RDNA2 lowering-parity sweep over the AMD lit suite

Status: **CONFIRMED** (static, no GPU). Date: 2026-10-19.
Harness: [gfx1030_lit_sweep.py](file:///home/trave/code/triton/docs/rdna2/gfx1030_lit_sweep.py)
Output: [conformance/gfx1030_lit_sweep.txt](file:///home/trave/code/triton/docs/rdna2/conformance/gfx1030_lit_sweep.txt)

## Method

Every `*.mlir` under `test/Conversion`, `test/TritonGPU`, `test/TritonAMDGPUTransforms`,
`test/Triton`, `test/Gluon`, `test/Proton` that runs `triton-opt ... | FileCheck` with a
`gfx-arch=` knob gets each of its RUN lines re-executed with **only** the arch value
rewritten to `gfx1030`. `triton-opt` needs no device, so this is a full-backend static
parity probe.

```
RUN lines rewritten to gfx1030: 177 of 406 arch-aware runs; pass=82 fail=95
```

Re-running the identical sweep with `--target gfx1031` (the part in this machine; the host's
`HSA_OVERRIDE_GFX_VERSION` otherwise hides it) gives **the same 177 / 82 / 95, the same bucket membership, and an empty unexplained list on
both**, so the parity statement covers both steppings, not just the one torch reports. Artifact:
[conformance/gfx1031_lit_sweep.txt](file:///home/trave/code/triton/docs/rdna2/conformance/gfx1031_lit_sweep.txt).

A failure means *that test's own expectations* do not hold on RDNA2. Because many of these
files are deliberately multi-arch (one line per ISA, each with its own check prefix),
rewriting the RDNA3 line to RDNA2 is *supposed* to fail, so raw failure counts say nothing;
what matters is whether each failure has a named cause. The script now buckets the causes
itself: rewritten failures are split into (a) files that already
carry a verbatim `gfx1030`/`gfx1031` RUN line, i.e. arch-comparison tests where a rewritten line
*must* fail -- 11 items, (b) files whose names announce an ISA feature RDNA2 lacks -- 2 items,
and (c) everything else, which is the signal list and is **empty on both parts**. The named causes
below are what those buckets encode, kept here as the reasoning rather than as a hand-maintained
count. The other 82 failures come from `gfx942`/`gfx950`/`gfx1200`/`gfx1250` lines.

## Every failure has a named cause

| count | file(s) | cause | verdict |
| --- | --- | --- | --- |
| 2 | `amd-buffer-cache-modifiers-rdna3.mlir` | RDNA2 has its own GFX10-native table (load `.cg`→5, `.cs`→2, store `.cs`→2, `.wt`→4), which is deliberately neither RDNA3's (1/7) nor CDNA's (3/17) | by design since the mapping landed — see [performance/rdna2-cache-modifiers-enabled.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-cache-modifiers-enabled.md); `amd-buffer-cache-modifiers-rdna2.mlir` pins the RDNA2 side |
| 2 | `accelerate-amd-matmul-wmma-gen1/gen2.mlir` | WMMA is RDNA3+; RDNA2 has no matrix instructions | ISA boundary, not a gap |
| 1 | `accelerate-amd-matmul-rdna-no-matrix.mlir` | the rewritten line was the file's own `RDNA3`-prefix WMMA expectation; the same file already runs `NO-MATRIX` on gfx1030/gfx1031 and passes | sweep artifact, not a gap |
| 1 | `minmax.mlir` | `supportMaximumMinimum()` lists CDNA4/RDNA4m/CDNA5, so RDNA2 is in the software-NaN group with RDNA3/gfx942 | **now asserted**: the file carries gfx1030/gfx1031 `SW-NAN` run lines |
| 1 | `ocp_fp8_cvt_lowering.mlir` | `HWCVT` expects hardware fp8 pack/unpack (gfx1170+); RDNA2 emulates | **now asserted**: gfx1030/gfx1031 `NOHW` run lines added |
| 6 | `amd-convert-buffer-ops-atomic-{rmw,fadd,fminmax}.mlir` | `supportsBufferAtomicRMW()` excludes RDNA2 | the M1 decision, gate kept closed with its own pinned tests |

The remaining 82 failures come from `gfx942`/`gfx950`/`gfx1250`/`gfx1200` lines: MFMA,
async copy, `tdm`, cluster/`mbarrier`, `ds_transpose`, scaled cvt, warp-id. Those features
do not exist on RDNA2 at all, so rewriting them says nothing.

**Net: zero unexplained RDNA2 differences.** Together with the 82 lines that pass
unchanged (including the vec-128 LDS tiles restored for RDNA2), this is the closest thing
to a backend-wide statement we can make without hardware: RDNA2's generic lowering matches
the arches upstream tests, and where it does not, the difference is a documented, named ISA
feature or one of the two gated behaviours (buffer atomics, cache-policy aux).

## Limits of this claim

* Coverage is the lit suite, not the frontend: it says nothing about runtime behaviour,
  numerical results, or kernel quality. Those remain covered by the on-device conformance
  probe (43/43 at native gfx1031) and its own limits.
* 406 arch-aware RUN lines exist across the tree; 227 have no `gfx-arch=` knob at all
  (transform passes keyed on encoding attributes instead) and are not exercised here.
* Passing with `gfx1030` substituted is *equality with that test's RDNA3/CDNA expectations*.
  Where RDNA2 legitimately differs (no WMMA, no HW fp8 cvt) the test cannot pass, so this
  sweep detects silence-but-wrong only for the parts that are arch-independent.
