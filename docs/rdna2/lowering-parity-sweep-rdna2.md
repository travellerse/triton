# RDNA2 lowering-parity sweep over the AMD lit suite

Status: **CONFIRMED** (static, no GPU). Date: 2026-10-19.
Harness: [gfx1030_lit_sweep.py](file:///home/trave/code/triton/docs/rdna2/gfx1030_lit_sweep.py)
Output: [conformance/gfx1030_lit_sweep.txt](file:///home/trave/code/triton/docs/rdna2/conformance/gfx1030_lit_sweep.txt)

## Method

Every `*.mlir` under `test/Conversion/amd`, `test/TritonGPU/amd`,
`test/TritonAMDGPUTransforms` that runs `triton-opt ... | FileCheck` with a `gfx-arch=`
knob gets each of its RUN lines re-executed with **only** the arch value rewritten to
`gfx1030`. `triton-opt` needs no device, so this is a full-backend static parity probe.

```
RUN lines rewritten to gfx1030: 171 of 226 arch-aware runs; pass=80 fail=91
```

A failure means *that test's own expectations* do not hold on RDNA2. Because many of these
files are deliberately multi-arch (one line per ISA, each with its own check prefix),
rewriting the RDNA3 line to RDNA2 is *supposed* to fail, so raw failure counts say nothing;
what matters is whether each failure has a named cause. The script buckets by the origin
arch and by file name; the RDNA3/RDNA1-origin failures are the only interesting class
(13 items), and all of them resolve as follows.

## Every failure has a named cause

| count | file(s) | cause | verdict |
| --- | --- | --- | --- |
| 2 | `amd-buffer-cache-modifiers-rdna3.mlir` | no GFX10 cache-policy mapping; RDNA2 emits `aux = 0` | documented today in [cache-modifier-gap-rdna2.md](file:///home/trave/code/triton/docs/rdna2/cache-modifier-gap-rdna2.md), pinned by `amd-buffer-cache-modifiers-rdna2.mlir` |
| 2 | `accelerate-amd-matmul-wmma-gen1/gen2.mlir` | WMMA is RDNA3+; RDNA2 has no matrix instructions | ISA boundary, not a gap |
| 1 | `accelerate-amd-matmul-rdna-no-matrix.mlir` | the rewritten line was the file's own `RDNA3`-prefix WMMA expectation; the same file already runs `NO-MATRIX` on gfx1030/gfx1031 and passes | sweep artifact, not a gap |
| 1 | `minmax.mlir` | `supportMaximumMinimum()` puts RDNA2 in the same software-NaN group as RDNA3/RDNA4/gfx942; only gfx950/gfx1170/gfx1250 take `LLVM-MINMAX` | by design |
| 1 | `ocp_fp8_cvt_lowering.mlir` | `HWCVT` expects hardware fp8 pack/unpack (gfx1170+); RDNA2 emulates | ISA boundary, already covered by the fp8 conformance cases |
| 6 | `amd-convert-buffer-ops-atomic-{rmw,fadd,fminmax}.mlir` | `supportsBufferAtomicRMW()` excludes RDNA2 | the M1 decision, gate kept closed with its own pinned tests |

The remaining 78 failures come from `gfx942`/`gfx950`/`gfx1250`/`gfx1200` lines: MFMA,
async copy, `tdm`, cluster/`mbarrier`, `ds_transpose`, scaled cvt, warp-id. Those features
do not exist on RDNA2 at all, so rewriting them says nothing.

**Net: zero unexplained RDNA2 differences.** Together with the 80 lines that pass
unchanged (including the vec-128 LDS tiles restored for RDNA2), this is the closest thing
to a backend-wide statement we can make without hardware: RDNA2's generic lowering matches
the arches upstream tests, and where it does not, the difference is a documented, named ISA
feature or one of the two gated behaviours (buffer atomics, cache-policy aux).

## Limits of this claim

* Coverage is the lit suite, not the frontend: it says nothing about runtime behaviour,
  numerical results, or kernel quality. Those remain covered by the on-device conformance
  probe (43/43 at native gfx1031) and its own limits.
* 226 arch-aware lines exist in total; 55 have no `gfx-arch=` knob (transform passes keyed
  on encoding attributes instead) and are not exercised here.
* Passing with `gfx1030` substituted is *equality with that test's RDNA3/CDNA expectations*.
  Where RDNA2 legitimately differs (no WMMA, no HW fp8 cvt) the test cannot pass, so this
  sweep detects silence-but-wrong only for the parts that are arch-independent.
