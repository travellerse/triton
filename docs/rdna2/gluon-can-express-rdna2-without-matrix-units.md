# Gluon can express RDNA2 kernels — just not matrix-unit or warp-specialized ones

Status: **CONFIRMED** statically (compile + emitted-instruction check, zero GPU launches).
Date: 2026-10-19. Device context: RX 6700 XT, host reports gfx1030 under
`HSA_OVERRIDE_GFX_VERSION=10.3.0`; the part is gfx1031, and both were compiled.
Probe: [conformance/gluon_rdna2_compile_probe.py](file:///home/trave/code/triton/docs/rdna2/conformance/gluon_rdna2_compile_probe.py)

This answers the standing open question "can Gluon express RDNA2 kernels at all", which decides
whether "frontier accelerated algorithms for RDNA2" (fast attention being the obvious one) has a
path at all: Gluon is the layer people use to hand-write peak-performance attention on NVIDIA, so
if RDNA2 could not be expressed in it, that goal would be structurally blocked.

## Result

`GluonASTSource` + `triton.compile(target=GPUTarget("hip", arch, 32))`, `num_warps=1`, and each
layer checked in the emitted ISA rather than only for a successful return:

```
===== gfx1100 -- control (RDNA3) =====
  load/store      COMPILES          spills=0  arith={}
  convert_layout  COMPILES          spills=0  arith={}
  shared memory   COMPILES          spills=0  arith={}
  dot_fma         COMPILES+HAS-IT   spills=0  arith={'v_dot2acc_f32_f16': 78}
===== gfx1030 -- RDNA2 under test =====
  load/store      COMPILES          spills=0  arith={}
  convert_layout  COMPILES          spills=0  arith={}
  shared memory   COMPILES          spills=0  arith={}
  dot_fma         COMPILES+HAS-IT   spills=0  arith={'v_dot2c_f32_f16': 128}
```

So on RDNA2, Gluon handles explicit layouts, layout conversion, shared memory (an LDS store/load
round-trip through `SwizzledSharedLayout`), and a dot — with **zero register spills** at 16×16×16.
The dot does not need a matrix unit: `gl.dot_fma` produces `tt.dot` whose operands are
`#ttg.dot_op<{opIdx=…, parent=#blocked}>` with a plain blocked accumulator, and Triton's RDNA2
lowering turns that into the family's packed half-precision dot instruction.

Three generations, three spellings of the same instruction class, all reached from identical Gluon
source: `gfx1030 → v_dot2c_f32_f16`, `gfx1100 → v_dot2acc_f32_f16`, `gfx1200 → v_dot2_f32_f16`.
The instruction *counts* differ (128 / 78 / 56); that is expected to come from per-generation
operand widths and fusion, but it is **not established here** — the probe does not claim it.

## What RDNA2 genuinely cannot have in Gluon

* **A matrix-unit layout.** `AMDWMMALayout` documents and asserts versions 1 = RDNA3, 2 = RDNA4,
  3 = CDNA5 ([_layouts.py:118-127](file:///home/trave/code/triton/python/triton/experimental/gluon/language/amd/_layouts.py#L118));
  `AMDMFMALayout` is CDNA-only. There is no RDNA2 version to name, and that is an ISA fact: RDNA2
  has neither WMMA nor MFMA (the compiler registers no matrix pattern for `ISAFamily::RDNA2` at all,
  which is what [test/TritonGPU/amd/accelerate-amd-matmul-rdna-no-matrix.mlir](file:///home/trave/code/triton/test/TritonGPU/amd/accelerate-amd-matmul-rdna-no-matrix.mlir)
  pins for gfx1030/gfx1031/gfx1010).
* **Per-family Gluon namespace.** `gl.amd` exposes `cdna3`, `cdna4`, `cdna5`, `rdna3`, `rdna4`,
  `gfx1250` — verified with `dir()`, no `rdna2`. Matrix entry points live in those namespaces
  (e.g. `gl.amd.rdna3.wmma`); there is no plain `gl.dot`.
* **Warp specialization.** [ConvertWarpSpecializeToLLVM.cpp:217](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp#L217)
  emits `"Warp specialization is only supported on CDNA5, got <target>"`. That rules out the
  producer/consumer structure the CDNA5 and NVIDIA attention kernels rely on.
* **Test coverage.** `is_hip_rdna()` is `rdna3 or rdna4m or rdna4`
  ([_internal_testing.py:173-174](file:///home/trave/code/triton/python/triton/_internal_testing.py#L173)), so the Gluon AMD tests skip on
  RDNA2, and `python/test/gluon/test_frontend.py` has no gfx103x target. This aggregate is left as
  it is on purpose: it was analysed in the M1 report, and widening it would start *running* cases
  whose prerequisites were never checked while making currently-passing ones skip.

## Consequence for the RDNA2 kernel plan

An RDNA2 flash-attention written in Gluon is **not blocked by the compiler's expressiveness**. The
available shape is: explicit blocked layouts + LDS staging + `dot_fma` on fp16 operands, with no
warp specialization and no matrix instruction. That matches what the measured envelope already says
about this card — fast attention on RDNA2 is register-bound
([rdna2-fast-attn-is-register-bound.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-fast-attn-is-register-bound.md)),
and a fused attention implementation loses to the current path at production geometry
([rdna2-fused-fa-loses-on-rdna2.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-fused-fa-loses-on-rdna2.md)).
Gluon buys *layout control* at the same register cost, not more compute.

## Limits of this claim

Static compilation only: nothing was launched, so correctness and speed of these kernels on RDNA2
are unverified. The 16×16×16 tile with zero spills is the easiest case; the register-bound result
says larger tiles are where this card stops. And `dot_fma` emitting `v_dot2c_f32_f16` says the
instruction was chosen, not that it is fast on this part.

## A probe lesson worth keeping

The first version of this probe reported "dot_fma COMPILES" on every architecture while emitting
**no dot instruction at all** — the operation had been optimised away, and the verdict was
decoration. Every case now asserts the expected mnemonic is present in the emitted ISA
(`COMPILES+HAS-IT` vs `COMPILES-BUT-MISSING-<op>`). A compile return code is not evidence a feature
works; a generated instruction is.
