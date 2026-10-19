"""Static (no GPU) probe: which layers of Gluon can an RDNA2 target actually compile?

Purpose: the "frontier kernels for RDNA2" question turns on whether Gluon -- the layer people use
to hand-write fast attention on NVIDIA -- can express an RDNA2 kernel at all. RDNA2 has neither
MFMA nor WMMA, so the interesting case is `gl.dot_fma`, whose operands are ordinary blocked and
dot-operand layouts rather than a matrix-unit layout. Each case below is one layer of the stack,
compiled with an explicit GPUTarget so nothing reaches a device; gfx1100 is the control.

Usage: PYTHONPATH=./python python docs/rdna2/conformance/gluon_rdna2_compile_probe.py
"""

import collections
import re
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon import GluonASTSource
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCKED_1D = gl.BlockedLayout(size_per_thread=[4], threads_per_warp=[32],
                              warps_per_cta=[1], order=[0])
OTHER_1D = gl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32],
                            warps_per_cta=[1], order=[0])
BLOCKED_2D = gl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[4, 8],
                              warps_per_cta=[1, 1], order=[1, 0])
SHARED = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
DOT_A = gl.DotOperandLayout(operand_index=0, parent=BLOCKED_2D, k_width=0)
DOT_B = gl.DotOperandLayout(operand_index=1, parent=BLOCKED_2D, k_width=0)


@gluon.jit
def copy_only(IN, OUT, BLOCK: gl.constexpr, LAYOUT: gl.constexpr):
    offs = gl.arange(0, BLOCK, layout=LAYOUT)
    gl.store(OUT + offs, gl.load(IN + offs))


@gluon.jit
def convert_only(IN, OUT, BLOCK: gl.constexpr, LAYOUT: gl.constexpr,
                 OTHER: gl.constexpr):
    src_offs = gl.arange(0, BLOCK, layout=LAYOUT)
    dst_offs = gl.arange(0, BLOCK, layout=OTHER)
    x = gl.load(IN + src_offs)
    gl.store(OUT + dst_offs, gl.convert_layout(x, layout=OTHER))


@gluon.jit
def lds_roundtrip(IN, OUT, M: gl.constexpr, N: gl.constexpr, LAYOUT: gl.constexpr,
                  SLAYOUT: gl.constexpr):
    smem = gl.allocate_shared_memory(gl.float32, [M, N], SLAYOUT)
    rows = gl.arange(0, M, layout=gl.SliceLayout(1, LAYOUT))
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, LAYOUT))
    smem.store(gl.load(IN + rows[:, None] * N + cols[None, :]))
    gl.store(OUT + rows[:, None] * N + cols[None, :], smem.load(LAYOUT))


@gluon.jit
def fma_dot(A, B, OUT, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
            LAYOUT: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr):
    arows = gl.arange(0, M, layout=gl.SliceLayout(1, LA))
    acols = gl.arange(0, K, layout=gl.SliceLayout(0, LA))
    brows = gl.arange(0, K, layout=gl.SliceLayout(1, LB))
    bcols = gl.arange(0, N, layout=gl.SliceLayout(0, LB))
    a = gl.load(A + arows[:, None] * K + acols[None, :])
    b = gl.load(B + brows[:, None] * N + bcols[None, :])
    acc = gl.zeros([M, N], gl.float32, layout=LAYOUT)
    out = gl.dot_fma(gl.convert_layout(a, LA), gl.convert_layout(b, LB), acc)
    orows = gl.arange(0, M, layout=gl.SliceLayout(1, LAYOUT))
    ocols = gl.arange(0, N, layout=gl.SliceLayout(0, LAYOUT))
    gl.store(OUT + orows[:, None] * N + ocols[None, :], out)


CASES_PLAIN = [
    ("load/store", copy_only, {"IN": "*fp32", "OUT": "*fp32"},
     {"BLOCK": 1024, "LAYOUT": BLOCKED_1D}),
    ("convert_layout", convert_only, {"IN": "*fp32", "OUT": "*fp32"},
     {"BLOCK": 1024, "LAYOUT": BLOCKED_1D, "OTHER": OTHER_1D}),
    ("shared memory", lds_roundtrip, {"IN": "*fp32", "OUT": "*fp32"},
     {"M": 32, "N": 32, "LAYOUT": BLOCKED_2D, "SLAYOUT": SHARED}),
    ("dot_fma", fma_dot, {"A": "*fp16", "B": "*fp16", "OUT": "*fp32"},
     {"M": 16, "N": 16, "K": 16, "LAYOUT": BLOCKED_2D, "LA": DOT_A, "LB": DOT_B}, "dot"),
]


def try_compile(arch, name, fn, signature, constexprs, expect_arith=None):
    """expect_arith: substring that MUST appear among the emitted v_* mnemonics. A kernel that
    compiles with the operation optimised away is not evidence of support -- the first version of
    this probe reported dot_fma as compiling on every arch while emitting no dot at all."""
    try:
        src = GluonASTSource(fn=fn, signature=signature, constexprs=constexprs)
        k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                           options={"num_warps": 1, "num_stages": 1})
        asm = k.asm.get("amdgcn", "")
        spills = "?"
        if "vgpr_spill_count:" in asm:
            spills = asm.split("vgpr_spill_count:")[1].split("\n")[0].strip()
        ops = collections.Counter(re.findall(r"^\s+([vgs]_\w+)", asm, re.M))
        arith = {o: c for o, c in ops.items() if re.match(r"v_(dot|fma|mac|mad|pk_)", o)}
        verdict = "COMPILES"
        if expect_arith and not any(expect_arith in o for o in arith):
            verdict = f"COMPILES-BUT-MISSING-{expect_arith}"
        elif expect_arith:
            verdict = "COMPILES+HAS-IT"
        print(f"  {name:15} {arch:8} {verdict:24} spills={spills:3} arith={arith}")
        return verdict == "COMPILES+HAS-IT" if expect_arith else True
    except Exception as e:  # noqa: BLE001 - the message IS the finding
        text = " ".join(l.strip() for l in str(e).strip().splitlines())
        print(f"  {name:15} {arch:8} FAILS: ...{text[-200:]}")
        return False


def main():
    for arch in ["gfx1100", "gfx1030"]:
        label = "control (RDNA3)" if arch == "gfx1100" else "RDNA2 under test"
        print(f"\n===== {arch} -- {label} =====")
        for case in CASES_PLAIN:
            name, fn, sig, cexpr = case[:4]
            try_compile(arch, name, fn, sig, cexpr,
                        case[4] if len(case) > 4 else None)
        print("  (no matrix-unit case: gl has no plain gl.dot; the matrix ops live in the\n"
              "   per-family namespaces gl.amd.{cdna3,cdna4,cdna5,rdna3,rdna4}, and there is\n"
              "   no rdna2 namespace -- AMDWMMALayout versions are 1=RDNA3, 2=RDNA4, 3=CDNA5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
