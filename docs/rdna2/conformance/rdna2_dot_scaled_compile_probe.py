"""Static probe: does tl.dot_scaled (MX scaled dot) reach an RDNA2 target at all?

The AMD scaled-upcast machinery is gated to newer families -- OptimizeDotOperands' LDS allocation for the
scale operand returns "NYI: Only supported on CDNA4", AccelerateAMDMatmul registers ScaledBlockedTo*
patterns only for CDNA4/CDNA5, and TargetFeatures carries supportsHwScaledUpcast()/supportsHwScaledDowncast
-- so the RDNA2 compatibility question is not whether it is fast but whether a kernel that uses the
feature *compiles*, fails with an actionable message, or quietly becomes something else. Neither the
op-class conformance probe nor the differential compile sweep covers dot_scaled, which is why this needed
its own instrument.

Shapes follow python/test/unit/language/test_compile_only.py::test_simple_dot_mxfp (b_scale indexed per
(N, K//32), every operand and scale passed as *u8), because guessing the packing here produces frontend
errors -- and a frontend error is target-independent and must never be reported as backend
incompatibility.

Compile-only: triton.compile against an explicit GPUTarget. No device, no launches.
"""

import argparse
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def make_kernel(pack_a, pack_b, fmt_a, fmt_b):
    """build a @triton.jit kernel for the given formats; pack_* = elements per byte."""

    @triton.jit
    def kern(a_base, b_base, a_scale, b_scale, out,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             PACK_A: tl.constexpr, PACK_B: tl.constexpr, FMT_A: tl.constexpr,
             FMT_B: tl.constexpr):
        ka: tl.constexpr = BLOCK_K // PACK_A
        kb: tl.constexpr = BLOCK_K // PACK_B
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * ka + tl.arange(0, ka)[None, :]
        # rhs is (K, N): dot_scaled reads K from rhs.shape[-2], so the orientation is part of the
        # contract -- with b laid out (N, K) the packed dims coincide by accident for e2m1 and the
        # verification fails for e4m3, which is a frontend error, not a backend one.
        b_ptr = b_base + tl.arange(0, kb)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        scale_k: tl.constexpr = BLOCK_K // 32
        sa_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * scale_k + tl.arange(0, scale_k)[None, :]
        sb_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * scale_k + tl.arange(0, scale_k)[None, :]
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot_scaled(a, tl.load(sa_ptr), FMT_A, b, tl.load(sb_ptr), FMT_B)
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, acc)

    return kern


CASES = [
    ("mxfp8", 1, 1, "e4m3", "e4m3"),
    ("mxfp4", 2, 2, "e2m1", "e2m1"),
    ("mixed", 1, 2, "e4m3", "e2m1"),
]

INTERESTING = re.compile(r"v_mfma\w*|v_wmma\w*|v_dot4c_i32_i8|v_dot2c_f32_f16|v_dot2\w*|"
                         r"ds_read_b\d+|global_load_dword|local_read")


def run(arch, block_m, block_n, block_k, warps):
    print(f"\n===== {arch} (BM={block_m} BN={block_n} BK={block_k} warps={warps}) =====")
    for name, pack_a, pack_b, fmt_a, fmt_b in CASES:
        fn = make_kernel(pack_a, pack_b, fmt_a, fmt_b)
        sig = {"a_base": "*u8", "b_base": "*u8", "a_scale": "*u8", "b_scale": "*u8",
               "out": "*fp32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
               "BLOCK_K": "constexpr", "PACK_A": "constexpr", "PACK_B": "constexpr",
               "FMT_A": "constexpr", "FMT_B": "constexpr"}
        consts = {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k,
                  "PACK_A": pack_a, "PACK_B": pack_b, "FMT_A": fmt_a, "FMT_B": fmt_b}
        try:
            kern = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts),
                                  target=GPUTarget("hip", arch, 32),
                                  options={"num_warps": warps, "num_stages": 1})
        except Exception as e:  # noqa: BLE001 - the message IS the finding
            print(f"  {name:8} FAILS: {' '.join(str(e).strip().split())[-240:]}")
            continue
        asm = kern.asm["amdgcn"]
        hits = {}
        for tok in INTERESTING.findall(asm):
            hits[tok] = hits.get(tok, 0) + 1
        spills = re.findall(r"vgpr_spill_count[=:](\d+)", asm)
        print(f"  {name:8} COMPILES instrs={len(asm.splitlines()):6} "
              f"dot/lds={dict(sorted(hits.items(), key=lambda kv: -kv[1]))} spills={spills[:1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030,gfx1031,gfx1100,gfx942,gfx1200")
    ap.add_argument("--bm", type=int, default=32)
    ap.add_argument("--bn", type=int, default=32)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--warps", type=int, default=4)
    args = ap.parse_args()
    for arch in args.arch.split(","):
        run(arch, args.bm, args.bn, args.bk, args.warps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
