"""Compile-only probe: which packed dot instructions Triton actually emits per AMD arch.

No device and no submission -- triton.compile against an explicit GPUTarget, then the histogram of
dot-family mnemonics in the emitted .amdgcn. The point is to answer "does RDNA2's V_DOT4/V_DOT8
capability go unused?" with emitted instructions rather than by reading gates, and to keep the
instruction count checkable: for BMxBNxBK with W warps, an ideal packed-int8 lowering issues
BM*BN*BK / (32*W) MACs per lane / (elements per instruction).
"""

import argparse
import collections
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

DOT_RE = re.compile(r"dot|wmma|madmac|madd_|machf|mohaf|mac_f32|fma_f32|permlane")


@triton.jit
def int_dot(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)[:, None]
    k = tl.arange(0, BK)[None, :]
    kk = tl.arange(0, BK)[:, None]
    n = tl.arange(0, BN)[None, :]
    a = tl.load(A + m * BK + k)
    b = tl.load(B + kk * BN + n)
    acc = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C + m * BN + n, acc)


@triton.jit
def f16_dot(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)[:, None]
    k = tl.arange(0, BK)[None, :]
    kk = tl.arange(0, BK)[:, None]
    n = tl.arange(0, BN)[None, :]
    a = tl.load(A + m * BK + k)
    b = tl.load(B + kk * BN + n)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + m * BN + n, acc)


def histogram(asm):
    ops = collections.Counter(re.findall(r"^\t([a-z_0-9]+)", asm, re.M))
    total = sum(ops.values())
    dot = {o: c for o, c in ops.items() if DOT_RE.search(o)}
    return total, dot


def report(name, fn, sig, arch, warps, bm, bn, bk, mac_per_instr):
    src = ASTSource(fn=fn, signature=sig,
                    constexprs={"BM": bm, "BN": bn, "BK": bk})
    try:
        kern = triton.compile(src, target=GPUTarget("hip", arch, 32),
                              options={"num_warps": warps, "num_stages": 1})
    except Exception as e:  # noqa: BLE001 - the message IS the finding
        print(f"  {name:12} {arch:8} FAILS: {' '.join(str(e).split())[-200:]}")
        return
    total, dot = histogram(kern.asm["amdgcn"])
    macs = bm * bn * bk
    per_lane = macs // (32 * warps)
    ideal = -(-per_lane // mac_per_instr)
    issued = sum(dot.values())
    print(f"  {name:12} {arch:8} instr={total:6} dot-family={dot or '{}'} "
          f"| MACs/lane={per_lane} one-instr-per-{mac_per_instr}-MAC ideal={ideal} issued={issued}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030,gfx1031,gfx942,gfx1100,gfx1200")
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=16)
    ap.add_argument("--bk", type=int, default=16)
    args = ap.parse_args()

    for arch in args.arch.split(","):
        report("i8->i32", int_dot, {"A": "*i8", "B": "*i8", "C": "*i32"},
               arch, 1, args.bm, args.bn, args.bk, 4)
        report("f16->f32", f16_dot, {"A": "*fp16", "B": "*fp16", "C": "*fp32"},
               arch, 1, args.bm, args.bn, args.bk, 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
