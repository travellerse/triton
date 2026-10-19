"""Static (no-device) probe: does RDNA2 get the in-thread-transpose LDS win?

Upstream #10390 enabled the InThreadTranspose pass for gfx942 / gfx110x / gfx115x /
gfx120x. It reported that on gfx1151 the pass turned 512 scalar `ds_load_u16` into 144
vectorized `ds_load_b128` for the flash-attention V operand (+3.8% median). gfx103x is
absent from that gate, so this asks whether RDNA2 can use it at all.

The kernel is attention-shaped: operand A is produced in registers, operand B is a
[BN, D] fp16 tile consumed with K = BN. For each arch the script reports

  * whether `amdg.in_thread_transpose` appears in TTGIR (the pass firing),
  * the dot-operand parent encoding (MFMA / WMMA / Blocked=FMA),
  * the global-load and LDS instruction mix in the final ISA text.

Two harness facts matter for reading the output:

  * The pass is also gated by `knobs.amd.use_in_thread_transpose`, forced True here, so
    the arch list in `is_in_thread_transpose_enabled` is not the only thing under test.
  * ASTSource gets no divisibility hints unless this script supplies them. Without
    `tt.divisibility = 16` every arch lowers to scalar `global_load_ushort`, which is a
    harness artifact, not an arch difference -- `--no-hints` reproduces that trap.

Usage: python3 rdna2_inthread_transpose_probe.py [--no-hints] [--arch gfx1030 ...]
"""

import argparse
import collections
import re
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.knobs as knobs

MEM_RE = re.compile(r"\b(?:ds|global|buffer|scratch)_(?:load|store|read|write)[a-z0-9_]*\b")


@triton.jit
def attn_v_kernel(Q, V, O, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    # operand A: [BM, BN], produced by a global load here
    p = tl.load(Q + offs_m[:, None] * BN + offs_n[None, :])
    # operand B: [BN, D], contiguous along D but consumed with K = BN
    v = tl.load(V + offs_n[:, None] * D + offs_d[None, :])
    acc = tl.dot(p, v)
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], acc.to(tl.float16))


def compile_one(arch, hints):
    attrs = {}
    if hints:
        attrs = {(i,): [["tt.divisibility", 16]] for i in range(3)}
    src = ASTSource(
        fn=attn_v_kernel,
        signature={"Q": "*fp16", "V": "*fp16", "O": "*fp16"},
        constexprs={"BM": 64, "BN": 64, "D": 64},
        attrs=attrs,
    )
    return triton.compile(src, target=GPUTarget("hip", arch, 32),
                          options={"num_warps": 4, "num_stages": 1})


def summarize(arch, k):
    ttgir = k.asm.get("ttgir", "")
    amdgcn = k.asm.get("amdgcn", "")
    mix = collections.Counter(MEM_RE.findall(amdgcn))
    parents = set(re.findall(r"parent = #(blocked\d*|ttg\.amd_\w+|amd_\w+|\w+)>", ttgir))
    dotops = set(re.findall(r"#ttg\.dot_op<\{opIdx = (\d), parent = (#\w+)", ttgir))
    fired = "in_thread_transpose" in ttgir
    print(f"  {arch:9s} pass_fired={'YES' if fired else ' no'} "
          f"dot_op_parents={sorted({p for _, p in dotops}) or 'none'}")
    print(f"          loads: {dict((t, n) for t, n in mix.items() if 'load' in t)}")
    print(f"          stores: {dict((t, n) for t, n in mix.items() if 'store' in t or 'write' in t)}")
    return fired, mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hints", action="store_true",
                    help="omit tt.divisibility, reproducing the all-scalar-load artifact")
    ap.add_argument("--arch", nargs="+", default=["gfx942", "gfx1100", "gfx1030", "gfx1031"])
    args = ap.parse_args()

    knobs.amd.use_in_thread_transpose = True
    hints = not args.no_hints
    print(f"== use_in_thread_transpose=True, divisibility_hints={hints} ==")
    for arch in args.arch:
        try:
            k = compile_one(arch, hints)
        except Exception as e:  # noqa: BLE001 - the probe reports, it never hides
            print(f"  {arch:9s} COMPILE FAILED {type(e).__name__}: {str(e).splitlines()[0][:110]}")
            continue
        summarize(arch, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
