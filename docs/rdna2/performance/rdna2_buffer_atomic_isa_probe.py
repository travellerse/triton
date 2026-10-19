"""Static probe: does the whole Triton pipeline reach RDNA2's native buffer atomics?

For each atomic op and element type, compile a kernel against an explicit GPUTarget (no device, no
submission), then report which buffer_atomic instructions the emitted .amdgcn contains. This is the
ISA-level end of the claim: the lit tests prove ConvertToBufferOps accepts the op, llc proves the
intrinsic is selectable, and this proves the real pipeline -- pass, ROCDL, MI lowering, assembler --
lands on the one-instruction form instead of the global_atomic_cmpswap retry loop Triton otherwise pays
for and/or/xor and integer min/max.

Kernels are written out one per op rather than generated, because @triton.jit needs real source lines.
Each is deliberately unmasked over a 1024-element block with 4 warps: that is the only shape the RDNA2
subset allows, since a masked lane's bounds suppression rests on the unsigned-offset question.

What this does NOT show is what the instruction does for a lane the descriptor bounds-suppresses; that
needs a device.
"""

import argparse
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

BLOCK = 1024
WARPS = 4


@triton.jit
def k_and_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_and(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_or_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_or(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_xor_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_xor(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_max_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_max(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_min_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_min(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_max_u32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_max(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_min_u32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_min(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_and_i64(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_and(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_max_i64(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_max(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_add_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_add(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_xchg_i32(PTR, VAL, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    res = tl.atomic_xchg(PTR + offs, tl.load(VAL + offs))
    tl.store(OUT + offs, res)


@triton.jit
def k_masked_and_i32(PTR, VAL, OUT, MASK, BLOCK: tl.constexpr):
    """Control: a user mask must keep this off the buffer path on RDNA2."""
    offs = tl.arange(0, BLOCK)
    m = tl.load(MASK + offs) != 0
    res = tl.atomic_and(PTR + offs, tl.load(VAL + offs), mask=m)
    tl.store(OUT + offs, res, mask=m)


@triton.jit
def loop_and_i32(DATA, BINS, N, BLOCK: tl.constexpr):
    """The hot form: an atomic inside a pipelined loop."""
    offs = tl.arange(0, BLOCK)
    for i in range(0, N, BLOCK):
        tl.atomic_and(BINS + offs, tl.load(DATA + i + offs))


@triton.jit
def loop_max_i32(DATA, BINS, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in range(0, N, BLOCK):
        tl.atomic_max(BINS + offs, tl.load(DATA + i + offs))


CASES = [
    ("and_i32", k_and_i32, "*i32", False),
    ("or_i32", k_or_i32, "*i32", False),
    ("xor_i32", k_xor_i32, "*i32", False),
    ("max_i32", k_max_i32, "*i32", False),
    ("min_i32", k_min_i32, "*i32", False),
    ("max_u32", k_max_u32, "*u32", False),
    ("min_u32", k_min_u32, "*u32", False),
    ("and_i64", k_and_i64, "*i64", False),
    ("max_i64", k_max_i64, "*i64", False),
    ("add_i32", k_add_i32, "*i32", False),
    ("xchg_i32", k_xchg_i32, "*i32", False),
    ("masked_and_i32", k_masked_and_i32, "*i32", True),
]


def run(arch):
    print(f"\n===== {arch} =====")
    for name, fn, pt, masked in CASES:
        sig = {"PTR": pt, "VAL": pt, "OUT": pt, "BLOCK": "constexpr"}
        if masked:
            sig["MASK"] = "*i1"
        src = ASTSource(fn=fn, signature=sig, constexprs={"BLOCK": BLOCK})
        try:
            kern = triton.compile(src, target=GPUTarget("hip", arch, 32),
                                  options={"num_warps": WARPS, "num_stages": 1})
        except Exception as e:  # noqa: BLE001 - the message IS the finding
            print(f"  {name:14} COMPILE FAILS: {' '.join(str(e).split())[-140:]}")
            continue
        asm = kern.asm["amdgcn"]
        buf = sorted(set(re.findall(r"\bbuffer_atomic_[a-z0-9_]+", asm)))
        cas = len(re.findall(r"global_atomic_cmpswap", asm))
        glob = sorted(set(re.findall(r"\bglobal_atomic_[a-z0-9_]+", asm)))
        form = " ".join(buf) if buf else (f"{' '.join(glob) or 'no atomic'} ({cas} cmpswap)")
        print(f"  {name:14} {form}")

    # The hot form, and the shape that is supposed to be refused. BLOCK=64 over 4 warps leaves half the
    # lanes redundant (64 elements across 128 lanes), which is a free variable in the layout -- exactly
    # the case the subset gate excludes because a redundant lane would be issued at an out-of-range
    # offset. BLOCK>=128 with one element per lane is provably active, so it should take the buffer path.
    print("  -- in a pipelined loop (num_stages=2) --")
    for fn, name in [(loop_and_i32, "loop and"), (loop_max_i32, "loop max")]:
        for block in (64, 128, 256, 1024):
            src = ASTSource(fn=fn, signature={"DATA": "*i32", "BINS": "*i32",
                                              "N": "i32", "BLOCK": "constexpr"},
                             constexprs={"BLOCK": block})
            try:
                kern = triton.compile(src, target=GPUTarget("hip", arch, 32),
                                      options={"num_warps": WARPS, "num_stages": 2})
            except Exception as e:  # noqa: BLE001 - the message IS the finding
                print(f"  {name} BLOCK={block:5} COMPILE FAILS: "
                      f"{' '.join(str(e).split())[-130:]}")
                continue
            asm = kern.asm["amdgcn"]
            buf = len(re.findall(r"\bbuffer_atomic_", asm))
            cas = len(re.findall(r"global_atomic_cmpswap", asm))
            verdict = "buffer" if buf and not cas else ("generic" if cas else "no atomic")
            print(f"  {name} BLOCK={block:5} buffer_atomic={buf} cmpswap={cas} -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030,gfx1031,gfx1100,gfx942")
    args = ap.parse_args()
    for arch in args.arch.split(","):
        run(arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
