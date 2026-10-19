"""Static probe: does tl.load(..., eviction_policy=...) reach the AMD backend at all?

Compares the *instruction stream* of two kernels that differ only in the eviction hint, against a
`.cg` control that is known to change the emitted cache-policy bits. Kernel-name metadata lines
(`.set .L<name>.*`, `.globl`, labels, `.loc`) are stripped, because they differ by construction and mean
nothing -- without that filter a byte diff reports ten "differences" for a hint that changes nothing.

No device, no launches: triton.compile against an explicit GPUTarget, then the emitted `.amdgcn`.
"""

import argparse
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def plain(IN, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(IN + offs))


@triton.jit
def evict_first(IN, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(IN + offs, eviction_policy="evict_first"))


@triton.jit
def evict_last(IN, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(IN + offs, eviction_policy="evict_last"))


@triton.jit
def cache_modifier_cg(IN, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(IN + offs, cache_modifier=".cg"))


ARMS = [(evict_first, "evict_first"), (evict_last, "evict_last"),
        (cache_modifier_cg, ".cg (control)")]


def instruction_stream(fn, arch, block):
    src = ASTSource(fn=fn, signature={"IN": "*fp32", "OUT": "*fp32", "BLOCK": "constexpr"},
                    constexprs={"BLOCK": block})
    kern = triton.compile(src, target=GPUTarget("hip", arch, 32),
                          options={"num_warps": 4, "num_stages": 1})
    text = re.sub(r"\b(plain|evict_first|evict_last|cache_modifier_cg)\b", "KERN",
                  kern.asm["amdgcn"])
    return [line.strip() for line in text.splitlines()
            if line.startswith("\t") and not line.strip().startswith((".", ";"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030,gfx1031,gfx1100,gfx942")
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()

    for arch in args.arch.split(","):
        base = instruction_stream(plain, arch, args.block)
        print(f"\n===== {arch} =====  (baseline: {len(base)} instruction lines, no hint)")
        for fn, name in ARMS:
            other = instruction_stream(fn, arch, args.block)
            delta = sum(1 for a, b in zip(base, other) if a != b) + abs(len(base) - len(other))
            verdict = "IDENTICAL -- hint dropped" if delta == 0 else f"{delta} lines differ"
            print(f"  {name:16} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
