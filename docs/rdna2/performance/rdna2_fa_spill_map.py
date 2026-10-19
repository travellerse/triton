"""Does a *real* online-softmax flash-attention inner loop have any spill-free shape on RDNA2?

The single-dot sweep (rdna2_dot_spill_map.py) showed RDNA2's FMA dot spills at almost every
useful tile. That table is optimistic, though: flash attention keeps the running max, the
running denominator, the raw scores tile and the fp16 cast of the scores alive at the same
time. This script compiles an actual FA forward step and reports the same numbers, per
(BLOCK_M, BLOCK_N, head_dim, warps), to produce the design envelope an RDNA2 attention
kernel has to live inside.

Static only -- triton.compile with an explicit GPUTarget never touches the GPU.

Usage: python3 rdna2_fa_spill_map.py [--archs gfx1030 gfx1100]
"""

import argparse
import re
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

SPILL = re.compile(r"\.vgpr_spill_count:\s*(\d+)")
PRIV = re.compile(r"\.private_segment_fixed_size:\s*(\d+)")
NGPR = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")
SHMEM = re.compile(r"\.amdhsa_group_segment_fixed_size\s+(\d+)")


@triton.jit
def fa_fwd(Q, K, V, O, SQ, SK, sm,
           BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr,
           EXP2: tl.constexpr):
    """One program = one query block, iterating over key blocks (the FA inner loop)."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, HD)

    q = tl.load(Q + offs_m[:, None] * SQ + offs_d[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)

    for start in range(0, SK, BN):
        cols = start + offs_n
        k = tl.load(K + cols[None, :] * SQ + offs_d[:, None])
        s = tl.dot(q, k) * sm
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        if EXP2:
            p = tl.math.exp2(s - m_ij[:, None])
        else:
            p = tl.exp(s - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij) if EXP2 else tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + cols[:, None] * SQ + offs_d[None, :])
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc.to(tl.float16))


def measure(arch, bm, bn, hd, warps, exp2):
    src = ASTSource(
        fn=fa_fwd,
        signature={"Q": "*fp16", "K": "*fp16", "V": "*fp16", "O": "*fp16",
                   "SQ": "i32", "SK": "i32", "sm": "fp32"},
        constexprs={"BM": bm, "BN": bn, "HD": hd, "EXP2": exp2},
        attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]],
               (2,): [["tt.divisibility", 16]], (3,): [["tt.divisibility", 16]],
               (5,): [["tt.divisibility", 16]]},
    )
    k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                       options={"num_warps": warps, "num_stages": 1})
    asm = k.asm["amdgcn"]

    def num(rx, default=0):
        m = rx.search(asm)
        return int(m.group(1)) if m else default

    return num(SPILL), num(PRIV), num(NGPR), num(SHMEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["gfx1030", "gfx1100"])
    ap.add_argument("--exp2", action="store_true", default=True)
    args = ap.parse_args()

    tiles = [(32, 32), (32, 64), (64, 32), (64, 64), (64, 128), (128, 64), (128, 128)]
    heads = [64, 128]
    warps_list = [1, 2, 4, 8]

    for arch in args.archs:
        print(f"\n===== {arch}: online-softmax FA forward step (BMxBN, HD, warps) =====")
        print(f"{'BMxBN':>9} {'HD':>4} {'W':>2} {'spills':>7} {'privB':>6} {'vgpr':>5} {'ldsB':>6}")
        free, spill_rows = [], []
        for (bm, bn) in tiles:
            for hd in heads:
                for w in warps_list:
                    if bm < w * 32 or bn < 16:
                        continue
                    try:
                        sp, priv, ngpr, lds = measure(arch, bm, bn, hd, w, args.exp2)
                    except Exception as e:  # noqa: BLE001 - report, never hide
                        print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2}  FAILED {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:70]}")
                        continue
                    print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2} {sp:>7} {priv:>6} {ngpr:>5} {lds:>6}"
                          + ("  <== SPILLS" if sp else ""))
                    (spill_rows if sp else free).append((bm, bn, hd, w, sp, ngpr))
        print(f"\n  spill-free configs ({len(free)}):")
        for bm, bn, hd, w, _, ngpr in free:
            print(f"    {bm}x{bn} HD={hd} warps={w} (next_vgpr={ngpr})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
