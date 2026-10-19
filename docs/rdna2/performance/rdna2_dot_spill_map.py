"""RDNA2 register-spill map for attention-shaped dots (static, no device needed).

Why this exists: the in-thread-transpose probe (rdna2_inthread_transpose_probe.py) showed
that RDNA2's dot runs on the FMA path (`v_dot2c_f32_f16`) with `#ttg.dot_op` whose parent
is a plain BlockedLayout, and that for a 64x64x64 fp16 dot the emitted kernel reports
`.vgpr_spill_count: 40` and a 164-byte private segment. Scratch traffic goes through
`buffer_load/store` to *global* memory, so a spilling attention tile is not a micro-level
curiosity -- it is a direct hit on the same inner loop that measured 5.96 TFLOPs for the
PV-shaped contraction versus 13.77 for the QK-shaped one (see p4-dot-ceiling-rdna2.md).

This script sweeps (tile M, tile N, K=head_dim, num_warps) and reports, per arch:

  * vgpr_spill_count and private_segment_fixed_size (from the amdhsa metadata),
  * next_free_vgpr (register pressure),
  * the contraction instruction used and how many of them.

Any config with a nonzero spill count is a config RDNA2 fast-attention must avoid, so the
output is a design constraint table, not a benchmark.

Usage: python3 rdna2_dot_spill_map.py [--archs gfx1030 gfx1100]
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
CONTR = re.compile(r"\b(v_dot2c_f32_f16|v_wmma_\w+|v_mfma_\w+|v_dot4_i32_i8)\b")


@triton.jit
def dot_tile(Q, V, O, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    p = tl.load(Q + offs_m[:, None] * BN + offs_n[None, :])
    v = tl.load(V + offs_n[:, None] * D + offs_d[None, :])
    acc = tl.dot(p, v)
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], acc.to(tl.float16))


def measure(arch, bm, bn, d, warps):
    src = ASTSource(
        fn=dot_tile,
        signature={"Q": "*fp16", "V": "*fp16", "O": "*fp16"},
        constexprs={"BM": bm, "BN": bn, "D": d},
        attrs={(i,): [["tt.divisibility", 16]] for i in range(3)},
    )
    k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                       options={"num_warps": warps, "num_stages": 1})
    asm = k.asm["amdgcn"]

    def num(rx, default=0):
        m = rx.search(asm)
        return int(m.group(1)) if m else default

    contr = {}
    for tok in CONTR.findall(asm):
        contr[tok] = contr.get(tok, 0) + 1
    return num(SPILL), num(PRIV), num(NGPR), contr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["gfx1030", "gfx1100"])
    args = ap.parse_args()

    tiles = [(32, 32), (64, 64), (128, 64), (64, 128), (128, 128)]
    dims = [64, 128]
    warp_counts = [1, 2, 4, 8]

    for arch in args.archs:
        print(f"\n===== {arch}: (BMxBN, D, warps) -> vgpr_spills priv_bytes next_vgpr contractions")
        print(f"{'tile':>10} {'D':>4} {'W':>2} {'spills':>7} {'priv':>5} {'vgpr':>5}  contraction")
        for (bm, bn) in tiles:
            for d in dims:
                for w in warp_counts:
                    if bm * bn < w * 32 or d < 16:
                        continue
                    try:
                        spills, priv, ngpr, contr = measure(arch, bm, bn, d, w)
                    except Exception as e:  # noqa: BLE001 - report, never hide
                        print(f"{bm:>4}x{bn:<5} {d:>4} {w:>2}  COMPILE FAILED "
                              f"{type(e).__name__}: {str(e).splitlines()[0][:70]}")
                        continue
                    flag = "  <== SPILLS" if spills else ""
                    cs = ", ".join(f"{k}x{v}" for k, v in sorted(contr.items()))
                    print(f"{bm:>4}x{bn:<5} {d:>4} {w:>2} {spills:>7} {priv:>5} {ngpr:>5}  {cs}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
