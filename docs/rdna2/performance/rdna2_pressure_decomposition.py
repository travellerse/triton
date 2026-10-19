"""Where do RDNA2's 222 spilled VGPRs in a 32x32 attention dot actually come from?

The measured facts that make this worth a compiler investigation:
  * 32x32, head_dim=128, 4 warps spills 222 VGPRs with next_free_vgpr pinned at 256,
  * the tiles alone need far fewer registers: acc 32x128 f32 over 128 lanes = 32 regs,
    A tile 32x32 f16 = 4 regs, B tile 32x128 f16 = 16 regs, scores 32x32 f32 = 8 regs,
  * num_stages does not change it, and zero-spill configs are *slower* (measured),
so something in the lowering is live-range-inefficient rather than the algorithm being big.

This script compiles one FA step with one construct removed at a time and reports
`vgpr_spill_count` / `next_free_vgpr` per variant, all statically. The delta localises the
pressure to a construct, which is what tells us which pass to fix.

Usage: python3 rdna2_pressure_decomposition.py [--arch gfx1030] [--ref gfx1100]
"""

import argparse
import collections
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

SPILL = re.compile(r"\.vgpr_spill_count:\s*(\d+)")
NGPR = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")


@triton.jit
def v_full(Q, K, V, O, KN, sm, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """Reference formulation: masked key tail + online softmax + both dots."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * HD + offs_d[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    for start in range(0, KN, BN):
        offs_n = start + tl.arange(0, BN)
        ok = offs_n < KN
        k = tl.load(K + offs_n[None, :] * HD + offs_d[:, None], mask=ok[None, :], other=0.0)
        s = tl.dot(q, k) * sm
        s = tl.where(ok[None, :], s, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + offs_n[:, None] * HD + offs_d[None, :], mask=ok[:, None], other=0.0)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc.to(tl.float16))


@triton.jit
def v_no_tail_mask(Q, K, V, O, KN, sm, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """Same, but KN is assumed a multiple of BN: no where(), no masked loads."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * HD + offs_d[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    for start in range(0, KN, BN):
        offs_n = start + tl.arange(0, BN)
        k = tl.load(K + offs_n[None, :] * HD + offs_d[:, None])
        s = tl.dot(q, k) * sm
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + offs_n[:, None] * HD + offs_d[None, :])
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc.to(tl.float16))


@triton.jit
def v_no_rescale(Q, K, V, O, KN, sm, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """No running max/sum bookkeeping: single-pass exp of scores, normalize at the end.
    Mathematically only valid because the test uses constant scores; it exists to show what
    m_i/l_i/alpha cost in registers."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * HD + offs_d[None, :])
    acc = tl.zeros([BM, HD], tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    for start in range(0, KN, BN):
        offs_n = start + tl.arange(0, BN)
        k = tl.load(K + offs_n[None, :] * HD + offs_d[:, None])
        s = tl.dot(q, k) * sm
        p = tl.math.exp2(s)
        l_i += tl.sum(p, 1)
        v = tl.load(V + offs_n[:, None] * HD + offs_d[None, :])
        acc = tl.dot(p.to(tl.float16), v, acc)
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc.to(tl.float16))


@triton.jit
def v_qk_only(Q, K, O, KN, sm, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """One dot only (the QK^T shape), softmax, store the probabilities."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * HD + offs_d[None, :])
    for start in range(0, KN, BN):
        offs_n = start + tl.arange(0, BN)
        k = tl.load(K + offs_n[None, :] * HD + offs_d[:, None])
        s = tl.dot(q, k) * sm
        p = tl.math.exp2(s - tl.max(s, 1)[:, None])
        tl.store(O + offs_m[:, None] * BN + offs_n[None, :], p.to(tl.float16))


@triton.jit
def v_pv_only(Q, K, V, O, KN, sm, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """One dot only (the PV shape): probabilities come from memory, no softmax bookkeeping."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    acc = tl.zeros([BM, HD], tl.float32)
    for start in range(0, KN, BN):
        offs_n = start + tl.arange(0, BN)
        p = tl.load(Q + offs_m[:, None] * BN + offs_n[None, :])
        v = tl.load(V + offs_n[:, None] * HD + offs_d[None, :])
        acc = tl.dot(p, v, acc)
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc.to(tl.float16))


@triton.jit
def v_pv_only_f32(Q, K, V, O, KN, sm, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """fp32 operands: no WMMA and no MFMA can take f32, so RDNA3 is forced onto the same FMA
    lowering RDNA2 uses. If RDNA3 shows the same inverted spill trend here, the defect is in
    the FMA path (shared code), not in RDNA2-specific handling."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    acc = tl.zeros([BM, HD], tl.float32)
    for start in range(0, KN, BN):
        offs_n = start + tl.arange(0, BN)
        p = tl.load(Q + offs_m[:, None] * BN + offs_n[None, :]).to(tl.float32)
        v = tl.load(V + offs_n[:, None] * HD + offs_d[None, :]).to(tl.float32)
        acc = tl.dot(p, v, acc)
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc)


VARIANTS = [
    ("full (masked tail + online softmax)", v_full),
    ("no tail masking", v_no_tail_mask),
    ("no online rescale (no m_i/alpha)", v_no_rescale),
    ("QK dot only", v_qk_only),
    ("PV dot only", v_pv_only),
    ("PV dot only, fp32 (forces FMA on RDNA3 too)", v_pv_only_f32),
]


TYPES = {"Q": "*fp16", "K": "*fp16", "V": "*fp16", "O": "*fp16", "KN": "i32", "sm": "fp32"}


def measure(fn, arch, bm, bn, hd, warps):
    names = [a for a in fn.arg_names if a not in ("BM", "BN", "HD")]
    signature = {n: TYPES[n] for n in names}
    src = ASTSource(fn=fn, signature=signature,
                    constexprs={"BM": bm, "BN": bn, "HD": hd},
                    attrs={(i,): [["tt.divisibility", 16]]
                           for i, n in enumerate(names) if n in ("Q", "K", "V", "O")})
    k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                       options={"num_warps": warps, "num_stages": 1})
    asm = k.asm["amdgcn"]

    def num(rx):
        m = rx.search(asm)
        return int(m.group(1)) if m else -1
    return num(SPILL), num(NGPR)


def detail(fn, arch, bm, bn, hd, warps):
    """Same measurement plus the memory-instruction mix and the operand layouts, so an
    inverted spill-vs-warps trend can be attributed to replication or to a layout choice."""
    names = [a for a in fn.arg_names if a not in ("BM", "BN", "HD")]
    signature = {n: TYPES[n] for n in names}
    src = ASTSource(fn=fn, signature=signature,
                    constexprs={"BM": bm, "BN": bn, "HD": hd},
                    attrs={(i,): [["tt.divisibility", 16]]
                           for i, n in enumerate(names) if n in ("Q", "K", "V", "O")})
    k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                       options={"num_warps": warps, "num_stages": 1})
    asm = k.asm["amdgcn"]
    ttgir = k.asm["ttgir"]
    loads = collections.Counter(re.findall(
        r"\b(?:global|buffer)_(?:load|store)[a-z0-9_]*\b|\bds_(?:read|write|load|store)[a-z0-9_]*\b", asm))
    def num(rx):
        x = rx.search(asm)
        return int(x.group(1)) if x else -1

    sp, ng = num(SPILL), num(NGPR)
    dotops = sorted(set(re.findall(r"#ttg\.dot_op<\{opIdx = \d, parent = #\w+(?:<[^>]*>)?\}>", ttgir)))
    blocks = sorted(set(re.findall(r"#blocked\d* = #ttg\.blocked<\{[^}]*\}>", ttgir)))
    return sp, ng, dict(loads), dotops, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030")
    ap.add_argument("--ref", default="gfx1100")
    ap.add_argument("--tile", default="32,32,128,4")
    ap.add_argument("--detail", action="store_true",
                    help="for one variant, sweep warps and print load mix + operand layouts")
    ap.add_argument("--variant", default="PV dot only", choices=[v[0] for v in VARIANTS])
    args = ap.parse_args()

    if args.detail:
        bm, bn, hd, _ = (int(x) for x in args.tile.split(","))
        fn = dict(VARIANTS)[args.variant]
        for w in (1, 2, 4):
            try:
                sp, ng, loads, dotops, blocks = detail(fn, args.arch, bm, bn, hd, w)
            except Exception as e:  # noqa: BLE001
                print(f"w={w}: FAILED {type(e).__name__}: {str(e).splitlines()[-1][:90]}")
                continue
            print(f"\nw={w}  spills={sp} next_free_vgpr={ng}")
            print(f"   loads: {loads}")
            print(f"   dot_operand layouts: {dotops}")
            for b in blocks:
                print(f"   {b}")
        return

    bm, bn, hd, w = (int(x) for x in args.tile.split(","))

    for arch in [args.arch, args.ref]:
        if not arch:
            continue
        print(f"\n== {arch}  tile BM={bm} BN={bn} HD={hd} warps={w} ==")
        print(f"{'variant':38s} {'spills':>7} {'vgpr':>6}")
        for label, fn in VARIANTS:
            try:
                sp, ng = measure(fn, arch, bm, bn, hd, w)
                print(f"{label:38s} {sp:7d} {ng:6d}")
            except Exception as e:  # noqa: BLE001
                print(f"{label:38s}  FAILED {type(e).__name__}: {str(e).splitlines()[0][:70]}")


if __name__ == "__main__":
    main()
