#!/usr/bin/env python3
"""Fused GGUF Q5_K dequant matvec: decode weights in registers and multiply by the activation.

The shipped path dequantises a whole weight tensor to fp16 (validated bit-exact in
`q5k_triton_dequant.py`) and then hands the materialised tensor to rocBLAS. For a decode-shaped
matvec (M=1) that writes and re-reads the fp16 weight for nothing. This kernel keeps the decoded
block in registers, multiplies it by the activation block and accumulates in f32.

The decode arithmetic is the same sequence as the shipped kernel, including its three fp16
roundings (`d*sc`, then `*q`, then `-dmin*mv`); the only difference from the reference is the f32
summation order, so the value check uses a relative tolerance rather than exact equality.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag q5k-matvec --budget-mib 3072 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/performance/q5k_triton_matvec.py
"""
import argparse
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/trave/tools/ComfyUI")
sys.path.insert(0, "/home/trave/tools/ComfyUI/custom_nodes/ComfyUI-GGUF")

QK_K = 256
BLOCK_BYTES = 176


@triton.jit
def mv5k(B, X, Out, N, K,
         BN: tl.constexpr, KK: tl.constexpr, BB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    valid = rows < N
    vmask = valid[:, None]
    v = tl.arange(0, KK)
    g = v >> 5
    c = v & 31
    nb = K // KK

    acc = tl.zeros((BN, KK), dtype=tl.float32)
    for kb in range(0, nb):
        base = (rows * nb + kb)[:, None] * BB

        d_bits = tl.load(B + base + 0, mask=vmask, other=0).to(tl.uint16) | (
            tl.load(B + base + 1, mask=vmask, other=0).to(tl.uint16) << 8)
        dm_bits = tl.load(B + base + 2, mask=vmask, other=0).to(tl.uint16) | (
            tl.load(B + base + 3, mask=vmask, other=0).to(tl.uint16) << 8)
        d = d_bits.to(tl.float16, bitcast=True)
        dmin = dm_bits.to(tl.float16, bitcast=True)

        qs = tl.load(B + base + 48 + (v >> 6) * 32 + c, mask=vmask, other=0).to(tl.int32)
        qh = tl.load(B + base + 16 + c, mask=vmask, other=0).to(tl.int32)
        lo = (qs >> ((g & 1) * 4)) & 15
        hi = (qh >> g) & 1
        q = (lo | (hi << 4)).to(tl.float32)

        s_d = tl.load(B + base + 4 + (g & 3), mask=vmask, other=0).to(tl.int32)
        s_m = tl.load(B + base + 8 + (g & 3), mask=vmask, other=0).to(tl.int32)
        s_md = tl.load(B + base + 12 + (g & 3), mask=vmask, other=0).to(tl.int32)
        sc = tl.where(g < 4, s_d & 63, (s_md & 15) | ((s_d >> 2) & 48))
        mv = tl.where(g < 4, s_m & 63, (s_md >> 4) | ((s_m >> 2) & 48))

        dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(tl.float16)
        minscale = (dmin.to(tl.float32) * mv.to(tl.float32)).to(tl.float16)
        prod = (dscale.to(tl.float32) * q).to(tl.float16)
        w = (prod.to(tl.float32) - minscale.to(tl.float32)).to(tl.float16)

        xb = tl.load(X + kb * KK + v).to(tl.float32)
        acc += w.to(tl.float32) * xb[None, :]

    tl.store(Out + rows, tl.sum(acc, axis=1), mask=valid)


@triton.jit
def deq5k_ref(B, Out, n_blocks, BPP: tl.constexpr, BB: tl.constexpr, KK: tl.constexpr):
    """Copy of the shipped dequant kernel, kept in-process so the A/B uses one clock state."""
    pid = tl.program_id(0)
    blk = pid * BPP + tl.arange(0, BPP)[:, None]
    v = tl.arange(0, 256)[None, :]
    valid = blk < n_blocks
    base = blk * BB
    g = v >> 5
    c = v & 31
    d_bits = tl.load(B + base + 0, mask=valid, other=0).to(tl.uint16) | (
        tl.load(B + base + 1, mask=valid, other=0).to(tl.uint16) << 8)
    dm_bits = tl.load(B + base + 2, mask=valid, other=0).to(tl.uint16) | (
        tl.load(B + base + 3, mask=valid, other=0).to(tl.uint16) << 8)
    d = d_bits.to(tl.float16, bitcast=True)
    dmin = dm_bits.to(tl.float16, bitcast=True)
    qs = tl.load(B + base + 48 + (v >> 6) * 32 + c, mask=valid, other=0).to(tl.int32)
    qh = tl.load(B + base + 16 + c, mask=valid, other=0).to(tl.int32)
    lo = (qs >> ((g & 1) * 4)) & 15
    hi = (qh >> g) & 1
    q = (lo | (hi << 4)).to(tl.float32)
    s_d = tl.load(B + base + 4 + (g & 3), mask=valid, other=0).to(tl.int32)
    s_m = tl.load(B + base + 8 + (g & 3), mask=valid, other=0).to(tl.int32)
    s_md = tl.load(B + base + 12 + (g & 3), mask=valid, other=0).to(tl.int32)
    sc = tl.where(g < 4, s_d & 63, (s_md & 15) | ((s_d >> 2) & 48))
    mv = tl.where(g < 4, s_m & 63, (s_md >> 4) | ((s_m >> 2) & 48))
    dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(tl.float16)
    minscale = (dmin.to(tl.float32) * mv.to(tl.float32)).to(tl.float16)
    prod = (dscale.to(tl.float32) * q).to(tl.float16)
    out = (prod.to(tl.float32) - minscale.to(tl.float32)).to(tl.float16)
    tl.store(Out + blk * KK + v, out, mask=valid)


def fused_matvec(blocks, x, N, K, bn=8, num_warps=4):
    out = torch.empty(N, dtype=torch.float32, device=blocks.device)
    mv5k[(triton.cdiv(N, bn),)](blocks, x, out, N, K, BN=bn, KK=QK_K,
                                BB=BLOCK_BYTES, num_warps=num_warps)
    return out


def shipped_matvec(blocks, x, N, K, bpp=8):
    n_blocks = blocks.shape[0]
    deq = torch.empty(n_blocks * QK_K, device=blocks.device, dtype=torch.float16)
    deq5k_ref[(triton.cdiv(n_blocks, bpp),)](blocks, deq, n_blocks, BPP=bpp,
                                             BB=BLOCK_BYTES, KK=QK_K, num_warps=4)
    return torch.matmul(deq.reshape(N, K), x)


def make_blocks(n_blocks, sanitize_d=True, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = torch.randint(0, 256, (n_blocks, BLOCK_BYTES), generator=g, dtype=torch.uint8)
    if sanitize_d:
        # fp16 1.0 little-endian at d and dmin; random bytes can otherwise put NaN/Inf there
        blocks[:, 0] = 0x00
        blocks[:, 1] = 0x3C
        blocks[:, 2] = 0x00
        blocks[:, 3] = 0x3C
    return blocks.to("cuda")


def torch_reference(blocks, x, N, K):
    """Dequantise with the shipped kernel and contract in f32.

    The shipped dequant was validated bit-exact against the plugin's `dequantize_blocks_Q5_K`
    (`q5k_triton_dequant.py`, report `rdna2-q5k-triton-dequant.md`), so it is the reference here;
    importing the plugin directly now needs its package context and ComfyUI venv. What this check
    isolates is the fused contraction, not the decode.
    """
    n_blocks = blocks.shape[0]
    deq = torch.empty(n_blocks * QK_K, device=blocks.device, dtype=torch.float16)
    deq5k_ref[(triton.cdiv(n_blocks, 8),)](blocks, deq, n_blocks, BPP=8,
                                           BB=BLOCK_BYTES, KK=QK_K, num_warps=4)
    return deq.reshape(N, K).float() @ x.float()


def check(N, K, bn):
    nb = K // QK_K
    blocks = make_blocks(N * nb)
    x = torch.randn(K, dtype=torch.float16, device="cuda")
    got = fused_matvec(blocks, x, N, K, bn=bn)
    ref = torch_reference(blocks, x, N, K)
    denom = ref.abs().clamp_min(1e-3)
    rel = ((got - ref).abs() / denom).max().item()
    ok = rel < 2e-2
    print(f"[{'PASS' if ok else 'FAIL'}] fused matvec {N}x{K} (BN={bn}) "
          f"max_rel={rel:.3e} max_abs={(got - ref).abs().max().item():.3e}")
    return ok


def bench(N, K, bn, num_warps, reps, batch):
    nb = K // QK_K
    blocks = make_blocks(N * nb)
    x = torch.randn(K, dtype=torch.float16, device="cuda")

    def timeit(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            for _ in range(batch):
                fn()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) / batch)
        return statistics.median(out)

    fused = lambda: fused_matvec(blocks, x, N, K, bn=bn, num_warps=num_warps)
    shipped = lambda: shipped_matvec(blocks, x, N, K)
    tf = timeit(fused)
    ts = timeit(shipped)
    tf2 = timeit(fused)
    band = abs(tf2 - tf) / tf * 100
    gf = N * K / tf * 1e-6          # dequantised weights per second
    print(f"{N}x{K} BN={bn}: fused {tf:.3f} ms | shipped(dequant+matmul) {ts:.3f} ms | "
          f"{ts / tf:.2f}x | A/A {band:.2f}% | {gf:.1f} Gweight/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()
    if not (args.check or args.bench):
        args.check = args.bench = True
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).gcnArchName, flush=True)

    ok = True
    if args.check:
        ok &= check(64, 1024, 8)
        ok &= check(256, 4096, 8)
        ok &= check(4096, 4096, 8)
    if args.bench:
        for n, k in ((4096, 4096), (12288, 4096), (4096, 12288), (24576, 4096)):
            bench(n, k, 8, 4, args.reps, args.batch)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
