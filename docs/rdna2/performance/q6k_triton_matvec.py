#!/usr/bin/env python3
"""Fused GGUF Q6_K dequant matvec for decode-shaped (M=1) contractions.

The shipped path dequantises a whole Q6_K tensor with the validated kernel and then reads the fp16
weight back for the matvec. This kernel decodes each 210-byte block in registers, multiplies it by
the activation block and accumulates in f32. The decode keeps the shipped kernel's two fp16
roundings (`d*sc`, then `*q`); only the f32 summation order differs from the reference.

Q6_K block layout (210 bytes, 256 values), from the shipped kernel:
  [0:128] ql, [128:192] qh, [192:208] int8 scales (one per 16), [208:210] d fp16.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag q6k-matvec --budget-mib 3072 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/performance/q6k_triton_matvec.py
"""
import argparse
import statistics
import sys

import torch
import triton
import triton.language as tl

QK_K = 256
Q6K_BYTES = 210


@triton.jit
def mv6k(B, X, Out, N, K, BN: tl.constexpr, KK: tl.constexpr, BB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    valid = rows < N
    vmask = valid[:, None]
    i = tl.arange(0, KK)
    nb = K // KK

    half = i // 128
    shift = 2 * ((i % 128) // 32)
    nib = (i % 128) // 64
    sc_idx = i // 16

    acc = tl.zeros((BN, KK), dtype=tl.float32)
    for kb in range(0, nb):
        base = (rows * nb + kb)[:, None] * BB

        qh = tl.load(B + base + 128 + half * 32 + (i % 32), mask=vmask, other=0).to(tl.int32)
        hi = (qh >> shift) & 3
        ql = tl.load(B + base + half * 64 + (i % 64), mask=vmask, other=0).to(tl.int32)
        lo = (ql >> (nib * 4)) & 15
        q = ((lo | (hi << 4)).to(tl.int32) - 32).to(tl.float32)

        d_bits = tl.load(B + base + 208, mask=vmask, other=0).to(tl.uint16) | (
            tl.load(B + base + 209, mask=vmask, other=0).to(tl.uint16) << 8)
        d = d_bits.to(tl.float16, bitcast=True)
        sc = tl.load(B + base + 192 + sc_idx, mask=vmask, other=0).to(tl.int8)

        dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(tl.float16)
        w = (dscale.to(tl.float32) * q).to(tl.float16)

        xb = tl.load(X + kb * KK + i).to(tl.float32)
        acc += w.to(tl.float32) * xb[None, :]

    tl.store(Out + rows, tl.sum(acc, axis=1), mask=valid)


@triton.jit
def deq6k_ref(B, Out, n_blocks, BPP: tl.constexpr, BB: tl.constexpr, KK: tl.constexpr):
    """Copy of the shipped dequant kernel, kept in-process so the A/B shares one clock state."""
    pid = tl.program_id(0)
    blk = pid * BPP + tl.arange(0, BPP)[:, None]
    i = tl.arange(0, KK)[None, :]
    valid = blk < n_blocks
    base = blk * BB
    half = i // 128
    qh_byte = base + 128 + half * 32 + (i % 32)
    shift = 2 * ((i % 128) // 32)
    hi = (tl.load(B + qh_byte, mask=valid, other=0).to(tl.int32) >> shift) & 3
    ql_byte = base + half * 64 + (i % 64)
    nib = (i % 128) // 64
    lo = (tl.load(B + ql_byte, mask=valid, other=0).to(tl.int32) >> (nib * 4)) & 15
    q = ((lo | (hi << 4)).to(tl.int32) - 32).to(tl.float32)
    d_bits = tl.load(B + base + 208, mask=valid, other=0).to(tl.uint16) | (
        tl.load(B + base + 209, mask=valid, other=0).to(tl.uint16) << 8)
    d = d_bits.to(tl.float16, bitcast=True)
    sc = tl.load(B + base + 192 + (i // 16), mask=valid, other=0).to(tl.int8)
    dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(tl.float16)
    out = (dscale.to(tl.float32) * q).to(tl.float16)
    tl.store(Out + blk * KK + i, out, mask=valid)


def fused_matvec(blocks, x, N, K, bn=8, num_warps=4):
    out = torch.empty(N, dtype=torch.float32, device=blocks.device)
    mv6k[(triton.cdiv(N, bn),)](blocks, x, out, N, K, BN=bn, KK=QK_K,
                                BB=Q6K_BYTES, num_warps=num_warps)
    return out


def shipped_matvec(blocks, x, N, K, bpp=8):
    n_blocks = blocks.shape[0]
    deq = torch.empty(n_blocks * QK_K, device=blocks.device, dtype=torch.float16)
    deq6k_ref[(triton.cdiv(n_blocks, bpp),)](blocks, deq, n_blocks, BPP=bpp,
                                             BB=Q6K_BYTES, KK=QK_K, num_warps=4)
    return torch.matmul(deq.reshape(N, K), x)


def make_blocks(n_blocks, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = torch.randint(0, 256, (n_blocks, Q6K_BYTES), generator=g, dtype=torch.uint8)
    # d is the last fp16; random bytes can make it NaN/Inf, so fix it to 1.0
    blocks[:, 208] = 0x00
    blocks[:, 209] = 0x3C
    return blocks.to("cuda")


def reference(blocks, x, N, K):
    n_blocks = blocks.shape[0]
    deq = torch.empty(n_blocks * QK_K, device=blocks.device, dtype=torch.float16)
    deq6k_ref[(triton.cdiv(n_blocks, 8),)](blocks, deq, n_blocks, BPP=8,
                                           BB=Q6K_BYTES, KK=QK_K, num_warps=4)
    return deq.reshape(N, K).float() @ x.float()


def check(N, K, bn):
    nb = K // QK_K
    blocks = make_blocks(N * nb)
    x = torch.randn(K, dtype=torch.float16, device="cuda")
    got = fused_matvec(blocks, x, N, K, bn=bn)
    ref = reference(blocks, x, N, K)
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
    ok = rel < 2e-2
    print(f"[{'PASS' if ok else 'FAIL'}] fused Q6_K matvec {N}x{K} (BN={bn}) "
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
    print(f"{N}x{K} BN={bn}: fused {tf:.3f} ms | shipped(dequant+matmul) {ts:.3f} ms | "
          f"{ts / tf:.2f}x | A/A {band:.2f}% | {N * K / tf * 1e-6:.1f} Gweight/s")


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
        for n, k in ((4096, 4096), (4096, 12288), (24576, 4096)):
            bench(n, k, 8, 4, args.reps, args.batch)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
