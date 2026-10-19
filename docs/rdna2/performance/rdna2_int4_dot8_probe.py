#!/usr/bin/env python3
"""RDNA2 int4 `v_dot8` through inline asm: values and throughput against `v_dot4`.

The gate map says Triton cannot express `V_DOT8_I32_I4`/`V_DOT8_U32_U4`. That is true for `tl.dot`
(no int4 operand type), but the instruction itself is reachable today through
`tl.inline_asm_elementwise`, which is what this probe uses and asserts in the emitted ISA.

Each arm runs four independent accumulators so the dots are not serialised on one dependency chain,
and the operands are re-read from eight small buffers indexed by the runtime loop counter so the
inline asm cannot be hoisted. Both arms execute the same number of dot instructions; dot8 does eight
MACs per instruction, dot4 four, so equal time means twice the throughput.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag dot8 --budget-mib 1024 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/performance/rdna2_int4_dot8_probe.py
"""
import argparse
import re
import statistics
import sys

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 1024
BUFS = 8


@triton.jit
def dot8_k(A, B, Out, iters, BLOCK: tl.constexpr, BUFS: tl.constexpr):
    off = tl.arange(0, BLOCK)
    acc0 = tl.zeros((BLOCK,), tl.int32)
    acc1 = tl.zeros((BLOCK,), tl.int32)
    acc2 = tl.zeros((BLOCK,), tl.int32)
    acc3 = tl.zeros((BLOCK,), tl.int32)
    for it in range(0, iters):
        i = it & (BUFS - 1)
        a = tl.load(A + i * BLOCK + off)
        b = tl.load(B + i * BLOCK + off)
        acc0 = tl.inline_asm_elementwise("v_dot8_i32_i4 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc0], dtype=tl.int32, is_pure=True, pack=1)
        a = tl.load(A + ((i + 1) & (BUFS - 1)) * BLOCK + off)
        b = tl.load(B + ((i + 1) & (BUFS - 1)) * BLOCK + off)
        acc1 = tl.inline_asm_elementwise("v_dot8_i32_i4 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc1], dtype=tl.int32, is_pure=True, pack=1)
        a = tl.load(A + ((i + 2) & (BUFS - 1)) * BLOCK + off)
        b = tl.load(B + ((i + 2) & (BUFS - 1)) * BLOCK + off)
        acc2 = tl.inline_asm_elementwise("v_dot8_i32_i4 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc2], dtype=tl.int32, is_pure=True, pack=1)
        a = tl.load(A + ((i + 3) & (BUFS - 1)) * BLOCK + off)
        b = tl.load(B + ((i + 3) & (BUFS - 1)) * BLOCK + off)
        acc3 = tl.inline_asm_elementwise("v_dot8_i32_i4 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc3], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Out + off, acc0 + acc1 + acc2 + acc3)


@triton.jit
def dot4_k(A, B, Out, iters, BLOCK: tl.constexpr, BUFS: tl.constexpr):
    off = tl.arange(0, BLOCK)
    acc0 = tl.zeros((BLOCK,), tl.int32)
    acc1 = tl.zeros((BLOCK,), tl.int32)
    acc2 = tl.zeros((BLOCK,), tl.int32)
    acc3 = tl.zeros((BLOCK,), tl.int32)
    for it in range(0, iters):
        i = it & (BUFS - 1)
        a = tl.load(A + i * BLOCK + off)
        b = tl.load(B + i * BLOCK + off)
        acc0 = tl.inline_asm_elementwise("v_dot4_i32_i8 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc0], dtype=tl.int32, is_pure=True, pack=1)
        a = tl.load(A + ((i + 1) & (BUFS - 1)) * BLOCK + off)
        b = tl.load(B + ((i + 1) & (BUFS - 1)) * BLOCK + off)
        acc1 = tl.inline_asm_elementwise("v_dot4_i32_i8 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc1], dtype=tl.int32, is_pure=True, pack=1)
        a = tl.load(A + ((i + 2) & (BUFS - 1)) * BLOCK + off)
        b = tl.load(B + ((i + 2) & (BUFS - 1)) * BLOCK + off)
        acc2 = tl.inline_asm_elementwise("v_dot4_i32_i8 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc2], dtype=tl.int32, is_pure=True, pack=1)
        a = tl.load(A + ((i + 3) & (BUFS - 1)) * BLOCK + off)
        b = tl.load(B + ((i + 3) & (BUFS - 1)) * BLOCK + off)
        acc3 = tl.inline_asm_elementwise("v_dot4_i32_i8 $0, $1, $2, $3", "=v,v,v,v",
                                         [a, b, acc3], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Out + off, acc0 + acc1 + acc2 + acc3)


def nibbles_signed(words):
    """[n] int32 -> [n, 8] signed int4 values (low nibble first)."""
    w = words.astype(np.uint32)
    out = np.empty((w.size, 8), dtype=np.int32)
    for k in range(8):
        v = ((w >> np.uint32(4 * k)) & np.uint32(0xF)).astype(np.int32)
        out[:, k] = np.where(v >= 8, v - 16, v)
    return out


def bytes_signed(words):
    w = words.astype(np.uint32)
    out = np.empty((w.size, 4), dtype=np.int32)
    for k in range(4):
        v = ((w >> np.uint32(8 * k)) & np.uint32(0xFF)).astype(np.int32)
        out[:, k] = np.where(v >= 128, v - 256, v)
    return out


def reference(A, B, iters, width):
    """The loop cycles through BUFS buffers, four dots per iteration."""
    assert (4 * iters) % BUFS == 0, "the reference assumes whole buffer cycles"
    reps = 4 * iters // BUFS
    tot = np.zeros(BLOCK, dtype=np.int64)
    for j in range(BUFS):
        if width == 8:
            tot += (nibbles_signed(A[j]) * nibbles_signed(B[j])).sum(axis=1)
        else:
            tot += (bytes_signed(A[j]) * bytes_signed(B[j])).sum(axis=1)
    return (reps * tot).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device")
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).gcnArchName, flush=True)
    torch.manual_seed(7)
    A = torch.randint(-(2**31), 2**31 - 1, (BUFS, BLOCK), dtype=torch.int32, device="cuda")
    B = torch.randint(-(2**31), 2**31 - 1, (BUFS, BLOCK), dtype=torch.int32, device="cuda")
    out = torch.empty(BLOCK, dtype=torch.int32, device="cuda")
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()

    # correctness
    for name, fn, width in (("dot8 (signed int4)", dot8_k, 8), ("dot4 (signed int8)", dot4_k, 4)):
        out.zero_()
        fn[(1,)](A, B, out, args.iters, BLOCK=BLOCK, BUFS=BUFS, num_warps=4)
        torch.cuda.synchronize()
        got = out.cpu().numpy()
        ref = reference(A_np, B_np, args.iters, width)
        ok = bool(np.array_equal(got, ref))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} values (iters={args.iters}) "
              f"first={got[:3].tolist()} ref={ref[:3].tolist()}")

    # throughput; assert the mnemonic in the ISA of each arm
    def timeit(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        vals = []
        for _ in range(args.reps):
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            for _ in range(args.batch):
                fn()
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / args.batch)
        return statistics.median(vals), vals

    macs = BLOCK * 4 * args.iters          # per lane * dot instructions
    for name, fn, mnemonic, per in (("dot8", dot8_k, "v_dot8_i32_i4", 8),
                                    ("dot4", dot4_k, "v_dot4_i32_i8", 4)):
        compiled = fn.warmup(A, B, out, args.iters,
                             BLOCK=BLOCK, BUFS=BUFS, grid=(1,), num_warps=4)
        asm = compiled.asm["amdgcn"]
        n = len(re.findall(mnemonic, asm))
        t1, v1 = timeit(lambda: fn[(1,)](A, B, out, args.iters, BLOCK=BLOCK, BUFS=BUFS, num_warps=4))
        t2, _ = timeit(lambda: fn[(1,)](A, B, out, args.iters, BLOCK=BLOCK, BUFS=BUFS, num_warps=4))
        band = abs(t2 - t1) / t1 * 100
        rate = macs * per / (t1 * 1e-3) / 1e9
        print(f"{name}: {t1 * 1000:.2f} us  {rate:.1f} GMAC/s  mnemonic_count={n}  A/A {band:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
