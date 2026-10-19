#!/usr/bin/env python3
"""Q6_K fused Triton dequant, validated bit-exact against the plugin's torch path.

Why Q6_K and not more Q5_K work: the production DiT file's header (read directly, not assumed) is
Q5_K 48 tensors / 4.027 B params (56.6%), Q6_K 144 tensors / 2.953 B params (41.5%), BF16 72,
Q8_0 1. So the already-shipped Q5_K kernel covers the majority but leaves 41.5% of the dequant
volume on the 26-30 GB/s torch path.

Q6_K block layout, 210 bytes per 256 values (6.56 bpv), taken from dequantize_blocks_Q6_K:
  [0:128]   ql   low nibbles, array order (2 groups of 64 bytes) x (low|high nibble) x 64 lanes
  [128:192] qh   2 high bits, array order (2 groups of 32 bytes) x (4 shift slots of 32)
  [192:208] scales  int8, one per 16-value superblock
  [208:210] d       fp16
  value i: ql byte = (i//128)*64 + (i%64), nibble = (i%128)//64
           qh byte = (i//128)*32 + (i%32), shift = 2*((i%128)//32)
           scale index = i//16
Rounding: torch does d*scales -> one fp16 materialisation, then *q -> a second. Reproduced exactly.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
sys.path.insert(0, "/home/trave/tools/ComfyUI/custom_nodes")
import importlib

import torch
import triton
import triton.language as tl

deq = importlib.import_module("ComfyUI-GGUF.dequant")
QK_K, Q6K_BYTES = 256, 210


@triton.jit
def deq6k(B, Out, n_blocks, BPP: tl.constexpr, BB: tl.constexpr, KK: tl.constexpr,
          OT: tl.constexpr):
    pid = tl.program_id(0)
    blk = pid * BPP + tl.arange(0, BPP)[:, None]
    i = tl.arange(0, KK)[None, :]
    valid = blk < n_blocks
    base = blk * BB

    half = i // 128           # which 64-byte ql group / 32-byte qh group
    qh_byte = base + 128 + half * 32 + (i % 32)
    shift = 2 * ((i % 128) // 32)
    hi = (tl.load(B + qh_byte, mask=valid, other=0).to(tl.int32) >> shift) & 3

    ql_byte = base + half * 64 + (i % 64)
    nib = (i % 128) // 64
    lo = (tl.load(B + ql_byte, mask=valid, other=0).to(tl.int32) >> (nib * 4)) & 15

    q = ((lo | (hi << 4)).to(tl.int32) - 32).to(tl.float32)

    d_bits = tl.load(B + base + 208, mask=valid, other=0).to(tl.uint16) | (
        tl.load(B + base + 209, mask=valid, other=0).to(tl.uint16) << 8)
    d = d_bits.to(tl.float16, bitcast=True).to(OT)
    sc = tl.load(B + base + 192 + (i // 16), mask=valid, other=0).to(tl.int8).to(OT)

    dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(OT)
    out = (dscale.to(tl.float32) * q).to(OT)
    tl.store(Out + blk * KK + i, out, mask=valid)


_OT = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def triton_q6k(blocks, dtype=torch.float16, bpp=4, num_warps=4):
    n = blocks.shape[0]
    out = torch.empty(n * QK_K, device=blocks.device, dtype=dtype)
    deq6k[(triton.cdiv(n, bpp),)](blocks, out, n, BPP=bpp, BB=Q6K_BYTES, KK=QK_K,
                                   OT=_OT[dtype], num_warps=num_warps)
    return out.reshape(n, QK_K)


def torch_q6k(blocks, dtype):
    return deq.dequantize_blocks_Q6_K(blocks, QK_K, Q6K_BYTES, dtype=dtype)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def clock_ramp(seconds=8.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a) * 1e-4
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def compare(ref, got):
    """NaN-aware exactness report.

    torch.equal is the wrong primitive here: random block bytes put a NaN in the fp16 `d`
    field (offset 224 of a Q6_K block), and NaN != NaN, so an *identical* decode reports as
    unequal. Compare positions of NaN/Inf separately, then require exact equality on the
    finite values.
    """
    rnan, gnan = torch.isnan(ref), torch.isnan(got)
    rin, gin = torch.isinf(ref), torch.isinf(got)
    finite = torch.isfinite(ref) & torch.isfinite(got)
    bad_nan = bool((rnan != gnan).any())
    bad_inf = bool((rin != gin).any())
    sign_inf = bool(((ref[rin].signbit() != got[gin].signbit()).any())
                    if rin.any() else False)
    finite_mismatch = int((ref[finite] != got[finite]).sum())
    return {"nan_positions_differ": bad_nan, "inf_positions_differ": bad_inf,
            "inf_sign_differs": sign_inf, "finite_mismatch_count": finite_mismatch,
            "nan_count": int(rnan.sum()),
            "exact": not (bad_nan or bad_inf or sign_inf) and finite_mismatch == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sanitize-d", action="store_true",
                    help="force the block's fp16 d field finite, so the comparison has no "
                         "NaN/Inf at all and exactness is unambiguous")
    ap.add_argument("--out", default="/tmp/q21-p1/q6k.json")
    args = ap.parse_args()

    def make_blocks(n):
        b = torch.randint(0, 256, (n, Q6K_BYTES), dtype=torch.uint8)
        if args.sanitize_d:
            # exponent 0x7C.. is Inf/NaN in fp16; pick a normal exponent, keep random mantissa
            hi = b[:, Q6K_BYTES - 1]
            hi = (hi & 0x03) | 0x38
            b[:, Q6K_BYTES - 1] = hi
        return b.cuda()

    rep = {"exactness": []}
    for n in (1, 3, 5, 8, 9, 17, 64, 4097):
        b = make_blocks(n)
        for dt in (torch.float16, torch.bfloat16):
            ref = torch_q6k(b, dt)
            got = triton_q6k(b, dt)
            c = compare(ref, got)
            rep["exactness"].append({"n": n, "dtype": str(dt), "sanitize_d": args.sanitize_d,
                                     **c})
            print(f"n={n:6d} {str(dt):16s} sanitize_d={args.sanitize_d} exact={c['exact']} "
                  f"finite_mismatch={c['finite_mismatch_count']} "
                  f"nan={c['nan_count']} nan_pos_differ={c['nan_positions_differ']} "
                  f"inf_pos_differ={c['inf_positions_differ']}", flush=True)

    if args.bench and torch.cuda.is_available():
        clock_ramp()
        rep["bench"] = []
        for label, out_f, in_f in [("q6k_gate_up", 24576, 4096), ("q6k_proj", 4096, 4096),
                                   ("q6k_down", 4096, 12288)]:
            n = (out_f * in_f) // QK_K
            b = make_blocks(n)
            fn_t = lambda: torch_q6k(b, torch.float16)
            fn_k = lambda: triton_q6k(b, torch.float16)
            for _ in range(3):
                timed(fn_t)
            t_ms = min(timed(fn_t) for _ in range(7))
            for _ in range(3):
                timed(fn_k)
            k_ms = min(timed(fn_k) for _ in range(7))
            traffic = (n * Q6K_BYTES + n * QK_K * 2) / 1e9
            row = {"case": label, "torch_ms": round(t_ms, 3), "triton_ms": round(k_ms, 3),
                   "speedup": round(t_ms / k_ms, 2), "GBps": round(traffic / (k_ms / 1000), 1),
                   "exact": compare(fn_t(), fn_k())["exact"]}
            rep["bench"].append(row)
            print(f"{label:14s} torch {row['torch_ms']:8.3f} -> triton {row['triton_ms']:8.3f} ms "
                  f"x{row['speedup']:.2f} ({row['GBps']:.0f} GB/s) exact={row['exact']}", flush=True)
            del b
            torch.cuda.empty_cache()

    json.dump(rep, open(args.out, "w"), indent=1)
    bad = [e for e in rep["exactness"] if not e["exact"]]
    print("exactness failures:", len(bad), "of", len(rep["exactness"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
