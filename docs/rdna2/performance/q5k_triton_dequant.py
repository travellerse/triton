#!/usr/bin/env python3
"""Fused Triton dequant for GGUF Q5_K, validated bit-exact against the plugin's torch path.

Baseline being attacked: the torch implementation runs at 26-30 GB/s while a plain copy of the same
bytes reaches 206-441 GB/s (docs/rdna2/performance/rdna2-dequant-baseline.md), because it expands
each 256-value block through reshapes, broadcasts and separate fp16 elementwise kernels.

Bit-exactness requires copying torch's rounding sequence, not just the arithmetic: PyTorch computes
half arithmetic in fp32 (opmath) but materialises each tensor op as fp16, so `d*q - dm` rounds twice.
This kernel reproduces that: per-superblock scale -> fp16, product -> fp16, difference -> fp16.

Block layout (176 bytes, QK_K = 256):
  [0:2] d fp16 | [2:4] dmin fp16 | [4:16] scales | [16:48] qh | [48:176] qs
  value v in 0..255: g = v>>5 (superblock), c = v&31
     qs byte  = 48 + (v>>6)*32 + c, nibble = (g & 1)      -> low 4 bits
     qh byte  = 16 + c, bit = g                            -> 5th bit
     scales: g<4 -> lo6(scales[g&3]);  else -> lo nibble of scales[8+i] | bits4-5 of scales[i]
             min : g<4 -> lo6(scales[4+g&3]); else -> hi nibble of scales[8+i] | bits4-5 of that byte
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
sys.path.insert(0, "/home/trave/tools/ComfyUI/custom_nodes/ComfyUI-GGUF")
import torch
import triton
import triton.language as tl

QK_K = 256
BLOCK_BYTES = 176


@triton.jit
def deq5k(B, Out, n_blocks, BPP: tl.constexpr, BB: tl.constexpr, KK: tl.constexpr,
              OT: tl.constexpr):
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
    d = d_bits.to(tl.float16, bitcast=True).to(OT)
    dmin = dm_bits.to(tl.float16, bitcast=True).to(OT)

    qs = tl.load(B + base + 48 + (v >> 6) * 32 + c, mask=valid, other=0).to(tl.int32)
    qh = tl.load(B + base + 16 + c, mask=valid, other=0).to(tl.int32)
    lo = (qs >> ((g & 1) * 4)) & 15
    hi = (qh >> g) & 1
    q = (lo | (hi << 4)).to(tl.float32)

    # scales[g&3] carries d (g<4) or the 5th/6th bits of sc (g>=4); scales[4+(g&3)] carries m
    s_d = tl.load(B + base + 4 + (g & 3), mask=valid, other=0).to(tl.int32)
    s_m = tl.load(B + base + 8 + (g & 3), mask=valid, other=0).to(tl.int32)
    s_md = tl.load(B + base + 12 + (g & 3), mask=valid, other=0).to(tl.int32)
    sc = tl.where(g < 4, s_d & 63, (s_md & 15) | ((s_d >> 2) & 48))
    mv = tl.where(g < 4, s_m & 63, (s_md >> 4) | ((s_m >> 2) & 48))

    # torch rounds each tensor op to fp16 (opmath is fp32 inside one op), so keep both roundings
    # torch keeps each tensor op in fp32 (opmath) but materialises the result in the output
    # dtype, so the rounding happens three times, not once
    dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(OT)
    minscale = (dmin.to(tl.float32) * mv.to(tl.float32)).to(OT)
    prod = (dscale.to(tl.float32) * q).to(OT)
    out = (prod.to(tl.float32) - minscale.to(tl.float32)).to(OT)

    tl.store(Out + blk * KK + v, out, mask=valid)


_OT = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def triton_dequant(blocks_u8, dtype=torch.float16, bpp=8, num_warps=4):
    n_blocks = blocks_u8.shape[0]
    out = torch.empty(n_blocks * QK_K, device=blocks_u8.device, dtype=dtype)
    deq5k[(triton.cdiv(n_blocks, bpp),)](blocks_u8, out, n_blocks, BPP=bpp, BB=BLOCK_BYTES,
                     KK=QK_K, OT=_OT[dtype], num_warps=num_warps)
    return out.reshape(n_blocks, QK_K)


def torch_reference(blocks_u8, dtype):
    from dequant import dequantize_blocks_Q5_K
    return dequantize_blocks_Q5_K(blocks_u8, QK_K, BLOCK_BYTES, dtype=dtype)


def random_blocks(n, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    b = torch.randint(0, 256, (n, BLOCK_BYTES), generator=g, dtype=torch.uint8)
    return b.to(device)


def bitwise_equal(a, b):
    # random bytes make d/dmin NaN or Inf, and NaN != NaN under torch.equal, so a plain
    # equality test reports a false failure when both sides produced the same NaN
    same_nan = torch.isnan(a) & torch.isnan(b)
    if not bool((torch.isnan(a) == torch.isnan(b)).all()):
        return False
    return bool(((a == b) | same_nan).all())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpp", type=int, default=8)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default="/tmp/q21-p1/q5k_triton.json")
    args = ap.parse_args()

    rep = {"cfg": {"BPP": args.bpp, "num_warps": args.warps}, "exactness": [], "bench": []}
    # realistic finite weights: sane fp16 d/dmin, random index and scale bytes
    for n in (1, 5, 8, 17, 256, 4097):
        b = random_blocks(n, seed=1000 + n)
        dv = torch.randn(n, dtype=torch.float16).view(torch.uint8).reshape(n, 2)
        mn = torch.randn(n, dtype=torch.float16).view(torch.uint8).reshape(n, 2)
        b[:, 0:2] = dv
        b[:, 2:4] = mn
        ref = torch_reference(b, torch.float16)
        got = triton_dequant(b, torch.float16, bpp=args.bpp, num_warps=args.warps)
        same = bool(torch.equal(ref, got))
        rep["exactness"].append({"n_blocks": n, "mode": "finite", "bitwise_equal": same,
                                 "nan_count": int(torch.isnan(ref).sum())})
        print(f"[finite] n_blocks={n:6d} bitwise={same} nans={rep['exactness'][-1]['nan_count']}",
              flush=True)

    # exactness first, over counts that are NOT multiples of the tile (tail masking)
    for n in (1, 3, 8, 9, 64, 100, 1023, 4096):
        b = random_blocks(n, seed=n)
        ref = torch_reference(b, torch.float16)
        got = triton_dequant(b, torch.float16, bpp=args.bpp, num_warps=args.warps)
        same = bitwise_equal(ref, got)
        rep["exactness"].append({"n_blocks": n, "bitwise_equal": same,
                                 "max_abs": float((ref.float() - got.float()).abs().max())})
        print(f"[exact] n_blocks={n:6d} bitwise={same} max_abs={rep['exactness'][-1]['max_abs']}",
              flush=True)
        if not same:
            bad = ~((ref == got) | (torch.isnan(ref) & torch.isnan(got)))
            idx = bad.nonzero()[0]
            print(f"   first mismatch at block {idx[0].item()} value {idx[1].item()}: "
                  f"torch={ref[idx[0], idx[1]].item()} triton={got[idx[0], idx[1]].item()}", flush=True)
    # bf16 too, since the plugin supports it
    b = random_blocks(64, seed=11)
    for dt in (torch.float16, torch.bfloat16):
        ref = torch_reference(b, dt)
        got = triton_dequant(b, dt, bpp=args.bpp, num_warps=args.warps)
        rep["exactness"].append({"n_blocks": 64, "dtype": str(dt),
                                 "bitwise_equal": bitwise_equal(ref, got)})
        print(f"[exact] dtype={dt} bitwise={rep['exactness'][-1]['bitwise_equal']}", flush=True)

    if args.bench:
        assert torch.cuda.is_available()
        clock_ramp()
        for name, out_f, in_f in [("gate_up", 24576, 4096), ("down", 4096, 12288),
                                  ("q_proj", 4096, 4096)]:
            n = (out_f * in_f) // QK_K
            blocks = random_blocks(n, seed=5)
            for bpp, warps in [(args.bpp, args.warps), (16, 8), (4, 4), (1, 4)]:
                fn = lambda: triton_dequant(blocks, torch.float16, bpp=bpp, num_warps=warps)
                fn()
                for _ in range(3):
                    timed(fn)
                ms = min(timed(fn) for _ in range(7))
                src = torch_reference(blocks, torch.float16)
                fn_t = lambda: torch_reference(blocks, torch.float16)
                for _ in range(2):
                    timed(fn_t)
                ms_t = min(timed(fn_t) for _ in range(5))
                traffic = (n * BLOCK_BYTES + n * QK_K * 2) / 1e9
                row = {"weight": name, "bpp": bpp, "warps": warps, "triton_ms": round(ms, 3),
                       "torch_ms": round(ms_t, 3), "speedup": round(ms_t / ms, 2),
                       "triton_GBps": round(traffic / (ms / 1000), 1),
                       "bitwise_vs_torch": bitwise_equal(src, fn())}
                rep["bench"].append(row)
                print(f"{name:8s} bpp{bpp:3d} w{warps}: triton {ms:8.3f} ms "
                      f"({row['triton_GBps']:6.1f} GB/s) vs torch {ms_t:8.3f} ms "
                      f"x{row['speedup']:.2f} exact={row['bitwise_vs_torch']}", flush=True)
            del blocks, src
            torch.cuda.empty_cache()

    json.dump(rep, open(args.out, "w"), indent=1)
    n_bad = sum(1 for r in rep["exactness"] if not r["bitwise_equal"])
    print(f"exactness failures: {n_bad} / {len(rep['exactness'])}  wrote {args.out}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
