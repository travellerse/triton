#!/usr/bin/env python3
"""Runtime conformance probe for the AMD software fp8 NaN handling on RDNA2.

Covers the two defects fixed in the software upcast paths:
  * tl.dot_scaled's scaled upcast turned an E4M3 0x7F/0xFF (NaN) operand into a
    finite +/-480 before the NaN guard existed;
  * the plain fp8->bf16 cast did the same for E4M3 0x7F/0xFF and E5M2 0x7D..0x7F.
It also sanity-checks finite values for the 8-bit formats dot_scaled accepts,
using uniform operands so the packed nibble order does not matter.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag nan-validate --budget-mib 1024 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      python3 docs/rdna2/conformance/rdna2_fp8_nan_probe.py
"""
import math
import sys

import torch
import triton
import triton.language as tl

BM = BN = 32
BK = 64
SK = BK // 32
FAILS = []


def record(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


@triton.jit
def dot_scaled_k(a_ptr, b_ptr, sa_ptr, sb_ptr, out,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PACK_A: tl.constexpr, PACK_B: tl.constexpr,
                 FMT_A: tl.constexpr, FMT_B: tl.constexpr):
    ka: tl.constexpr = BK // PACK_A
    kb: tl.constexpr = BK // PACK_B
    sk: tl.constexpr = BK // 32
    a = tl.load(a_ptr + tl.arange(0, BM)[:, None] * ka + tl.arange(0, ka)[None, :])
    b = tl.load(b_ptr + tl.arange(0, kb)[:, None] * BN + tl.arange(0, BN)[None, :])
    sa = tl.load(sa_ptr + tl.arange(0, BM)[:, None] * sk + tl.arange(0, sk)[None, :])
    sb = tl.load(sb_ptr + tl.arange(0, BN)[:, None] * sk + tl.arange(0, sk)[None, :])
    acc = tl.dot_scaled(a, sa, FMT_A, b, sb, FMT_B)
    tl.store(out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], acc)


@triton.jit
def dot_scaled_mpack_k(a_ptr, b_ptr, sa_ptr, sb_ptr, out,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # 32x64 * 64x32 with the lhs packed along M (two adjacent rows per byte)
    a = tl.load(a_ptr + tl.arange(0, 16)[:, None] * BK + tl.arange(0, BK)[None, :])
    b = tl.load(b_ptr + tl.arange(0, BK // 2)[:, None] * BN + tl.arange(0, BN)[None, :])
    sk: tl.constexpr = BK // 32
    sa = tl.load(sa_ptr + tl.arange(0, BM)[:, None] * sk + tl.arange(0, sk)[None, :])
    sb = tl.load(sb_ptr + tl.arange(0, BN)[:, None] * sk + tl.arange(0, sk)[None, :])
    acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", lhs_k_pack=False)
    tl.store(out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], acc)


@triton.jit
def cast_k(x_ptr, o_ptr, BLOCK: tl.constexpr, E4: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    if E4:
        y = x.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    else:
        y = x.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
    tl.store(o_ptr + tl.arange(0, BLOCK), y)


def u8(seq, shape=None):
    t = torch.tensor(list(seq), dtype=torch.uint8, device="cuda")
    return t.reshape(shape) if shape else t


def run_dot(fmt_a, fmt_b, pack_a, pack_b, a, b, sa=None, sb=None):
    a = u8(a, (BM, BK // pack_a))
    b = u8(b, (BK // pack_b, BN))
    sa = u8(sa, (BM, SK)) if sa is not None else torch.full((BM, SK), 127, dtype=torch.uint8, device="cuda")
    sb = u8(sb, (BN, SK)) if sb is not None else torch.full((BN, SK), 127, dtype=torch.uint8, device="cuda")
    out = torch.empty((BM, BN), dtype=torch.float32, device="cuda")
    dot_scaled_k[(1,)](a, b, sa, sb, out, BM=BM, BN=BN, BK=BK,
                       PACK_A=pack_a, PACK_B=pack_b, FMT_A=fmt_a, FMT_B=fmt_b,
                       num_warps=4)
    return out


def check_uniform(fmt, pack, one_byte):
    out = run_dot(fmt, fmt, pack, pack,
                  [one_byte] * (BM * (BK // pack)),
                  [one_byte] * ((BK // pack) * BN))
    ok = bool(torch.equal(out, torch.full_like(out, float(BK))))
    record(f"{fmt} finite uniform operands -> {BK}", ok,
           "" if ok else f"got {out[0, 0].item()}")


def nan_row(fmt, one_byte, bad):
    a = bytearray([one_byte] * (BM * BK))
    a[0] = bad
    out = run_dot(fmt, fmt, 1, 1, bytes(a), [one_byte] * (BK * BN))
    row0_nan = bool(torch.isnan(out[0]).all())
    rest_ok = bool(torch.equal(out[1:], torch.full_like(out[1:], float(BK))))
    record(f"{fmt} 0x{bad:02X} operand poisons its row", row0_nan and rest_ok,
           f"row0_nan={row0_nan} rest_ok={rest_ok} row0={out[0, :3].tolist()}")


def cast_case(name, e4, bytes_in, nan_bytes, finite_values):
    x = u8(bytes_in)
    o = torch.empty(x.numel(), dtype=torch.bfloat16, device="cuda")
    cast_k[(1,)](x, o, BLOCK=x.numel(), E4=e4, num_warps=1)
    got = o.float()
    nan_idx = [i for i, b in enumerate(bytes_in) if b in nan_bytes]
    ok = bool(torch.isnan(got[nan_idx]).all()) if nan_idx else True
    for i, v in finite_values.items():
        ok = ok and got[i].item() == v
    record(name, ok, f"got={got.tolist()}")
    return got


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device")
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).gcnArchName, flush=True)

    check_uniform("e4m3", 1, 0x38)  # 1.0
    check_uniform("e5m2", 1, 0x3C)  # 1.0
    check_uniform("e2m1", 2, 0x22)  # both nibbles 0x2 == 1.0

    nan_row("e4m3", 0x38, 0x7F)
    nan_row("e4m3", 0x38, 0xFF)
    nan_row("e5m2", 0x3C, 0x7D)

    sa = bytearray([127] * (BM * SK))
    sa[0] = 0xFF
    out = run_dot("e4m3", "e4m3", 1, 1, [0x38] * (BM * BK), [0x38] * (BK * BN),
                  sa=bytes(sa))
    ok = bool(torch.isnan(out[0]).all()) and bool(
        torch.equal(out[1:], torch.full_like(out[1:], float(BK))))
    record("e4m3 E8M0 NaN scale poisons its row", ok, f"row0={out[0, :3].tolist()}")

    cast_case("cast e4m3->bf16: NaN, NaN, 1.0, 448, -448", True,
              [0x7F, 0xFF, 0x38, 0x7E, 0xFE, 0x38, 0x38, 0x38], {0x7F, 0xFF},
              {2: 1.0, 3: 448.0, 4: -448.0})
    cast_case("cast e5m2->bf16: NaN, NaN, +inf, -inf, 57344, -57344", False,
              [0x7D, 0xFD, 0x7C, 0xFC, 0x7B, 0xFB, 0x3C, 0x3C], {0x7D, 0xFD},
              {2: math.inf, 3: -math.inf, 4: 57344.0, 5: -57344.0, 6: 1.0})

    # max finite E4M3 through the scaled upcast's bias multiply
    out = run_dot("e4m3", "e4m3", 1, 1, [0x7E] * (BM * BK), [0x7E] * (BK * BN))
    expect = float(BK) * 448.0 * 448.0
    ok = bool(torch.equal(out, torch.full_like(out, expect)))
    record("e4m3 448x448 uniform -> 64*448^2", ok, f"got {out[0, 0].item()}")

    # Packed FP4 nibble order: the uniform e2m1 case above cannot tell low-first from high-first.
    # A is 1.0 on even K and 2.0 on odd K, and a byte's low nibble is the even-K element. Then
    # 0x12 = (low 1.0, high 0.5) gives 32*1 + 32*1 = 64, and 0x21 = (low 0.5, high 1.0) gives
    # 16 + 64 = 80; a swapped pack order fails both.
    a_mixed = [0x38 if k % 2 == 0 else 0x40 for _ in range(BM) for k in range(BK)]
    for byte, expect in ((0x12, 64.0), (0x21, 80.0)):
        out = run_dot("e4m3", "e2m1", 1, 2, a_mixed, [byte] * ((BK // 2) * BN))
        ok = bool(torch.equal(out, torch.full_like(out, expect)))
        record(f"e2m1 pack order byte 0x{byte:02X} -> {expect:g}", ok,
               f"got {out[0, 0].item()}")

    # MXFP4 packed along M: a byte holds rows (2i, 2i+1) in its low/high nibble, so an all-0x21
    # lhs against an all-1.0 rhs gives 32 on even rows and 64 on odd rows (0x12 swaps them).
    b_ones = torch.full((BK // 2, BN), 0x22, dtype=torch.uint8, device="cuda")
    scales = torch.full((BM, SK), 127, dtype=torch.uint8, device="cuda")
    sb_scales = torch.full((BN, SK), 127, dtype=torch.uint8, device="cuda")
    for byte, even, odd in ((0x21, 32.0, 64.0), (0x12, 64.0, 32.0)):
        a = torch.full((BM // 2, BK), byte, dtype=torch.uint8, device="cuda")
        out = torch.empty((BM, BN), dtype=torch.float32, device="cuda")
        dot_scaled_mpack_k[(1,)](a, b_ones, scales, sb_scales, out,
                                 BM=BM, BN=BN, BK=BK, num_warps=4)
        want = torch.empty_like(out)
        want[0::2] = even
        want[1::2] = odd
        ok = bool(torch.equal(out, want))
        record(f"e2m1 M-packed byte 0x{byte:02X} -> rows {even:g}/{odd:g}", ok,
               f"got {out[0, 0].item()}/{out[1, 0].item()}")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
