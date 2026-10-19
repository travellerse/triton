#!/usr/bin/env python3
"""Runtime conformance for the tl.* surface that the op-class probe does not cover.

`rdna2_feature_compile_probe.py` only shows that these lower; this checks the values on gfx1031:
histogram, sort, argmax, map_elementwise, inline asm, interleave, join/split, a volatile load, and
a device_assert on a condition that is always true (it must stay silent). Each case is one small
program, so a failure names the feature rather than a suite.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag features --budget-mib 1024 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/conformance/rdna2_feature_runtime_probe.py
"""
import sys

import torch
import triton
import triton.language as tl

FAILS = []


def record(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


@triton.jit
def k_hist(x_ptr, o_ptr, BLOCK: tl.constexpr, BINS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BINS), tl.histogram(x, BINS))


@triton.jit
def k_sort(x_ptr, o_ptr, BLOCK: tl.constexpr, DESC: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BLOCK), tl.sort(x, descending=DESC))


@triton.jit
def k_argmax(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr, tl.argmax(x, axis=0))


@triton.jit
def _mapper(x):
    if x > 0:
        return x
    return -x


@triton.jit
def k_map(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BLOCK), tl.map_elementwise(_mapper, x))


@triton.jit
def k_asm(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    y = tl.inline_asm_elementwise("v_add_f32 $0, $1, $1", "=v,v", [x],
                                  dtype=tl.float32, is_pure=True, pack=1)
    tl.store(o_ptr + tl.arange(0, BLOCK), y)


@triton.jit
def k_interleave(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    y = tl.load(y_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, 2 * BLOCK), tl.interleave(x, y))


@triton.jit
def k_join_split(x_ptr, y_ptr, a_ptr, b_ptr, s_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    y = tl.load(y_ptr + tl.arange(0, BLOCK))
    j = tl.join(x, y)
    a, b = tl.split(j)
    tl.store(a_ptr + tl.arange(0, BLOCK), a)
    tl.store(b_ptr + tl.arange(0, BLOCK), b)
    tl.store(s_ptr + tl.arange(0, BLOCK), tl.sum(j, axis=1))


@triton.jit
def k_volatile(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK), volatile=True)
    tl.store(o_ptr + tl.arange(0, BLOCK), x)


@triton.jit
def k_assert(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.device_assert(x > -1.0e30, "unreachable: every input is far above the bound")
    tl.device_assert(tl.sum(x) > -1.0e30, "unreachable: sum of small values")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device")
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).gcnArchName, flush=True)
    torch.manual_seed(7)

    # histogram
    BLOCK, BINS = 512, 16
    x = torch.randint(0, BINS, (BLOCK,), dtype=torch.int32, device="cuda")
    o = torch.zeros(BINS, dtype=torch.int32, device="cuda")
    k_hist[(1,)](x, o, BLOCK=BLOCK, BINS=BINS, num_warps=4)
    torch.cuda.synchronize()
    ref = torch.bincount(x.long(), minlength=BINS).to(torch.int32)
    record("tl.histogram", bool(torch.equal(o, ref)), f"got={o.tolist()} want={ref.tolist()}")

    # sort ascending / descending on distinct values
    BLOCK = 256
    x = torch.randperm(BLOCK, dtype=torch.int32, device="cuda").to(torch.float32)
    for desc in (False, True):
        o = torch.empty_like(x)
        k_sort[(1,)](x, o, BLOCK=BLOCK, DESC=desc, num_warps=4)
        torch.cuda.synchronize()
        ref = torch.sort(x, descending=desc).values
        record(f"tl.sort descending={desc}", bool(torch.equal(o, ref)),
               f"got={o[:4].tolist()} want={ref[:4].tolist()}")

    # argmax
    oi = torch.empty(1, dtype=torch.int32, device="cuda")
    k_argmax[(1,)](x, oi, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    record("tl.argmax", int(oi.item()) == int(torch.argmax(x)), f"got={int(oi.item())}")

    # map_elementwise (branch per element)
    xm = torch.randn(BLOCK, device="cuda")
    om = torch.empty_like(xm)
    k_map[(1,)](xm, om, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    record("tl.map_elementwise", bool(torch.equal(om, xm.abs())), f"max_abs={(om - xm.abs()).abs().max():.3g}")

    # inline asm
    oa = torch.empty_like(xm)
    k_asm[(1,)](xm, oa, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    record("tl.inline_asm_elementwise v_add_f32", bool(torch.equal(oa, xm + xm)),
           f"max_abs={(oa - 2 * xm).abs().max():.3g}")

    # interleave
    BLOCK = 128
    xa = torch.arange(BLOCK, dtype=torch.float32, device="cuda")
    ya = xa + 1000
    oi2 = torch.empty(2 * BLOCK, dtype=torch.float32, device="cuda")
    k_interleave[(1,)](xa, ya, oi2, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    ref = torch.stack([xa, ya], dim=1).flatten()
    record("tl.interleave", bool(torch.equal(oi2, ref)), f"got={oi2[:4].tolist()}")

    # join / split
    A = torch.empty_like(xa)
    B = torch.empty_like(xa)
    S = torch.empty_like(xa)
    k_join_split[(1,)](xa, ya, A, B, S, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    ok = bool(torch.equal(A, xa) and torch.equal(B, ya) and torch.equal(S, xa + ya))
    record("tl.join + tl.split", ok)

    # volatile load
    ov = torch.empty_like(xa)
    k_volatile[(1,)](xa, ov, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    record("tl.load volatile=True", bool(torch.equal(ov, xa)))

    # always-true device asserts must stay silent (an abort would kill this process)
    k_assert[(1,)](torch.zeros(BLOCK, device="cuda"), BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    record("tl.device_assert (true condition stays silent)", True)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
