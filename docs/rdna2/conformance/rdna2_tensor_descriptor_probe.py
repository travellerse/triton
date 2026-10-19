#!/usr/bin/env python3
"""Runtime conformance for tl.make_tensor_descriptor on RDNA2's decomposed ABI.

RDNA2 has no TDM, so descriptors take the DECOMPOSED path: loads/stores become ordinary memory
operations with explicit bounds. The unit tests for descriptors are gated on TMA/TDM-capable
targets, so this checks the three behaviours that matter on gfx103x directly:

  * an in-bounds descriptor copy round-trips;
  * a load that runs past the shape pads with the descriptor's padding option;
  * a store that runs past the shape drops the out-of-range elements instead of writing them.

The buffers carry an 8-element right margin and 8 guard rows that are never part of the descriptor
shape, so a store that is not bounds-checked corrupts a checked sentinel.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag desc --budget-mib 1024 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/conformance/rdna2_tensor_descriptor_probe.py
"""
import sys

import torch
import triton
import triton.language as tl

M, N = 40, 48
PITCH = N + 8
GUARD_ROWS = 8
BM = BN = 32
FAILS = []


def record(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


@triton.jit
def desc_kernel(in_ptr, out_ptr, M, N, PITCH,
                BM: tl.constexpr, BN: tl.constexpr,
                OFF_M: tl.constexpr, OFF_N: tl.constexpr, MODE: tl.constexpr):
    src = tl.make_tensor_descriptor(in_ptr, shape=[M, N], strides=[PITCH, 1],
                                    block_shape=[BM, BN])
    dst = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[PITCH, 1],
                                    block_shape=[BM, BN])
    if MODE == 0:      # copy at the given offset
        block = src.load([OFF_M, OFF_N])
        dst.store([OFF_M, OFF_N], block)
    elif MODE == 1:    # write a constant at the given offset
        dst.store([OFF_M, OFF_N], tl.full((BM, BN), 5.0, tl.float32))
    else:              # load at the given offset, store in-bounds for inspection
        block = src.load([OFF_M, OFF_N])
        dst.store([0, 0], block)


def launch(src, dst, off_m, off_n, mode):
    desc_kernel[(1,)](src, dst, M, N, PITCH, BM=BM, BN=BN,
                      OFF_M=off_m, OFF_N=off_n, MODE=mode, num_warps=4)
    torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device")
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).gcnArchName, flush=True)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    in_guard = torch.full((M + GUARD_ROWS, PITCH), -7.0, dtype=torch.float32, device="cuda")
    in_guard[:M, :N] = torch.arange(1, M * N + 1, dtype=torch.float32,
                                    device="cuda").reshape(M, N)

    def fresh_out():
        return torch.full((M + GUARD_ROWS, PITCH), -1.0, dtype=torch.float32, device="cuda")

    def guard_ok(out, expected=-1.0):
        ok = bool((out[M:, :] == expected).all()) and bool((out[:M, N:] == expected).all())
        return ok

    # 1) in-bounds copy
    out = fresh_out()
    launch(in_guard, out, 0, 0, 0)
    record("descriptor copy in-bounds", bool(torch.equal(out[:BM, :BN], in_guard[:BM, :BN]))
           and guard_ok(out))

    # 2) load past the shape pads with zeros; the padded block is stored in-bounds for inspection.
    # Only the 32x32 stored block is checked: columns past BN in `out` are the outer sentinel.
    out = fresh_out()
    launch(in_guard, out, 20, 30, 2)
    rows_valid, cols_valid = M - 20, N - 30        # 20 x 18 real elements
    valid_ref = in_guard[20:M, 30:N]
    valid_got = out[:rows_valid, :cols_valid]
    mism = (valid_got != valid_ref).nonzero()
    ok = bool(mism.numel() == 0)
    ok = ok and bool((out[rows_valid:BM, :BN] == 0).all())
    ok = ok and bool((out[:BM, cols_valid:BN] == 0).all())
    ok = ok and guard_ok(out)
    record("descriptor load past the shape pads zeros", ok,
           f"valid={rows_valid}x{cols_valid} first_mismatch={mism[:3].tolist()}")

    # 3) a store past the shape must drop out-of-range elements, not write the guard
    out = fresh_out()
    launch(in_guard, out, 20, 30, 1)
    ok = bool((out[20:M, 30:N] == 5.0).all())
    untouched = torch.ones_like(out, dtype=torch.bool)
    untouched[20:M, 30:N] = False
    ok = ok and bool((out[untouched] == -1.0).all())
    record("descriptor store past the shape is bounds-checked", ok)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
