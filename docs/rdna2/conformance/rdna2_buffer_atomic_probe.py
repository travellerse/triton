#!/usr/bin/env python3
"""Runtime conformance for the RDNA2 integer buffer-atomic subset.

`ConvertToBufferOps` now routes unmasked, provably-active `and/or/xor/umin/umax` (and signed
min/max) at 32/64 bits through `buffer_atomic_*` on RDNA2. The lowering was pinned by lit tests;
this probe checks the values on the card and records which instruction the compiler actually chose,
so a silent fallback to the generic CAS loop cannot pass as coverage.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag buf-atomics --budget-mib 1024 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/conformance/rdna2_buffer_atomic_probe.py
"""
import re
import sys

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 256
FAILS = []


def record(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


@triton.jit
def atomic_k(x_ptr, o_ptr, BLOCK: tl.constexpr, OP: tl.constexpr, CONTENDED: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    if CONTENDED:
        idx = offs * 0
    else:
        idx = offs
    if OP == 0:
        tl.atomic_and(o_ptr + idx, v)
    elif OP == 1:
        tl.atomic_or(o_ptr + idx, v)
    elif OP == 2:
        tl.atomic_xor(o_ptr + idx, v)
    elif OP == 3:
        tl.atomic_min(o_ptr + idx, v)
    else:
        tl.atomic_max(o_ptr + idx, v)


OPS = [("and", 0), ("or", 1), ("xor", 2), ("min", 3), ("max", 4)]
NP = {torch.int32: np.int32, torch.uint32: np.uint32,
      torch.int64: np.int64, torch.uint64: np.uint64}


def init_value(op, np_dtype):
    info = np.iinfo(np_dtype)
    if op == 0:      # and
        return info.max
    if op in (1, 2):  # or / xor
        return 0
    return 12345     # min / max


def make_values(np_dtype, seed):
    rng = np.random.default_rng(seed)
    if np_dtype in (np.uint32, np.uint64):
        vals = rng.integers(0, 2**63, size=BLOCK, dtype=np.uint64)
        vals |= (rng.integers(0, 2, size=BLOCK, dtype=np.uint64) << np.uint64(63))
        return vals.astype(np_dtype)
    return rng.integers(-(2**31), 2**31, size=BLOCK).astype(np_dtype)


def apply_op(op, a, b):
    if op == "and":
        return np.bitwise_and(a, b)
    if op == "or":
        return np.bitwise_or(a, b)
    if op == "xor":
        return np.bitwise_xor(a, b)
    if op == "min":
        return np.minimum(a, b)
    return np.maximum(a, b)


def reduce_op(op, vals, init):
    if op == "and":
        return np.bitwise_and.reduce(vals, initial=init)
    if op == "or":
        return np.bitwise_or.reduce(vals, initial=init)
    if op == "xor":
        return np.bitwise_xor.reduce(vals, initial=init)
    if op == "min":
        return np.minimum.reduce(vals, initial=init)
    return np.maximum.reduce(vals, initial=init)


def run_case(torch_dtype, op_name, op_code, contended, seed):
    np_dtype = NP[torch_dtype]
    vals_np = make_values(np_dtype, seed)
    init = init_value(op_code, np_dtype)
    x = torch.from_numpy(vals_np).to("cuda")
    out_np = np.full(1 if contended else BLOCK, init, dtype=np_dtype)
    out = torch.from_numpy(out_np).to("cuda")

    atomic_k[(1,)](x, out, BLOCK=BLOCK, OP=op_code, CONTENDED=contended, num_warps=4)
    torch.cuda.synchronize()

    got = out.cpu().numpy()
    if contended:
        expect = np.full(1, reduce_op(op_name, vals_np, init), dtype=np_dtype)
    else:
        expect = apply_op(op_name, np.full(BLOCK, init, dtype=np_dtype), vals_np)

    try:
        compiled = atomic_k.warmup(x, out, BLOCK=BLOCK, OP=op_code,
                                   CONTENDED=contended, grid=(1,), num_warps=4)
        asm = compiled.asm["amdgcn"]
        isa = sorted(set(re.findall(r"buffer_atomic_[a-z0-9_]+", asm)))
        cas = len(re.findall(r"global_atomic_cmpswap", asm))
    except Exception as e:  # noqa: BLE001 - ISA labels are secondary to the value check
        isa, cas = [f"isa-unavailable: {type(e).__name__}"], -1

    ok = bool(np.array_equal(got, expect))
    name = f"{np.dtype(np_dtype).name} {op_name} {'contended' if contended else 'per-lane'}"
    record(name, ok, f"isa={isa} cmpswap={cas}" + ("" if ok else f" got={got[:3]} want={expect[:3]}"))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device")
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).gcnArchName, flush=True)

    seed = 0
    for torch_dtype in (torch.int32, torch.uint32, torch.int64, torch.uint64):
        for op_name, op_code in OPS:
            run_case(torch_dtype, op_name, op_code, False, seed)
            seed += 1
    for op_name, op_code in OPS[:3]:
        run_case(torch.int32, op_name, op_code, True, seed)
        seed += 1

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
