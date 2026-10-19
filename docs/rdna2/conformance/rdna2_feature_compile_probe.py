"""Static probe: does the rest of the tl.* surface compile for RDNA2?

The op-class probe (`rdna2_opclass_probe.py`) measures numerical behaviour on the card and the
dot_scaled probe covers the MX feature that used to abort. Neither answers the broader
compatibility question: are there language features whose lowering paths never grew a gfx103x arm
and therefore fail (or silently degrade) at compile time? This probe compiles one small kernel per
feature against an explicit `GPUTarget`, so it runs with no device and no launches.

The runtime values of these features (histogram, sort, argmax, map_elementwise, inline asm,
interleave, join/split, a volatile load, always-true device_asserts) are checked separately by
`rdna2_feature_runtime_probe.py`; all of them pass on native gfx1031.

It also records two kernel-option behaviours that are *not* compile failures:
  * num_ctas > 1 is rejected up front with "num_ctas > 1 not supported on gfx1030" -- hardware has
    no clusters, and the rejection is actionable.
  * maxnreg is silently dropped by HIP parse_options (it is a CUDAOptions field). LLVM AMDGPU has
    no per-function VGPR cap to map it onto (only amdgpu-waves-per-eu, an occupancy target, which
    the backend already exposes as the `waves_per_eu` option), so inventing a mapping would be a
    policy guess. The probe prints DROPPED for it so the drop is at least visible in the record.

Run it against any Triton build; label results by the emitted target.
  PYTHONPATH=/home/trave/code/triton/python python3 docs/rdna2/conformance/rdna2_feature_compile_probe.py
"""
import argparse
import traceback

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def k_descriptor(a_ptr, o_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    ia = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[BM, BN])
    blk = ia.load([0, 0])
    io = tl.make_tensor_descriptor(o_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[BM, BN])
    io.store([0, 0], blk)


@triton.jit
def k_histogram(x_ptr, o_ptr, BLOCK: tl.constexpr, BINS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BINS), tl.histogram(x, BINS))


@triton.jit
def k_sort(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BLOCK), tl.sort(x))


@triton.jit
def _abs_if_neg(x):
    if x > 0:
        return x
    return -x


@triton.jit
def k_map(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BLOCK), tl.map_elementwise(_abs_if_neg, x))


@triton.jit
def k_inline_asm(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    y = tl.inline_asm_elementwise("v_add_f32 $0, $1, $1", "=v,v", [x],
                                  dtype=tl.float32, is_pure=True, pack=1)
    tl.store(o_ptr + tl.arange(0, BLOCK), y)


@triton.jit
def k_device_print_assert(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.device_assert(x > 0, "must be positive")
    tl.device_print("x0 ", tl.sum(x))


@triton.jit
def k_argmax(x_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr, tl.argmax(x, axis=0))


@triton.jit
def k_interleave(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    y = tl.load(y_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, 2 * BLOCK), tl.interleave(x, y))


# name, kernel, signature, constexprs, option overrides, expected outcome
CASES = [
    ("descriptor", k_descriptor,
     {"a_ptr": "*fp16", "o_ptr": "*fp16", "M": "i32", "N": "i32"},
     {"BM": 32, "BN": 32}, {}, "COMPILES"),
    ("histogram", k_histogram, {"x_ptr": "*i32", "o_ptr": "*i32"},
     {"BLOCK": 256, "BINS": 16}, {}, "COMPILES"),
    ("sort", k_sort, {"x_ptr": "*fp32", "o_ptr": "*fp32"}, {"BLOCK": 256}, {},
     "COMPILES"),
    ("map_elementwise", k_map, {"x_ptr": "*fp32", "o_ptr": "*fp32"}, {"BLOCK": 256},
     {}, "COMPILES"),
    ("inline_asm", k_inline_asm, {"x_ptr": "*fp32", "o_ptr": "*fp32"},
     {"BLOCK": 256}, {}, "COMPILES"),
    ("device_print_assert", k_device_print_assert, {"x_ptr": "*fp32"},
     {"BLOCK": 256}, {}, "COMPILES"),
    ("argmax", k_argmax, {"x_ptr": "*fp32", "o_ptr": "*i32"}, {"BLOCK": 256}, {},
     "COMPILES"),
    ("interleave", k_interleave,
     {"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32"}, {"BLOCK": 128}, {},
     "COMPILES"),
    ("num_ctas2", k_histogram, {"x_ptr": "*i32", "o_ptr": "*i32"},
     {"BLOCK": 256, "BINS": 16}, {"num_ctas": 2}, "REJECTED: num_ctas"),
    ("maxnreg64", k_sort, {"x_ptr": "*fp32", "o_ptr": "*fp32"}, {"BLOCK": 256},
     {"maxnreg": 64}, "DROPPED"),
]


def run(arch):
    print(f"\n===== {arch} =====")
    failures = 0
    for name, fn, sig, consts, overrides, expected in CASES:
        options = {"num_warps": 4, "num_stages": 1, **overrides}
        try:
            cc = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts),
                                target=GPUTarget("hip", arch, 32), options=options)
        except Exception as e:  # noqa: BLE001 - the message is the finding
            msg = " ".join(str(e).strip().split())
            if expected.startswith("REJECTED") and expected.split(": ")[1] in msg:
                print(f"  {name:20} {expected} ({msg[:90]})")
            else:
                print(f"  {name:20} FAILS: {msg[-240:]}")
                failures += 1
            continue
        asm = cc.asm["amdgcn"]
        note = ""
        if expected == "DROPPED":
            # The option is filtered out before HIPOptions is built, so identical code is the
            # observable consequence of the drop.
            note = " (silently ignored; see module docstring)"
        elif expected != "COMPILES":
            print(f"  {name:20} UNEXPECTED SUCCESS (expected {expected})")
            failures += 1
            continue
        print(f"  {name:20} COMPILES instrs={len(asm.splitlines())}{note}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030,gfx1031,gfx1100,gfx942")
    args = ap.parse_args()
    rc = 0
    for arch in args.arch.split(","):
        try:
            rc |= run(arch)
        except Exception:  # noqa: BLE001 - a probe crash must name the case it died on
            traceback.print_exc()
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
