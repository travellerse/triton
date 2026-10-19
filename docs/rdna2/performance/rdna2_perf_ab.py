#!/usr/bin/env python3
"""RDNA2 codegen performance A/B: measure what the restored family actually buys.

The gfx103x restoration turns on two codegen levers that were never timed:
  * the vec-128 LDS load/store tile (getSharedLdStTiles), used by cross-layout
    round trips through shared memory;
  * isRDNA() now being true, which routes cross-lane shuffles through the DPP
    butterfly instead of ds_bpermute.

Method is the one validated in the P1 study, because this card idles at 500 MHz:
clock ramp before any timing, one deterministic input draw shared by every arm,
sync-bracketed wall ms, median of reps, arms interleaved, and an A/A repetition of
the baseline arm so the noise band is measured rather than assumed. Results are
labelled by the emitted target, not by the physical part.

Run under the admission wrapper, once per build variant:
  env PYTHONPATH=/home/trave/code/triton/python python3 docs/rdna2/performance/rdna2_perf_ab.py --out /tmp/x.json
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/code/triton/python")
import torch
import triton
import triton.language as tl


def driver_mib():
    try:
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                           capture_output=True, text=True, timeout=15)
        for ln in r.stdout.splitlines():
            if ln.startswith("card0"):
                return int(ln.split(",")[2].strip()) / 2**20
    except Exception:
        pass
    return -1.0


def clock_ramp(seconds=8.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a) * 1e-4
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- kernels
@triton.jit
def k_trans(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    # round trip through shared memory with a 128-bit-wide operand
    tl.store(o_ptr + rn[:, None] * M + rm[None, :], tl.trans(v))


@triton.jit
def k_rms(x_ptr, o_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
    var = tl.sum(v * v, axis=0) / n_cols
    tl.store(o_ptr + row * n_cols + cols, v * tl.rsqrt(var + 1e-6), mask=cols < n_cols)


@triton.jit
def k_softmax(x_ptr, o_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
    m = tl.max(v, axis=0)
    e = tl.exp2((v - m) * 1.4426950408889634)
    s = tl.sum(e, axis=0)
    tl.store(o_ptr + row * n_cols + cols, e / s, mask=cols < n_cols)


@triton.jit
def k_dot(a_ptr, b_ptr, o_ptr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
        acc += tl.dot(a, b).to(tl.float32)
    tl.store(o_ptr + rm[:, None] * BN + rn[None, :], acc.to(tl.float16))


WORKLOADS = [
    # label, launch, grid, kwargs for tensor creation
    ("trans_1024x1024_f32", lambda x, o: k_trans[(8, 8)](x, o, M=1024, N=1024, BM=128, BN=128, num_warps=4)),
    ("trans_2048x512_f32", lambda x, o: k_trans[(16, 4)](x, o, M=2048, N=512, BM=128, BN=128, num_warps=4)),
    ("rms_5120x1024_f16", lambda x, o: k_rms[(5120,)](x, o, n_cols=1024, BLOCK=1024, num_warps=4)),
    ("rms_5120x4096_f16", lambda x, o: k_rms[(5120,)](x, o, n_cols=4096, BLOCK=4096, num_warps=8)),
    ("softmax_5120x1024_f16", lambda x, o: k_softmax[(5120,)](x, o, n_cols=1024, BLOCK=1024, num_warps=4)),
    ("dot_64x64x1024_f16_1tile", lambda x, o: k_dot[(1,)](x, o, o, K=1024, BM=64, BN=64, BK=64, num_warps=4)),
]


def make_pair(label):
    g = torch.Generator(device="cuda").manual_seed(4242)
    if label.startswith("trans_1024"):
        x = torch.randn(1024, 1024, generator=g, device="cuda", dtype=torch.float32)
        return x, torch.empty(1024, 1024, device="cuda", dtype=torch.float32)
    if label.startswith("trans_2048"):
        x = torch.randn(2048, 512, generator=g, device="cuda", dtype=torch.float32)
        return x, torch.empty(512, 2048, device="cuda", dtype=torch.float32)
    if label.startswith("rms_5120x1024"):
        x = torch.randn(5120, 1024, generator=g, device="cuda", dtype=torch.float16)
        return x, torch.empty_like(x)
    if label.startswith("rms_5120x4096"):
        x = torch.randn(5120, 4096, generator=g, device="cuda", dtype=torch.float16)
        return x, torch.empty_like(x)
    if label.startswith("softmax"):
        x = torch.randn(5120, 1024, generator=g, device="cuda", dtype=torch.float16)
        return x, torch.empty_like(x)
    if label.startswith("dot_"):
        x = torch.randn(1024, 1024, generator=g, device="cuda", dtype=torch.float16)
        return x, torch.empty(1024, 1024, device="cuda", dtype=torch.float16)
    raise SystemExit(label)


def timed(fn, inner=1):
    # a single small kernel is launch-latency bound and its A/A band swallows any real
    # codegen difference, so each sample brackets a batch of launches and reports per launch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / inner


def resource_info():
    """Register/scratch usage reported by the compiled artifacts, if this build exposes it.

    Scratch (private segment) bytes above zero is how a spill shows up on AMD, which is
    the quantity the fast-attention work has to drive down.
    """
    info = {}
    for jit in (k_trans, k_rms, k_softmax, k_dot):
        caches = []
        for attr in ("device_caches", "cache"):
            holder = getattr(jit, attr, None)
            if isinstance(holder, dict):
                for value in holder.values():
                    inner = value[0] if isinstance(value, tuple) and value else value
                    if isinstance(inner, dict):
                        caches.append(inner)
                if attr == "device_caches":
                    break
        for cache in caches:
            for key, kern in cache.items():
                md = getattr(kern, "metadata", None)
                d = md._asdict() if hasattr(md, "_asdict") else (md if isinstance(md, dict) else {})
                picked = {k: v for k, v in d.items()
                          if any(t in k.lower() for t in ("spill", "register", "scratch",
                                                           "private", "shared", "local"))}
                if picked:
                    info[f"{getattr(jit.fn, '__name__', 'jit')}::{key}"] = picked
    return info


def measure(label, launch, reps, warmup, inner):
    x, o = make_pair(label)
    for _ in range(warmup):
        timed(lambda: launch(x, o), inner)
    return sorted(timed(lambda: launch(x, o), inner) for _ in range(reps))[reps // 2], x, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--variant", default="head")
    ap.add_argument("--inner", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    clock_ramp()
    props = torch.cuda.get_device_properties(0)
    rep = {
        "variant": args.variant,
        "identity": {
            "physical_device": props.name,
            "gcnArchName": props.gcnArchName,
            "warp_size": props.warp_size,
            "multi_processor_count": props.multi_processor_count,
            "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
            "triton_version": triton.__version__,
            "triton_package": os.path.dirname(triton.__file__),
            "torch": torch.__version__,
        },
        "reps": args.reps,
        "inner_launches": args.inner,
        "warmup": args.warmup,
        "results": {},
    }

    for label, launch in WORKLOADS:
        # A/A first: same arm twice to measure drift, then the arm again
        a1, _, _ = measure(label, launch, args.reps, args.warmup, args.inner)
        a2, _, _ = measure(label, launch, args.reps, args.warmup, args.inner)
        a3, x, o = measure(label, launch, args.reps, args.warmup, args.inner)
        rep["results"][label] = {
            "samples_medians_ms": [round(a1, 3), round(a2, 3), round(a3, 3)],
            "median_ms": round(a3, 3),
            "aa_spread_ms": round(max(a1, a2, a3) - min(a1, a2, a3), 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }
        del x, o
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"{label:26s} medians={rep['results'][label]['samples_medians_ms']} "
              f"spread={rep['results'][label]['aa_spread_ms']:.3f} ms", flush=True)

    rep["resource_info"] = resource_info()
    json.dump(rep, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
