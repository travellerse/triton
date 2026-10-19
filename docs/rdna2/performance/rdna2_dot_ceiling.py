#!/usr/bin/env python3
"""What is the achievable ceiling for tl.dot on RDNA2, and when does it spill?

The workload's fused attention reaches 5.12 TFLOPs while rocBLAS reaches 17.59 on the same
shapes, and RDNA2 has no WMMA/MFMA to fall back on, so everything goes through the FMA dot
path. Before changing any kernel this measures the ceiling of that path directly: if Triton
can reach ~15 TFLOPs on square GEMM, the attention kernel is structured wrong; if it caps far
below that, the lowering itself is the problem and the fix belongs in the compiler.

Spills are read from the compiled artifact's global_scratch_size, which is how scratch shows
up on AMD. Square shapes measure the peak; the two skinny shapes mirror the attention inner
loops (QK^T is wide-output/short-K, PV is narrow-output/long-K).

Run under the admission wrapper.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/code/triton/python")
import torch
import triton
import triton.language as tl


@triton.jit
def gemm(a_ptr, b_ptr, c_ptr, M, N, K,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP * num_pid_n
    gid = pid // num_in_group
    first_pid_m = gid * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


SHAPES = [
    ("sq_1024", 1024, 1024, 1024),
    ("sq_2048", 2048, 2048, 2048),
    ("qk_like", 1024, 8192, 128),     # wide output, short K
    ("pv_like", 1024, 128, 8192),     # narrow output, long K
]

CONFIGS = [
    (32, 32, 32, 1, 1),
    (64, 32, 32, 2, 1),
    (64, 64, 32, 4, 1),
    (64, 64, 64, 4, 1),
    (128, 64, 32, 4, 1),
    (128, 128, 32, 4, 1),
    (128, 128, 64, 8, 1),
    (128, 128, 64, 4, 1),
    (64, 128, 64, 4, 1),
]


def driver_free_mib():
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


def timed(fn, inner):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / inner


def scratch_of(jit):
    sizes = []
    holder = getattr(jit, "device_caches", None) or getattr(jit, "cache", None)
    if not isinstance(holder, dict):
        return sizes
    for value in holder.values():
        inner = value[0] if isinstance(value, tuple) and value else value
        if not isinstance(inner, dict):
            continue
        for kern in inner.values():
            md = getattr(kern, "metadata", None)
            d = md._asdict() if hasattr(md, "_asdict") else (md if isinstance(md, dict) else {})
            sizes.append({"scratch": d.get("global_scratch_size"),
                          "shared": d.get("shared"),
                          "warps": d.get("num_warps")})
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    clock_ramp()
    props = torch.cuda.get_device_properties(0)
    report = {"identity": {"device": props.name, "gcn": props.gcnArchName,
                           "warp_size": props.warp_size, "mpc": props.multi_processor_count,
                           "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                           "triton": triton.__version__,
                           "triton_pkg": os.path.dirname(triton.__file__)},
              "rows": []}

    for name, M, N, K in SHAPES:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)
        ref = (a.float() @ b.float()).half()
        flops = 2.0 * M * N * K
        for (BM, BN, BK, W, G) in CONFIGS:
            if BM > M or BN > N:
                continue
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            launch = lambda: gemm[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, GROUP=G, num_warps=W)
            try:
                c.zero_()
                launch()
                torch.cuda.synchronize()
                err = float((c.float() - ref.float()).abs().max())
                if err >= 1.0:
                    report["rows"].append({"shape": name, "cfg": [BM, BN, BK, W],
                                           "wrong_result": err})
                    print(f"{name:9s} BM{BM:4d} BN{BN:4d} BK{BK:3d} w{W}  WRONG max_abs_err={err:.4f}")
                    continue
                med = min(timed(launch, args.inner) for _ in range(args.reps))
            except Exception as exc:
                report["rows"].append({"shape": name, "cfg": [BM, BN, BK, W],
                                       "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
                continue
            tf = flops / (med / 1000) / 1e12
            report["rows"].append({"shape": name, "cfg": [BM, BN, BK, W], "ms": round(med, 4),
                                   "tflops": round(tf, 2), "max_abs_err": err})
            print(f"{name:9s} BM{BM:4d} BN{BN:4d} BK{BK:3d} w{W}  {med:8.4f} ms  "
                  f"{tf:6.2f} TFLOPs  err={err:.3f}", flush=True)
        report.setdefault("scratch", {})[name] = scratch_of(gemm)
        del a, b, c, ref
        torch.cuda.empty_cache()

    best = {}
    for r in report["rows"]:
        if "tflops" not in r:
            continue
        if r["shape"] not in best or r["tflops"] > best[r["shape"]]["tflops"]:
            best[r["shape"]] = r
    report["best"] = best
    report["vram_used_mib"] = driver_free_mib()
    json.dump(report, open(args.out, "w"), indent=1)
    print("\nbest per shape:")
    for s, r in best.items():
        print(f"  {s:9s} {r['tflops']:6.2f} TFLOPs @ BM{r['cfg'][0]} BN{r['cfg'][1]} "
              f"BK{r['cfg'][2]} w{r['cfg'][3]}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
