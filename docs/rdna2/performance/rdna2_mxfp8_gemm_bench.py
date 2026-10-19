"""Benchmark: e4m3 MXFP8 GEMM (scaled dot, FMA path), 64x64x128 tiles.

Run through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag mx-gemm --budget-mib 2048 \
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
      PYTHONPATH=/home/trave/code/triton/python \
      python3 docs/rdna2/performance/rdna2_mxfp8_gemm_bench.py 10 4096 8
"""
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

M = N = K = 4096
BM = BN = 64
BK = 128
SK = BK // 32


@triton.jit
def gemm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, K, N,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    sk: tl.constexpr = BK // 32
    offs_sk = tl.arange(0, sk)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        sa = tl.load(sa_ptr + offs_m[:, None] * (K // 32) + (k // 32 + offs_sk)[None, :])
        sb = tl.load(sb_ptr + offs_n[:, None] * (K // 32) + (k // 32 + offs_sk)[None, :])
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    global M, N, K
    if len(sys.argv) > 2:
        M = N = K = int(sys.argv[2])
    warps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    torch.manual_seed(0)
    a = torch.randint(0, 120, (M, K), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 120, (K, N), dtype=torch.uint8, device="cuda")
    sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
    sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
    c = torch.empty((M, N), dtype=torch.float32, device="cuda")
    grid = (M // BM, N // BN)

    def launch():
        gemm[grid](a, b, sa, sb, c, K, N, BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=2)

    for _ in range(8):          # compile + clock ramp
        launch()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        launch()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(times)
    tf = 2 * M * N * K / (med * 1e-3) / 1e12
    print(f"gfx target: {torch.cuda.get_device_properties(0).gcnArchName} size={M} warps={warps} "
          f"median={med:.3f} ms  TFLOPs={tf:.2f}  all={[round(t, 2) for t in times]}")


if __name__ == "__main__":
    main()
