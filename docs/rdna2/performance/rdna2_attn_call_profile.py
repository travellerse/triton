#!/usr/bin/env python3
"""Kernel-level breakdown of the production attention call after the key repack.

The repacked call measures 90.75 ms while its three big kernels account for only ~71 ms
(QK^T 24.0 + PV 23.1 + softmax 24.0), so ~19.7 ms per call -- 630 ms per step -- is somewhere
else. This attributes it: every device kernel the real attention_sub_quad launches, with its
time, for the flag off and flag on arms.

Run under admission with the workload interpreter.
"""
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
import torch
import comfy.ldm.modules.attention as A
from torch.profiler import profile, ProfilerActivity

B, H, D = 1, 32, 128
Q_TOK, K_TOK, QCS = 5120, 10287, 1024


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
    g = torch.Generator(device="cuda").manual_seed(9)
    q = torch.randn(B, Q_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    v = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)

    def call():
        return A.attention_sub_quad(q, k, v, H, mask=None, query_chunk_size=QCS,
                                    inplace_softmax=True)

    clock_ramp()
    out = {"shapes": {"q_tokens": Q_TOK, "k_tokens": K_TOK, "heads": H, "D": D},
           "identity": {"gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION")},
           "arms": {}}
    for name, flag in (("flag_off", False), ("flag_on", True)):
        A.TRITON_REPACK_KEY = flag
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000 / 5
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        rows = sorted(({"kernel": e.key[:74], "count": e.count,
                        "ms": round(e.device_time_total / 1e3, 3)}
                       for e in prof.key_averages() if e.device_time_total > 0),
                      key=lambda r: -r["ms"])
        total_dev = sum(r["ms"] for r in rows)
        out["arms"][name] = {"wall_ms_per_call": round(wall, 3),
                             "device_ms_sum": round(total_dev, 3),
                             "unattributed_ms": round(wall - total_dev, 3),
                             "kernels": rows[:14],
                             "repack_stats": dict(A._repack_stats)}
        print(f"=== {name}: wall {wall:.3f} ms/call, device sum {total_dev:.3f}, "
              f"unattributed {wall - total_dev:.3f}")
        for r in rows[:12]:
            print(f"   {r['ms']:8.3f} ms x{r['count']:<3d} {r['kernel'][:74]}")
    A.TRITON_REPACK_KEY = False
    path = "/tmp/q21-p1/attn_profile.json"
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
