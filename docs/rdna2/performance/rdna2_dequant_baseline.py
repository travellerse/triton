#!/usr/bin/env python3
"""How far is the GGUF Q5_K torch dequant from the card's bandwidth ceiling?

OPT-036 measured dequantisation at ~943 ms/step but never compared it to what the memory system can
do. This times the plugin's own dequantize_blocks_Q5_K at real production weight shapes and puts two
reference points beside it: a bare fp16 write of the same output size, and a same-size copy. If the
dequant is near those, it is already bandwidth-bound and a kernel rewrite is pointless; if it is far
above, the gap is the prize and its size is known before any code is written.

Run under admission with the workload interpreter.
"""
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
sys.path.insert(0, "/home/trave/tools/ComfyUI/custom_nodes/ComfyUI-GGUF")
import torch
from dequant import dequantize_blocks_Q5_K, QK_K

# production DiT weight shapes: (out, in) per Linear
SHAPES = [("gate_up", 24576, 4096), ("down", 4096, 12288), ("q_proj", 4096, 4096)]
BYTES_PER_BLOCK = 176      # Q5_K: d 2 + dmin 2 + scales 12 + qh 32 + qs 128


def clock_ramp(seconds=8.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a) * 1e-4
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main():
    clock_ramp()
    rep = {"identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION")},
           "shapes": [], "step_estimate": {}}
    total_ms = 0.0
    # per step: 32 blocks x (gate_up + down + q + k + v + out) projections
    CALLS_PER_STEP = {"gate_up": 32, "down": 32, "q_proj": 32 * 4}
    for name, out_f, in_f in SHAPES:
        params = out_f * in_f
        n_blocks = params // QK_K
        blocks = torch.randint(0, 256, (n_blocks, BYTES_PER_BLOCK),
                               device="cuda", dtype=torch.uint8)
        out_bytes = params * 2
        in_bytes = n_blocks * BYTES_PER_BLOCK

        ref = dequantize_blocks_Q5_K(blocks, QK_K, BYTES_PER_BLOCK, dtype=torch.float16)
        torch.cuda.synchronize()

        def deq():
            return dequantize_blocks_Q5_K(blocks, QK_K, BYTES_PER_BLOCK, dtype=torch.float16)

        dst = torch.empty(params, device="cuda", dtype=torch.float16)
        src = torch.empty(params, device="cuda", dtype=torch.float16)

        for _ in range(3):
            deq()
        deq_ms = min(timed(deq) for _ in range(7))
        copy_ms = min(timed(lambda: dst.copy_(src)) for _ in range(7))
        write_ms = min(timed(lambda: dst.fill_(1.0)) for _ in range(7))

        gbps_deq = (in_bytes + out_bytes) / (deq_ms / 1000) / 1e9
        gbps_copy = (in_bytes + out_bytes) / (copy_ms / 1000) / 1e9
        gbps_write = out_bytes / (write_ms / 1000) / 1e9
        rep["shapes"].append({
            "name": name, "shape": [out_f, in_f], "params": params, "blocks": n_blocks,
            "dequant_ms": round(deq_ms, 3), "copy_ms": round(copy_ms, 3),
            "fill_ms": round(write_ms, 3),
            "dequant_GBps": round(gbps_deq, 1), "copy_GBps": round(gbps_copy, 1),
            "fill_GBps": round(gbps_write, 1),
            "gap_vs_copy_x": round(deq_ms / copy_ms, 2),
            "max_abs_vs_itself": float((deq().float() - ref.float()).abs().max()),
        })
        print(f"{name:8s} {out_f}x{in_f}: dequant {deq_ms:8.3f} ms ({gbps_deq:6.1f} GB/s) | "
              f"copy {copy_ms:6.3f} ms ({gbps_copy:6.1f} GB/s) | fill {write_ms:6.3f} ms "
              f"({gbps_write:6.1f} GB/s) | gap x{deq_ms / copy_ms:.2f}", flush=True)
        step_ms = deq_ms * CALLS_PER_STEP[name]
        total_ms += step_ms
        rep["step_estimate"][name] = round(step_ms, 1)
        del blocks, ref, dst, src
        torch.cuda.empty_cache()

    rep["step_estimate_total_ms"] = round(total_ms, 1)
    rep["step_reference_ms"] = 9616.8
    rep["step_pct"] = round(total_ms / 9616.8 * 100, 2)
    json.dump(rep, open("/tmp/q21-p1/dequant_baseline.json", "w"), indent=1)
    print(f"\ndequant share of step (this model only): {total_ms:.1f} ms "
          f"= {rep['step_pct']}%  wrote /tmp/q21-p1/dequant_baseline.json")


if __name__ == "__main__":
    main()
