#!/usr/bin/env python3
"""Same-process A/B/A of the tiled key repack inside the REAL production attention call.

Runs ComfyUI's own attention_sub_quad at the steady-state DiT geometry (q=5120, k=10287, 32 heads,
D=128, fp16, mask=None, query_chunk_size=1024, inplace_softmax) and flips only
attention.TRITON_REPACK_KEY between arms, so the two arms share the process, the allocator state,
the clocks and the inputs. The flag is toggled through the module attribute rather than the
environment so no second interpreter start-up or JIT cache miss can leak into the comparison.

Method notes: 8 s clock ramp (the card idles at 500 MHz); the baseline arm is run three times to
measure the A/A band; outputs are compared between arms and against a chunked fp32 reference.
Run with the workload's own interpreter (the one that can import ComfyUI), under admission.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
import torch
import comfy.ldm.modules.attention as A

B, H, D = 1, 32, 128
Q_TOK, K_TOK, QCS = 5120, 10287, 1024
FLOPS = 2 * 2 * H * Q_TOK * K_TOK * D


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    g = torch.Generator(device="cuda").manual_seed(2024)
    q = torch.randn(B, Q_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    v = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)

    def call():
        return A.attention_sub_quad(q, k, v, H, mask=None, query_chunk_size=QCS,
                                    inplace_softmax=True)

    clock_ramp()

    rep = {"shapes": {"B": B, "heads": H, "dim_head": D, "q_tokens": Q_TOK,
                      "k_tokens": K_TOK, "query_chunk_size": QCS, "mask": None},
           "flops": FLOPS, "step_calls": 32, "step_ms_reference": 9616.8,
           "identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                        "triton": getattr(torch, "__version__", None),
                        "production_binding": A.optimized_attention.__name__,
                        "comfy_head": os.popen("git -C /home/trave/tools/ComfyUI rev-parse HEAD")
                        .read().strip()},
           "arms": []}

    outputs = {}
    # interleaved baseline / candidate / baseline / candidate, with a final baseline for drift
    seq = [("off", False), ("on", True), ("off", False), ("on", True), ("off", False)]
    for name, flag in seq:
        A.TRITON_REPACK_KEY = flag
        for _ in range(args.warmup):
            timed(call)
        torch.cuda.reset_peak_memory_stats()
        s = sorted(timed(call) for _ in range(args.reps))
        med = s[args.reps // 2]
        if name == "on" and "on" not in outputs:
            outputs["on"] = call().clone()
        if name == "off" and "off" not in outputs:
            outputs["off"] = call().clone()
        rep["arms"].append({"arm": name, "flag": flag,
                            "samples_ms": [round(x, 3) for x in s],
                            "median_ms": round(med, 3),
                            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                            "repack_stats": dict(A._repack_stats)})
        print(f"{name:4s} median {med:8.3f} ms  {FLOPS / (med / 1000) / 1e12:6.2f} TFLOPs  "
              f"peak {rep['arms'][-1]['peak_allocated_mib']:.0f} MiB", flush=True)

    off = [a for a in rep["arms"] if a["arm"] == "off"]
    on = [a for a in rep["arms"] if a["arm"] == "on"]
    off_med = [a["median_ms"] for a in off]
    on_med = [a["median_ms"] for a in on]
    band = max(off_med) - min(off_med)
    delta = min(on_med) - max(off_med)
    rep["summary"] = {
        "baseline_medians_ms": off_med,
        "candidate_medians_ms": on_med,
        "aa_band_ms": round(band, 3),
        "best_case_delta_ms": round(min(on_med) - max(off_med), 3),
        "median_delta_ms": round(sum(on_med) / len(on_med) - sum(off_med) / len(off_med), 3),
        "speedup": round((sum(off_med) / len(off_med)) / (sum(on_med) / len(on_med)), 4),
        "step_saved_ms": round((sum(off_med) / len(off_med)) - (sum(on_med) / len(on_med)), 3),
        "step_pct": round(((sum(off_med) / len(off_med)) - (sum(on_med) / len(on_med)))
                          * 32 / 9616.8 * 100, 2),
        "output_max_abs_on_vs_off": float((outputs["on"].float()
                                           - outputs["off"].float()).abs().max()),
        "outputs_bitwise_equal": bool(torch.equal(outputs["on"], outputs["off"])),
    }
    A.TRITON_REPACK_KEY = False
    json.dump(rep, open(args.out, "w"), indent=1)
    print(json.dumps(rep["summary"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
