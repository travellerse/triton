#!/usr/bin/env python3
"""Which operand layout do the Qwen-Image-2.1 GEMMs want, and can the dequant produce it for free?

The key-repack result (docs/rdna2/performance/rdna2-key-repack-improvement.md) says a strided
operand can cost rocBLAS a factor of 2.3 on this card. The same question has never been asked about
the model's other big GEMMs, which Phase 50 measured at 9.4-12.4 TFLOPs while the identical
attention GEMM reaches 17.5 in isolation.

Shapes are derived from the model definition, not guessed: dim = 32 heads x 128 = 4096,
mlp_ratio = 3 -> hidden 12288 and gate_up output 24576, and the steady-state block runs on the 5120
target tokens (the text prefix is cached, model.py:403).

For each shape every candidate operand arrangement is timed twice:
  inside  - the reorganisation is part of the measured call, i.e. what it would really cost per call
  assumed - the packed operand is built once outside, i.e. what it would cost if the producer (the
            GGUF dequant, which already writes a fresh fp16 buffer every step) wrote that layout
            directly and the repack were free
That distinction is the whole decision: a layout only worth having if paid for once, per weight.

Arms are interleaved with the production arrangement repeated for the noise band. Run under
the admission wrapper with the workload interpreter.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
import torch

DIM, HIDDEN, TOKENS = 4096, 12288, 5120
SHAPES = [
    ("proj_qkv_out", TOKENS, DIM, DIM),        # M, K, N  (x @ W.t(), W row-major [N,K])
    ("mlp_gate_up", TOKENS, DIM, 2 * HIDDEN),
    ("mlp_down", TOKENS, HIDDEN, DIM),
]


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


def arrangements(x, w):
    """All ways to compute y = x @ W.t() that differ only in operand memory order."""
    N, K = w.shape
    wt_strided = w.t()                       # [K, N], strides (1, K) -- what F.linear hands over
    wt_packed = w.t().contiguous()           # [K, N] packed
    return {
        "linear_stridedB": lambda: torch.matmul(x, wt_strided),
        "linear_packedB_inside": lambda: torch.matmul(x, w.t().contiguous()),
        "transposed_view": lambda: torch.matmul(w, x.t()).t(),
        "transposed_materialised": lambda: torch.matmul(w, x.t()).t().contiguous(),
    }, {"wt_strided": wt_strided, "wt_packed": wt_packed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    clock_ramp()
    rep = {"identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION")},
           "shapes": [{"name": n, "M": M, "K": K, "N": Nn} for n, M, K, Nn in SHAPES],
           "results": {}}

    for name, M, K, N in SHAPES:
        g = torch.Generator(device="cuda").manual_seed(5 + M)
        x = torch.randn(M, K, generator=g, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, generator=g, device="cuda", dtype=torch.float16)
        funcs, views = arrangements(x, w)
        flops = 2.0 * M * K * N
        out = {"flops": flops}

        # correctness: every arrangement must compute the same thing
        ref = torch.matmul(x.float(), w.float().t())
        for fname, fn in funcs.items():
            y = fn()
            torch.cuda.synchronize()
            err = float((y.float() - ref).abs().max())
            out.setdefault("max_abs_vs_fp32", {})[fname] = err

        # the "packed operand is free" arm uses the prebuilt view
        funcs_all = dict(funcs)
        funcs_all["linear_packedB_free"] = lambda: torch.matmul(x, views["wt_packed"])
        order = ["linear_stridedB", "linear_packedB_free", "linear_packedB_inside",
                 "transposed_view", "transposed_materialised",
                 "linear_stridedB", "linear_packedB_free", "linear_stridedB"]
        for fname in order:
            fn = funcs_all[fname]
            for _ in range(args.warmup):
                timed(fn)
            s = sorted(timed(fn) for _ in range(args.reps))
            med = s[args.reps // 2]
            a = out.setdefault(fname, {"medians_ms": []})
            a["medians_ms"].append(round(med, 3))

        base = max(out["linear_stridedB"]["medians_ms"])
        for fname, a in out.items():
            if not isinstance(a, dict) or "medians_ms" not in a:
                continue
            med = max(a["medians_ms"])
            a["median_ms"] = round(med, 3)
            a["tflops"] = round(flops / (med / 1000) / 1e12, 2)
            a["vs_stridedB_ms"] = round(base - med, 3)
            print(f"{name:13s} {fname:24s} {med:8.3f} ms  {a['tflops']:6.2f} TFLOPs  "
                  f"gain {a['vs_stridedB_ms']:+7.3f} ms", flush=True)

        # which Tensile kernel each arrangement selects
        from torch.profiler import profile, ProfilerActivity
        for fname, fn in funcs_all.items():
            fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            rows = sorted(({"kernel": e.key[:58], "ms": round(e.device_time_total / 1e3, 3)}
                           for e in prof.key_averages() if e.device_time_total > 0),
                          key=lambda r: -r["ms"])
            out[fname]["kernels"] = rows[:2]

        rep["results"][name] = out
        # step-level arithmetic using Phase 50's call counts
        calls = {"proj_qkv_out": 4 * 32, "mlp_gate_up": 32, "mlp_down": 32}[name]
        for fname, a in out.items():
            if isinstance(a, dict) and "median_ms" in a:
                a["step_ms_at_calls"] = round(a["median_ms"] * calls, 1)
        del x, w, views, funcs_all
        torch.cuda.empty_cache()

    json.dump(rep, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
