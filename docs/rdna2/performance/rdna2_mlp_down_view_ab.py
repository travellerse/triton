#!/usr/bin/env python3
"""Should the MLP down projection also return a view, like gate_up already does?

Q21-OPT-035 made the gate_up product a view but left the down projection materialising
(linear_input_act -> linear_transposed with the default materialise=True). Measured GEMM-only at
the production shape, the materialised form costs 36.74 ms against 30.60 ms for the view, but a
strided output is paid for again by whoever reads it -- here the gated residual add. So the
comparison has to be the PAIR, not the GEMM alone.

Shapes from the model: dim 4096, hidden 12288, 5120 steady-state target tokens.

Run under admission with the workload interpreter.
"""
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/tools/ComfyUI")
import torch

DIM, HIDDEN, TOKENS = 4096, 12288, 5120
FLOPS = 2.0 * TOKENS * HIDDEN * DIM


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
    g = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(TOKENS, HIDDEN, generator=g, device="cuda", dtype=torch.float16)
    w = torch.randn(DIM, HIDDEN, generator=g, device="cuda", dtype=torch.float16)
    resid = torch.randn(TOKENS, DIM, generator=g, device="cuda", dtype=torch.float16)
    gate = torch.randn(1, 1, DIM, generator=g, device="cuda", dtype=torch.float16)

    def down_view():
        return torch.matmul(w, x.t()).t()

    def down_materialised():
        return torch.matmul(w, x.t()).t().contiguous()

    def pair_view():
        y = down_view()
        return resid + y * gate

    def pair_materialised():
        y = down_materialised()
        return resid + y * gate

    def pair_forward_form():
        # the orientation this module deliberately avoids, kept as a reference point
        return resid + torch.matmul(x, w.t()) * gate

    arms = {"view_then_add": pair_view, "materialised_then_add": pair_materialised,
            "forward_form_then_add": pair_forward_form,
            "view_only": down_view, "materialised_only": down_materialised}

    rep = {"shapes": {"tokens": TOKENS, "hidden": HIDDEN, "dim": DIM},
           "flops": FLOPS,
           "identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                        "alloc_conf": os.environ.get("PYTORCH_HIP_ALLOC_CONF")},
           "arms": {}}
    order = ["materialised_then_add", "view_then_add", "forward_form_then_add",
             "materialised_only", "view_only", "materialised_then_add", "view_then_add",
             "materialised_then_add"]
    for name in order:
        fn = arms[name]
        for _ in range(3):
            timed(fn)
        s = sorted(timed(fn) for _ in range(9))
        med = s[9 // 2]
        a = rep["arms"].setdefault(name, {"medians_ms": []})
        a["medians_ms"].append(round(med, 3))
        print(f"{name:24s} median {med:8.3f} ms  {FLOPS / (med / 1000) / 1e12:6.2f} TFLOPs",
              flush=True)

    for name, a in rep["arms"].items():
        a["best_ms"] = min(a["medians_ms"])
    base = rep["arms"]["materialised_then_add"]["best_ms"]
    for name, a in rep["arms"].items():
        a["gain_vs_materialised_pair_ms"] = round(base - a["best_ms"], 3)
        a["step_gain_ms_x32"] = round((base - a["best_ms"]) * 32, 1)
    band = max(rep["arms"]["materialised_then_add"]["medians_ms"]) - \
        min(rep["arms"]["materialised_then_add"]["medians_ms"])
    rep["aa_band_ms"] = round(band, 3)

    # the two pair arms must agree numerically; a view only changes memory order
    rep["correctness"] = {
        "pair_view_vs_materialised_max_abs": float((pair_view().float()
                                                    - pair_materialised().float()).abs().max()),
        "view_is_noncontiguous": bool(not down_view().is_contiguous()),
        "materialised_is_contiguous": bool(down_materialised().is_contiguous()),
    }
    json.dump(rep, open("/tmp/q21-p1/mlp_down_view.json", "w"), indent=1)
    print("A/A band:", round(band, 3), "ms |", json.dumps(rep["correctness"]))
    print("wrote /tmp/q21-p1/mlp_down_view.json")


if __name__ == "__main__":
    main()
