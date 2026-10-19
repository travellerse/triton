#!/usr/bin/env python3
"""In-situ A/B/A of the Triton Q5_K dequant through the plugin's own dequantize() entry point.

The standalone kernel benchmark (docs/rdna2/performance/q5k_triton_dequant.py) proved bit-exactness
and speed against dequantize_blocks_Q5_K directly. This checks the thing that actually matters: the
production entry point `dequantize(data, qtype, oshape, dtype)` -- the reshaping, the qtype dispatch
and the returned shape -- with the flag toggled through the module attribute so both arms share one
process, plus peak allocated VRAM, because memory pressure is the reason weight caching was rejected.

Run under the admission wrapper with the workload interpreter.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["GGUF_TRITON_DEQUANT"] = "1"
sys.path.insert(0, "/home/trave/tools/ComfyUI")
sys.path.insert(0, "/home/trave/tools/ComfyUI/custom_nodes")

import gguf
import importlib
import torch

deq = importlib.import_module("ComfyUI-GGUF.dequant")
dtt = importlib.import_module("ComfyUI-GGUF.dequant_triton")

SHAPES = [("gate_up", 24576, 4096), ("down", 4096, 12288), ("q_proj", 4096, 4096)]

# Block geometry per quant type. The fp16 scale field is forced finite: a random bit pattern
# makes d NaN or Inf, and then the comparison is no longer about the decode.
QTYPE = {
    "Q5_K": {"enum": gguf.GGMLQuantizationType.Q5_K, "bytes": 176},
    "Q6_K": {"enum": gguf.GGMLQuantizationType.Q6_K, "bytes": 210},
}


def make_blocks(qtype, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed & 0xFFFF)
    raw = torch.randint(0, 256, (n, QTYPE[qtype]["bytes"]), generator=g, dtype=torch.uint8)
    if qtype == "Q5_K":
        for off in (0, 2):
            bits = torch.randn(n, dtype=torch.float16).view(torch.uint8).reshape(n, 2)
            raw[:, off:off + 2] = bits
    else:  # Q6_K keeps d in the last two bytes: normal exponent, random mantissa
        raw[:, -1] = (raw[:, -1] & 0x03) | 0x38
    return raw.cuda()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


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
    assert torch.cuda.is_available()
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtype", default="Q5_K", choices=sorted(QTYPE))
    ap.add_argument("--out", default="/tmp/q21-p1/insitu.json")
    args = ap.parse_args()
    qenum = QTYPE[args.qtype]["enum"]
    clock_ramp()
    rep = {"identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "qtype": args.qtype,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                        "plugin_head": os.popen("git -C /home/trave/tools/ComfyUI/custom_nodes/"
                                                "ComfyUI-GGUF rev-parse --short HEAD").read().strip()},
           "arms": {}}

    for name, out_f, in_f in SHAPES:
        n = (out_f * in_f) // 256
        raw = make_blocks(args.qtype, n, hash((name, out_f)))

        for label, flag in (("off", False), ("on", True), ("off", False), ("on", True), ("off", False)):
            dtt.TRITON_DEQUANT = flag
            fn = lambda: deq.dequantize(raw, qenum, (out_f, in_f), dtype=torch.float16)
            for _ in range(2):
                timed(fn)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            s = sorted(timed(fn) for _ in range(7))
            med = s[3]
            a = rep["arms"].setdefault(f"{name}_{label}", {"medians_ms": [], "peak_mib": []})
            a["medians_ms"].append(round(med, 3))
            a["peak_mib"].append(round(torch.cuda.max_memory_allocated() / 2**20, 1))

        dtt.TRITON_DEQUANT = False
        ref = deq.dequantize(raw, qenum, (out_f, in_f), dtype=torch.float16)
        dtt.TRITON_DEQUANT = True
        got = deq.dequantize(raw, qenum, (out_f, in_f), dtype=torch.float16)
        a_off = rep["arms"][f"{name}_off"]["medians_ms"]
        a_on = rep["arms"][f"{name}_on"]["medians_ms"]
        rep["arms"][name] = {
            "shape": [out_f, in_f],
            "off_best_ms": min(a_off), "on_best_ms": min(a_on),
            "off_all": a_off, "on_all": a_on,
            "aa_band_ms": round(max(a_off) - min(a_off), 3),
            "speedup": round(min(a_off) / min(a_on), 2),
            "peak_off_mib": max(rep["arms"][f"{name}_off"]["peak_mib"]),
            "peak_on_mib": max(rep["arms"][f"{name}_on"]["peak_mib"]),
            "shape_matches": tuple(got.shape) == tuple(ref.shape),
            "bitwise_equal": bool(torch.equal(ref, got)),
            "dtype_matches": got.dtype == ref.dtype,
            "stats": dict(dtt.stats),
        }
        for k in (f"{name}_off", f"{name}_on"):
            rep["arms"].pop(k, None)
        r = rep["arms"][name]
        print(f"{name:8s} off {r['off_best_ms']:8.3f} ms -> on {r['on_best_ms']:8.3f} ms "
              f"x{r['speedup']:.2f} | A/A {r['aa_band_ms']:.3f} ms | peak {r['peak_off_mib']:.0f}"
              f" -> {r['peak_on_mib']:.0f} MiB | bitwise={r['bitwise_equal']} "
              f"shape={r['shape_matches']} stats={r['stats']}", flush=True)
        del raw, ref, got
        torch.cuda.empty_cache()

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    bad = [n for n, r in rep["arms"].items() if not (r["bitwise_equal"] and r["shape_matches"])]
    print("integrated-path exactness failures:", bad or "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
