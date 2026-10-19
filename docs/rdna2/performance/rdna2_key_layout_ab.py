#!/usr/bin/env python3
"""Is the production attention call slow because of the key layout it hands rocBLAS?

Discovered while timing a flash-attention attempt: the same chunked baddbmm/softmax/bmm chain, fed a
key tensor obtained as `k.transpose(1,2)` from an [N, D]-contiguous buffer, ran at 74.5 ms per call,
while the production layout (obtained the way attention_sub_quad builds it, by
`permute(0,2,3,1).reshape(b*h, dim, -1)`, which leaves the *token* axis contiguous) measures
116.10 ms for the identical shapes. Same FLOPs, same library, 1.56x apart -- which also matches the
P1 finding that QK^T ran at 6.55 TFLOPs in production while the same GEMM reached 17.59 in isolation.

This isolates the variable. Each arm builds the key in a different memory order, nothing else
changes, and each arm is profiled so the Tensile kernel actually selected is recorded rather than
assumed. Arms are interleaved with a repeated reference arm for the noise band.

Layouts, all derived from the same [B, N, H*D] source that a projection would produce:
  prod_permute  : permute(0,2,3,1).reshape(...)            -> token axis stride 1 (what ships)
  head_contig   : contiguous [BH, D, N] built explicitly   -> token axis stride 1, packed
  d_contig      : k.transpose(1,2) from [BH, N, D] packed  -> head-dim axis stride 1
Run under the admission wrapper.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/code/triton/python")
import torch

B, H, D = 1, 32, 128
Q_TOK, K_TOK, QCS = 5120, 10287, 1024
FLOPS = 2 * 2 * H * Q_TOK * K_TOK * D


def timed(fn, inner=1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / inner


def clock_ramp(seconds=8.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a) * 1e-4
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def build_keys(k_src, q_src, v_src):
    """k_src: [B, N, H*D] as a projection returns. Returns dict of key_t layouts."""
    # exactly what attention_sub_quad does with skip_reshape=False
    b, n, _ = k_src.shape
    kh = k_src.reshape(b, n, H, D)
    prod_permute = kh.permute(0, 2, 3, 1).reshape(b * H, D, n)
    head_contig = prod_permute.contiguous()
    packed_n_d = kh.permute(0, 2, 1, 3).reshape(b * H, n, D)
    d_contig = packed_n_d.transpose(1, 2)
    q_h = q_src.reshape(b, Q_TOK, H, D).permute(0, 2, 1, 3).reshape(b * H, Q_TOK, D).contiguous()
    v_h = v_src.reshape(b, n, H, D).permute(0, 2, 1, 3).reshape(b * H, n, D).contiguous()
    return {"prod_permute": prod_permute, "head_contig": head_contig,
            "d_contig": d_contig}, q_h, v_h


def make_call(key_t, q_h, v_h, out):
    scale = D ** -0.5
    def call():
        for c0 in range(0, Q_TOK, QCS):
            c1 = min(c0 + QCS, Q_TOK)
            scores = torch.baddbmm(
                torch.empty(H, c1 - c0, K_TOK, device="cuda", dtype=torch.float16),
                q_h[:, c0:c1], key_t, alpha=scale, beta=0)
            torch.softmax(scores, dim=-1, out=scores)
            torch.bmm(scores, v_h, out=out[:, c0:c1])
    return call


def gemm_names(call, inner):
    from torch.profiler import profile, ProfilerActivity
    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(inner):
            call()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        if e.device_time_total > 0:
            rows.append({"kernel": e.key[:70], "count": e.count,
                         "ms": round(e.device_time_total / 1e3 / max(inner, 1), 3)})
    rows.sort(key=lambda r: -r["ms"])
    return rows[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    clock_ramp()
    g = torch.Generator(device="cuda").manual_seed(77)
    q_src = torch.randn(B, Q_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    k_src = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    v_src = torch.randn(B, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    layouts, q_h, v_h = build_keys(k_src, q_src, v_src)
    out = torch.empty(H, Q_TOK, D, device="cuda", dtype=torch.float16)

    rep = {"shapes": {"B": B, "H": H, "D": D, "q_tokens": Q_TOK, "k_tokens": K_TOK,
                      "query_chunk_size": QCS},
           "flops": FLOPS,
           "layouts": {n: {"shape": list(t.shape), "strides": list(t.stride()),
                           "contiguous": t.is_contiguous()}
                       for n, t in layouts.items()},
           "identity": {"device": torch.cuda.get_device_name(0),
                        "gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION")},
           "arms": {}}

    order = [("prod_permute", 0), ("head_contig", 1), ("prod_plus_copy", 2),
             ("prod_permute", 3), ("head_contig", 4), ("prod_plus_copy", 5),
             ("prod_permute", 6), ("prod_plus_copy", 7)]
    calls = {n: make_call(t, q_h, v_h, out) for n, t in layouts.items()}

    # honest arm: the packed layout is materialised INSIDE the timed region, because a
    # comparison that builds it outside silently donates the copy cost to the baseline
    kh = k_src.reshape(B, K_TOK, H, D)
    scale = D ** -0.5

    def call_with_copy():
        packed = kh.permute(0, 2, 3, 1).reshape(B * H, D, K_TOK).contiguous()
        for c0 in range(0, Q_TOK, QCS):
            c1 = min(c0 + QCS, Q_TOK)
            scores = torch.baddbmm(
                torch.empty(H, c1 - c0, K_TOK, device="cuda", dtype=torch.float16),
                q_h[:, c0:c1], packed, alpha=scale, beta=0)
            torch.softmax(scores, dim=-1, out=scores)
            torch.bmm(scores, v_h, out=out[:, c0:c1])
        return packed

    calls["prod_plus_copy"] = call_with_copy
    for name, _ in order:
        call = calls[name]
        for _ in range(args.warmup):
            timed(call)
        s = sorted(timed(call) for _ in range(args.reps))
        med = s[args.reps // 2]
        a = rep["arms"].setdefault(name, {"samples_ms": [], "medians_ms": []})
        a["samples_ms"].append([round(x, 3) for x in s])
        a["medians_ms"].append(round(med, 3))
        print(f"{name:13s} median {med:8.3f} ms  {FLOPS / (med / 1000) / 1e12:6.2f} TFLOPs",
              flush=True)

    for name in list(calls.keys()):
        rep["arms"].setdefault(name, {"samples_ms": [], "medians_ms": []})
    for name in calls.keys():
        rep["arms"][name]["kernels"] = gemm_names(calls[name], 1)
        rep["arms"][name]["tflops_at_median"] = round(
            FLOPS / (max(rep["arms"][name]["medians_ms"]) / 1000) / 1e12, 2)

    # numerical equivalence across layouts: same math, so differences must be rounding-level
    results = {}
    for name, call in calls.items():
        out.zero_()
        try:
            call()
        except Exception as exc:
            rep["arms"][name]["error"] = str(exc)[:120]
            continue
        torch.cuda.synchronize()
        results[name] = out.clone()
    ref = results["prod_permute"]
    for name, r in results.items():
        rep["arms"][name]["max_abs_vs_prod_permute"] = float((r.float() - ref.float()).abs().max())

    json.dump(rep, open(args.out, "w"), indent=1)
    print("\nkernels per layout:")
    for name, a in rep["arms"].items():
        print(f"  {name}: " + " | ".join(f"{k['ms']:.2f}ms x{k['count']} {k['kernel'][:38]}"
                                          for k in a["kernels"][:3]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
