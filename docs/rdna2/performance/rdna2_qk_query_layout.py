#!/usr/bin/env python3
"""Is the in-call QK^T slow because the QUERY operand is strided?

With batch size 1 (the production case) every q/k/v "reshape" in attention_sub_quad is a view, not a
copy: merging the size-1 batch dim is free, and the head merge satisfies PyTorch's view condition.
So the operand handed to baddbmm is the projection output read as [heads, tokens, dim] with strides
(dim, heads*dim, 1) -- head-interleaved -- while the probe that measured 23-24 ms used a contiguous
[heads, tokens, dim] tensor with strides (tokens*dim, dim, 1).

Same FLOPs, same key, same chunking; only the A operand's memory order differs. If this is the gap,
making the query contiguous costs one 42 MB copy (~0.3 ms) and recovers the rest.

Arms are interleaved with the strided arm repeated for the noise band. Run under admission.
"""
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, "/home/trave/code/triton/python")
import torch

H, D = 32, 128
Q_TOK, K_TOK, QCS = 5120, 10287, 1024
SCALE = D ** -0.5
FLOPS_QK = 2.0 * H * Q_TOK * K_TOK * D


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
    g = torch.Generator(device="cuda").manual_seed(23)
    # exactly what the DiT hands attention_sub_quad: [B, N, H*D] fp16
    q_src = torch.randn(1, Q_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)
    k_src = torch.randn(1, K_TOK, H * D, generator=g, device="cuda", dtype=torch.float16)

    q_strided = q_src.reshape(1, Q_TOK, H, D).permute(0, 2, 1, 3).reshape(H, Q_TOK, D)
    q_contig = q_strided.contiguous()
    key_packed = k_src.reshape(1, K_TOK, H, D).permute(0, 2, 3, 1).reshape(H, D, K_TOK).contiguous()
    key_strided = k_src.reshape(1, K_TOK, H, D).permute(0, 2, 3, 1).reshape(H, D, K_TOK)

    print("query strides   strided:", q_strided.stride(), " contiguous:", q_contig.stride(),
          "| strided is view:", q_strided.data_ptr() == q_src.data_ptr())
    print("key   strides strided:", key_strided.stride(), "   packed:", key_packed.stride())

    def gemms(q, key):
        def run():
            for c0 in range(0, Q_TOK, QCS):
                buf = torch.empty(H, QCS, K_TOK, device="cuda", dtype=torch.float16)
                torch.baddbmm(buf, q[:, c0:c0 + QCS], key, alpha=SCALE, beta=0)
        return run

    def gemms_with_query_copy(q):
        def run():
            qc = q.contiguous()
            for c0 in range(0, Q_TOK, QCS):
                buf = torch.empty(H, QCS, K_TOK, device="cuda", dtype=torch.float16)
                torch.baddbmm(buf, qc[:, c0:c0 + QCS], key_packed, alpha=SCALE, beta=0)
        return run

    arms = {
        "q_strided_key_strided": gemms(q_strided, key_strided),      # production today
        "q_strided_key_packed": gemms(q_strided, key_packed),        # repack shipped so far
        "q_contig_key_packed": gemms(q_contig, key_packed),         # probe's arrangement
        "q_contig_key_strided": gemms(q_contig, key_strided),
        "q_strided_plus_copy_key_packed": gemms_with_query_copy(q_strided),
    }

    order = ["q_strided_key_packed", "q_contig_key_packed", "q_strided_key_strided",
             "q_contig_key_strided", "q_strided_plus_copy_key_packed",
             "q_strided_key_packed", "q_contig_key_packed", "q_strided_key_packed"]
    rep = {"shapes": {"H": H, "D": D, "q_tokens": Q_TOK, "k_tokens": K_TOK, "chunk": QCS},
           "strides": {"q_strided": list(q_strided.stride()), "q_contig": list(q_contig.stride()),
                        "key_strided": list(key_strided.stride()),
                        "key_packed": list(key_packed.stride())},
           "identity": {"gcn": torch.cuda.get_device_properties(0).gcnArchName,
                        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                        "alloc_conf": os.environ.get("PYTORCH_HIP_ALLOC_CONF")},
           "arms": {}}
    for name in order:
        fn = arms[name]
        for _ in range(3):
            timed(fn)
        s = sorted(timed(fn) for _ in range(9))
        med = s[9 // 2]
        a = rep["arms"].setdefault(name, {"medians_ms": []})
        a["medians_ms"].append(round(med, 3))
        print(f"{name:33s} median {med:8.3f} ms  {FLOPS_QK / (med / 1000) / 1e12:6.2f} TFLOPs",
              flush=True)

    for name, a in rep["arms"].items():
        a["best_ms"] = min(a["medians_ms"])
        a["tflops"] = round(FLOPS_QK / (a["best_ms"] / 1000) / 1e12, 2)
    ref = rep["arms"]["q_strided_key_packed"]["best_ms"]
    for name, a in rep["arms"].items():
        a["gain_vs_shipped_ms"] = round(ref - a["best_ms"], 3)
        a["step_gain_ms_x32"] = round((ref - a["best_ms"]) * 32, 1)
    band = max(rep["arms"]["q_strided_key_packed"]["medians_ms"]) - \
        min(rep["arms"]["q_strided_key_packed"]["medians_ms"])
    rep["aa_band_ms"] = round(band, 3)
    json.dump(rep, open("/tmp/q21-p1/qk_query_layout.json", "w"), indent=1)
    print("A/A band:", round(band, 3), "ms")
    print("wrote /tmp/q21-p1/qk_query_layout.json")


if __name__ == "__main__":
    main()
