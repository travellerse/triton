# Fused Triton flash-attention loses on RDNA2 at the production geometry — and it is structural

Status: **REGRESSION** on time, measured; **MEASURED IMPROVEMENT** on peak VRAM and accuracy,
measured; the "cannot be fixed by tuning" part is **CONFIRMED arithmetic** from previously
measured ceilings. Date: 2026-10-19, tags `faprod`/`faprod2`, `state=EXCLUSIVE_COOPERATIVE_LOCK`.
Harness: [rdna2_fa_prod_ab.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_fa_prod_ab.py)
Artifacts: [fa_prod.log](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/fa_prod.log)
(the OOM), [fa_prod2.log](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/fa_prod2.log),
[fa_prod2.json](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/fa_prod2.json).

## Setup

Production geometry, profiled earlier: `heads=32`, `q_tokens=5120`, `kv_tokens=10287`
(prefix-cached text keys, so the key tail needs masking), `head_dim=128`, fp16, `mask=None`.
That is 862.9 GFLOP per attention call, split evenly between the QK^T and the PV product.

Baseline arm is ComfyUI's own shape of the computation: `attention_sub_quad` chunks the query
set and, when the chunk ladder in `comfy/ldm/modules/attention.py:453` can afford it, runs
`bmm(q_chunk, K^T) -> softmax -> bmm(p, V)` with the whole key set per chunk. The harness
reimplements that loop (fp16 softmax, fp32 accumulation inside ATen) rather than importing
ComfyUI, so it is the *algorithm and operand layout* being compared, not the library version.

Triton arms are an online-softmax flash-attention forward step (no rescale-free trick, masked
key tail, fp32 running max/sum/acc), at the tile configs the spill and ILP measurements
identified — including the recommended `32×32 × num_warps=4`.

A/A band **0.08%** (baseline repeated as the last arm: 92.378 vs 92.301 ms).

## Result

| arm | median ms | TFLOPs | max rel err | peak VRAM |
| --- | --- | --- | --- | --- |
| subquad (library), q=2048 | **92.38** | **9.34** | 1.99e-3 | 2901 MiB |
| subquad (library), q=1024 | 93.67 | 9.21 | 1.99e-3 | 1607 MiB |
| Triton FA 32×32, w4 | 182.00 | 4.74 | **4.97e-4** | **313 MiB** |
| Triton FA 64×32, w4 | 224.85 | 3.84 | 4.97e-4 | 313 MiB |
| Triton FA 32×64, w4 | 230.82 | 3.74 | 4.97e-4 | 313 MiB |
| Triton FA 32×64, w2 | 278.94 | 3.09 | 4.97e-4 | 313 MiB |
| Triton FA 32×128, w4 | 286.35 | 3.01 | 4.97e-4 | 313 MiB |

The best fused arm is **1.97× slower** than the unfused library path. It is also **4× more
accurate** against the fp32 reference and needs **9.3× less peak VRAM** (313 vs 2901 MiB).

The Triton numbers are consistent with the standalone measurement in
`rdna2-spill-cost-measured.md` (4.74 vs 4.82 TFLOPs for the same `32×32 × w4` config), so
this is not a fluke of one run; and `32×32` beating every bigger tile here reproduces the
spill-vs-ILP trade-off in the correct direction.

## Why no amount of tuning makes this a win

Both products carry 431.4 GFLOP. Using the **measured** Triton ceilings for this card
(`p4-dot-ceiling-rdna2.md`: QK-shaped 13.77 TFLOPs, PV-shaped 5.96 TFLOPs), a fused kernel
that hits both ceilings exactly would still take

```
431.4/13.77 + 431.4/5.96 = 31.3 ms + 72.4 ms = 103.7 ms  >  92.4 ms measured
```

Flash attention forces *both* contractions through the same engine. On RDNA2 that engine is
the FMA dot path, whose throughput on the PV-shaped contraction (long K, N=128) is ~2.3×
worse than on the QK shape, while rocBLAS runs the same PV product with a tuned GEMM
(the earlier profile put the real workload's PV at 17.16 TFLOPs). So fusing trades a fast
library GEMM for a slow fused one in order to save a softmax pass — and the softmax pass it
saves is worth less than the GEMM it loses. The 24 ms softmax in the production profile is
simply smaller than the ~50 ms the PV product would gain by moving from rocBLAS to FMA.

**Consequence for the RDNA2 plan:** "port flash attention" is not the frontier-algorithm
route on this hardware. The levers that survive the ceiling arithmetic are the ones that keep
the library GEMMs and attack the *bandwidth-bound* steps around them:

* operand layout — already measured as **1.417×** on this exact call
  (`rdna2-key-repack-improvement.md`, 111.0 → 78.5 ms);
* softmax materialization — the upcast copy costs ~5.4 GB of traffic per call; `fp16`,
  `inplace` and `upcast` variants are being measured now (arms `subquad_fp16`,
  `subquad_inplace`, `subquad_upcast`, plus `q=1024` which already showed **45% less peak
  VRAM for +1.4% time**);
* and, where a fused kernel is genuinely useful, its role is **memory**, not speed: 313 MiB
  vs 2901 MiB for the same call is the number that matters on a 12 GiB card that is already
  offloading 4550 MB of weights.

## Evidence from the failed first attempt (`faprod`)

The un-chunked path cannot even be expressed on this card: `F.scaled_dot_product_attention`
at `H=32, QN=5120, KN=10287` tried to allocate **6.28 GiB** and threw
`torch.OutOfMemoryError` ("GPU 0 has a total capacity of 11.98 GiB of which 4.62 GiB is
free") — see `artifacts/fa_prod.log`. ComfyUI chunks because it must. The harness now records
`OOM` as an arm outcome instead of dying mid-measurement.

## Limits of this conclusion

* One fused formulation was tested: online softmax with masked key tail, `num_stages=1`, no
  causal skip, `head_dim=128` only. A better RDNA2 fused kernel could close part of the 90 ms
  gap, but it would have to exceed the *measured* PV-shaped ceiling of 5.96 TFLOPs to matter,
  and that ceiling is a property of the FMA dot lowering plus register pressure, not of this
  kernel's bookkeeping.
* The baseline is a reimplementation of ComfyUI's chunked algorithm inside the harness
  (same operand layout, same fp16 softmax), so it is same-session comparable but not literally
  the production call with its `baddbmm` and mask plumbing.
* No whole-step measurement: attention is ~13.6% of a diffusion step, so these per-call
  numbers do not translate into step time without the geometry gate that is still BLOCKED.
