# RDNA2 GEMM operand-layout map, and what is left inside the attention call

Date 2026-10-19. All runs admitted (`state=EXCLUSIVE_COOPERATIVE_LOCK`), gfx1030 emitted target,
`docs/rdna2/performance/rdna2_mlp_layout_ab.py` and `rdna2_attn_call_profile.py`.

## 1. The model's other GEMMs are already on their best arrangement

Sweeping operand memory order over the three big GEMM shapes (dim 4096 = 32×128, hidden 12288,
5120 target tokens, steady state):

| shape | F.linear (strided B) | packed B, paid per call | packed B, if produced free | transposed view | transposed + materialised |
|---|---|---|---|---|---|
| projections 5120×4096×4096 | 9.857 ms / 17.43 | 16.286 / 10.55 | **9.174 / 18.73** | 9.490 / 18.10 | 15.981 / 10.75 |
| gate_up 5120×4096×24576 | 93.202 / 11.06 | 131.448 / 7.84 | **54.153 / 19.03** | 56.003 / 18.41 | 121.382 / 8.49 |
| down 5120×12288×4096 | 165.072 / **3.12** | 61.391 / 8.40 | 39.027 / 13.21 | **30.599 / 16.84** | 36.743 / 14.03 |

Production (`SwiGLUFeedForward`, model.py:63-74) already uses the transposed-view form for both MLP
matrices, which is the best or second-best arm in every row. This independently reproduces the
comment's own measurement (2.97 vs 13.4 TFLOPs for the down projection; here 3.12 vs 13.21), so
**that lever is harvested, not open** — and `F.linear` remains 5.4× worse on the down projection.

Two residual facts worth keeping: paying for the repack *inside* the call is a loss on every shape
(gate_up 93.2 → 131.4 ms), so any packed-weight scheme only pays if the producer writes that layout;
and the projections do have ~7% headroom (9.857 → 9.174 ms, ≈87 ms/step) if the GGUF dequant could
emit packed weights for free — untested, and it collides with the 12 GiB budget that already killed
weight caching (OPT-036).

## 2. What the 20% "residual" bucket actually is

Phase 50 left `q/k/v/out projections + norm/adaln/rope` as one 1937.5 ms/step bucket. The measured
projection GEMM alone is 9.857 ms × 128 calls/step = **1261 ms**, so ≈676 ms/step is the
non-GEMM part (adaln, rms/rope, the fused comfy-kitchen kernels). The bucket is no longer undifferentiated.

## 3. Inside the repacked attention call, kernel by kernel

| | flag off | flag on |
|---|---|---|
| wall | 110.99 ms | 90.65 ms |
| QK^T `Alik_Bljk_MT128x256x16` | 63.367 ms ×5 | — |
| QK^T `Ailk_Bljk_MT128x128x8` | — | 38.823 ms ×5 |
| softmax `cunn_SoftMaxForwardFast` | 23.785 ms ×5 | 23.950 ms ×5 |
| PV `Ailk_Bljk_MT128x256x16` | 23.060 ms ×5 | 23.499 ms ×5 |
| `_repack.kd` | — | 3.514 ms |
| cat + elementwise | 0.51 ms | 0.52 ms |
| unattributed (host gaps) | 0.271 ms | 0.344 ms |

The arithmetic closes exactly: −24.54 (QK^T) + 3.51 (repack) + 0.44 (PV) = −20.34 ms ≈ the measured
20.3 ms, and device time accounts for 99.7% of wall, so there is no host-side overhead left to chase.

Softmax at 23.95 ms for 5 chunks of [32,1024,10287] fp16 is 6.75 GB of traffic = **~282 GB/s**, i.e.
bandwidth-saturated: it cannot be improved without fusing it away.

## 4. The next measured target, and an open discrepancy I am not hiding

The isolated layout study measured the packed `MT128x128x8` QK^T at 24.01 ms; **in the real call the
same tile costs 38.82 ms**. I do not yet have an explanation, so I am not claiming the difference as
available headroom. Candidates to test: the `alpha=scale` epilogue, the per-chunk `torch.empty`
allocation of a 675 MB buffer, and L2/Infinity-Cache state when the packed key is written by the
repack immediately before being read. If any of those is the cause, up to ~15 ms/call (~480 ms/step)
is addressable without touching the algorithm; that is the next experiment, and it must be measured,
not assumed.
