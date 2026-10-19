# P1 — Workload-first performance optimization: broadcast-head-mask candidate

Date: 2026-10-14. Host: RX 6700 XT (gfx1031 silicon), all code emitted for **gfx1030** under the
pre-existing `HSA_OVERRIDE_GFX_VERSION=10.3.0`. torch `2.13.0+rocm7.2`, hip `7.2.53211`.
This phase is a performance experiment. It is not a continuation of the M1 target-support audit.

## 1. The actual workload, and the M1 premise that turned out to be false

Workload: the ComfyUI image-edit pipeline **Qwen-Image-2.1 Edit** (`Qwen-Image-2.1-Turbo-AD-Q5_K.gguf`
DiT, `qwen3vl_8b_int8_convrot` TE, TAESD preview), production geometry **1024×1280 → 5120 image
tokens, k_tokens=10287**, 8 steps, `--use-quad-cross-attention --enable-triton-backend --fp16-unet
--cpu-vae`. Tree: `/home/trave/tools/ComfyUI`, branch `self`, HEAD `aceb31c8b`, launcher
`evidence/harness/launch.sh`.

Steady-state baseline from Phase 50 (2026-10-13, grade CONFIRMED): **9616.8 ms/step**, of which the
attention bucket is **3678.9 ms = 38.3%** (32 calls × 114.97 ms).

The M1M premise — "this workload needs `f32 atomic_add`" — **does not survive contact with the
workload**. `grep` over `/home/trave/tools/ComfyUI/comfy/**.py` and
`comfy-kitchen/comfy_kitchen/**.py` finds **zero `tl.atomic_add`** on the active path; the production
attention is `baddbmm`/`softmax`/`bmm` through rocBLAS, and the (disabled) Triton fused kernel is
`tl.dot`/`tl.load`/`tl.store` only. The claim originates in `m1-implementation-report.md:1014-1016`
(op-class hypothesis) and `:3316-3319`, not in any measurement in `evidence/`. **No buffer-atomic
work was pursued this phase**, and the broad `supportsBufferAtomicRMW()` gate was not touched.

## 2. Bottleneck selected, and the evidence behind it

Selected: the **additive block-causal mask path** in `attention_sub_quad`.

Evidence at selection time (all static): `comfy/ldm/modules/attention.py:390` expands a 2-D
`(q, k)` mask to `(b*heads, q, k)`; for the text segment that is `32 × 5167 × 5167` fp16 =
**1.7087 GB** per call; the consumer is `attn_scores += mask`
(`sub_quadratic_attention.py:177`). First-hand shapes: `reports/43:59` (5167 + 5120 = 10287),
`reports/42:190`. Expected mechanism: delete a score-matrix-sized device copy per masked call.

**Both halves of that premise were later falsified by measurement (§5, §6). Recording them as
false is this phase's result.**

## 3. Baseline recorded before tuning

| item | value |
|---|---|
| source revision | triton `b575c0c38d` (branch `work/rdna2-gfx1031-m0`, M1 working-tree changes intact and untouched) |
| workload tree | ComfyUI `self` @ `aceb31c8b`, clean before edit |
| target | physical gfx1031 / **codegen gfx1030** via `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| measured workload | `attention_sub_quad` at production shapes, out-of-process |
| warm-up | 8 s clock ramp (the card idles at 500 MHz) + 2 warm-up reps per arm |
| timing | `torch.cuda.synchronize()`-bracketed `perf_counter`, median of 7 reps, arms interleaved |
| baseline arm result | text arm pooled median **203.30 ms**, sd 1.95 ms, n=21 (3 A arms) |
| correctness gate | NaN-aware bit-identity (see §6) |
| host RAM | `MemAvailable` 17842 MiB at admission, 13189 MiB during the run |
| VRAM at admission | `vram_used_mib` 1271–1330 of 12272 |

Out-of-process validity: the image/mask=None arm measured **116.10–116.83 ms** against the in-run
production figure of **114.97 ms** (`reports/50:95-97` recorded the same 2.6% agreement). The call
being measured is the one the production step actually runs.

## 4. The candidate

One surgical, env-gated, **default-off** change (full diff: `candidate-broadcast-head-mask.patch`).
Inside `attention_sub_quad` only, a shared 2-D mask is passed as a `[1, q, k]` broadcast view
instead of being head-expanded. Nothing else was modified: no target-support predicates, no admission
utility, no workload configuration, no other kernel.

```python
if BROADCAST_HEAD_MASK and bs == 1 and len(mask.shape) == 2:
    mask = mask.unsqueeze(0)
else:
    mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(...)
```

## 5. Did the intended code path execute? Yes — and it showed the mechanism was imaginary

The mask as observed entering `efficient_dot_product_attention`:

| arm | shape | `is_contiguous()` | bytes |
|---|---|---|---|
| baseline | `[32, 5167, 5167]` | **false** | 53 395 778 of storage, 1.7087 GB logical |
| candidate | `[1, 5167, 5167]` | true | 53 395 778 |

`.expand(...).reshape(...)` on a size-1 leading dim **folds back into a stride-0 view instead of
copying**. There was never a 1.71 GB device copy to remove. Two independent confirmations:

* peak allocator memory is **bit-identical** across every arm: `1082.63330078125 MiB` in all three
  baseline and both candidate arms.
* the per-call kernel list is unchanged, including its cost (`artifacts/broadcast_mask_ab.json`):

| kernel | baseline | candidate |
|---|---|---|
| `Cijk_Alik_Bljk_HHS_...MT128x64x8` (QKᵀ) | 150.36 ms ×5 | 149.42 ms ×5 |
| `Cijk_Ailk_Bljk_HHS_...MT64x32x8` (PV) | 24.58 ms ×5 | 24.64 ms ×5 |
| `cunn_SoftMaxForwardFast` | 14.50 ms ×6 | 14.36 ms ×6 |
| `elementwise_kernel_manual_unroll` (the mask add) | 9.68 ms ×6 | 10.01 ms ×6 |
| any copy kernel for the mask | **absent** | absent |

A second, independent error in the selection: **the masked call is not in the steady-state step at
all.** `model.py:403/418` — once the prefix cache is warm, steps 2..8 run `prefix_cached_attention`
(target rows only, `mask=None`) and the block-causal prefix pass is skipped. So even a real 1.71 GB
copy would have had a ceiling of a few calls in step 1, not 64 per step. The attention bucket of
Phase 50 (32 × 114.97 ms) is the 32 unmasked target calls, which this run reproduced at 116.1 ms.

## 6. Correctness gates

`artifacts/equiv.log`, CPU-only equivalence test, **run under admission** because
`model_management` requires a device at import (`wrapper_rc=0`, `state=EXCLUSIVE_COOPERATIVE_LOCK`):
**19/19 PASS**, including an A/A control on every case. Covered: multi-chunk with irregular tail,
`q == k` causal mask, `b=2`, a 3-D per-batch mask (must keep the materialising path), the no-mask
path, the single-chunk fast path, a singleton-query-row mask, a finite (non-0/-inf) additive bias,
and NaN position on a fully-masked row.

Production-shape check on device: `{"bitwise_equal": true, "max_abs_diff": 0.0, "nan_count": 0}`.
Semantics (ordering, masked positions, NaN propagation) are preserved exactly. Two apparent failures
during bring-up were defects in my test (a NaN-vs-NaN comparison, and an invalid 17×17 mask against
13 keys), not in the candidate — recorded because the first one nearly produced a false regression.

## 7. Performance result and noise analysis

Pre-registered rule: an improvement counts only if it exceeds the same-session A/A band. It did not
approach it.

| arm | median ms |
|---|---|
| text/A1 | 203.70 |
| text/A2 | 204.33 |
| text/A3 | 202.53 |
| text/B1 (candidate) | 203.66 |
| text/B2 (candidate) | 202.92 |

* **A/A drift band = 1.80 ms (0.88%)** on medians; pooled sd 1.95 ms.
* pooled baseline 203.30 ms (n=21) vs candidate 203.55 ms (n=14) → **Δ = +0.25 ms (+0.12%)**,
  i.e. nominally *slower*, at **0.45 × se(Δ)** — indistinguishable from zero.
* specificity control (mask=None, code path never touched by the candidate): A1 116.83 / A2 116.79 /
  B1 116.68 → A/A spread 0.04 ms, Δ −0.15 ms. As expected, nothing.

Every sample is retained in `artifacts/broadcast_mask_ab.json`; none was discarded, and no sample was
invalidated by contamination (fault list empty in all six admission records).

The four claims, separated: code executed — **yes**; expected code generated — **n/a** (host-side
change, no codegen involved); measured call improved — **no**; complete workload improved —
**not measured** (see §9).

## 8. GPU admission evidence

All runs went through `docs/rdna2/admission/gpu_admission.py` with `--acquire` (the fail-open hole M1M
fixed). Every acquisition has a matching release in `/tmp/qwen21-vram/gpu_lock_history.log`:

| tag | state | budget MiB | target_rc |
|---|---|---|---|
| p1-preflight | IDLE_OBSERVED_NO_EXCLUSIVITY (informational, no lock) | — | 0 |
| p1-cpu-equiv / -3 / -4 | EXCLUSIVE_COOPERATIVE_LOCK | 1024 | 1 / 1 / 1 (my test bugs) |
| p1-cpu-equiv5 | EXCLUSIVE_COOPERATIVE_LOCK | 1024 | 0 |
| p1-bcast-ab | EXCLUSIVE_COOPERATIVE_LOCK | 4096 | 0 |
| p1-hotprofile | EXCLUSIVE_COOPERATIVE_LOCK | 3072 | 0 |

Lock ownership was verified by re-`flock` plus `/proc/<pid>/fd` on each acquisition; the two
`gpulock.sh` copies were byte-identical (`7882dd64…`); no KFD PID, no port listener, no lock fd
holder, and no `gfxhub`/amdgpu fault line at any probe. One warning: a stale `p1-cpu-equiv5` record
whose PID was gone with no fd holder — the documented normal residue, correctly not treated as a
holder. Per contract §12 this is cooperative exclusion only.

Peak VRAM actually observed: 2965 MiB (masked-call arms), 3923 MiB (10287-key control), 1550 MiB
allocator peak for the hot call — all inside the declared budgets. A **full production run stays
inadmissible**: its honest peak of 10.8–11.6 GiB cannot satisfy `used + budget + 1200 ≤ 12272`, so
end-to-end timing was not attempted rather than faked.

## 9. Decision: `NO MATERIAL IMPROVEMENT`

The candidate is correct, inert, and has been **reverted** — ComfyUI is back to a clean
`self` @ `aceb31c8b`. The complexity is not retained, because neither the measured call nor the
measured step changed at all.

What this phase established with measurements rather than reading:

1. The M1 `f32`-atomic premise has **no workload existence** (zero `tl.atomic_add`).
2. The head-expanded mask is **already a stride-0 view**; there is no copy to eliminate.
3. The masked block-causal call is **not in steps 2..8** at all (prefix cache), so it could never
   have moved the steady-state step.
4. The production steady-state attention call is now decomposed per kernel (new measurement,
   `artifacts/hot_call_profile.json`, 116.10 ms median, 7.43 TFLOPs aggregate):

   | component | ms/call | achieved rate | ×32 calls |
   |---|---|---|---|
   | `QKᵀ` `Cijk_Alik_Bljk_MT128x256x16` | **65.99** | **6.55 TFLOPs** | 2112 ms |
   | `PV` `Cijk_Ailk_Bljk_MT128x256x16` | 25.17 | 17.16 TFLOPs | 805 ms |
   | `cunn_SoftMaxForwardFast` | 24.38 | ~276 GB/s (bandwidth-saturated) | 780 ms |
   | cat + misc | 0.54 | — | 17 ms |

## 10. Next best candidate, from these findings

`QKᵀ` and `PV` in the same call have **identical FLOPs** (432 GFLOP each) and run at **6.55 vs
17.16 TFLOPs**. The QKᵀ operand layout is the measured gap, and the recoverable amount is
**40.8 ms/call × 32 = 1.31 s/step ≈ 13.6% of the 9616.8 ms step** — an order of magnitude larger
than anything this phase's candidate could have touched, and softmax is already at hardware
bandwidth so it is not the target.

One specific caveat must be checked before that direction is called new: `contiguous_key` is applied
only inside the `skip_reshape=True` branch (`attention.py:342`), while the DiT path takes the
`skip_reshape=False` branch (`:336-341`) where the key becomes a strided `(bh, dim, tokens)` view.
That means Phase 42's rejection of "`contiguous_key` recovers attention GEMM efficiency" (ledger
falsification #15) and `reports/OPT-031`'s "negative benefit at k=10287" may both have measured an
**inert switch** on this geometry. The next phase should first assert, from inside the real call,
whether the flag ever changed the operand it was meant to change — and only then run a single
contiguous-key A/B on the per-block call. That is one narrowly scoped experiment, not a reopened
capability audit.

## Artifacts

`candidate-broadcast-head-mask.patch` · `artifacts/broadcast_mask_ab.json` ·
`artifacts/hot_call_profile.json` · `artifacts/equiv.log` · `artifacts/ab.log` · `artifacts/hot.log`.
Harness (new, additive, nothing overwritten): `/home/trave/code/rdna2-attention/evidence/harness/
broadcast_mask/{bench_broadcast_mask.py,test_broadcast_mask_equivalence.py,profile_hot_call.py}`.
Admission JSON records: `/tmp/rdna2-admission/audit/*-p1-*.json`.
