# Q6_K Triton dequant: bit-exact, 7.2–14.1× in situ, ~249 ms/step of the Qwen-Image DiT

Status: **MEASURED IMPROVEMENT** (integrated path, admitted runs), **CONFIRMED** bit-exactness.
Date: 2026-10-19. Device: RX 6700 XT (native gfx1031, `HSA_OVERRIDE_GFX_VERSION=10.3.0`),
tags `q6k2`, `q6kinsitu`. Plugin commit `c751c20`.
Kernel: [q6k_triton_dequant.py](file:///home/trave/code/triton/docs/rdna2/performance/q6k_triton_dequant.py),
harness: [rdna2_dequant_insitu_ab.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_dequant_insitu_ab.py)
Artifacts: [q6k_raw.log](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/q6k_raw.log),
[q6k_san.log](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/q6k_san.log),
[q6k_insitu.log](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/q6k_insitu.log),
[q6k_insitu.json](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/q6k_insitu.json).

## Exactness — including the trap that made me misread the first run

The validator compares **NaN/Inf positions and sign bits separately, then requires exact
equality on finite values**. My first pass used `torch.equal`, and because random block bytes
put a NaN in the fp16 `d` field (bytes 208-209 of a 210-byte Q6_K block), 9 of 16 cases
reported as failures with `first mismatch [2, 166] nan nan` — an *identical* decode, since
NaN != NaN. This is the fourth time that trap bit me in this program; the validator now cannot
hit it, and `--sanitize-d` gives a mode with no NaN at all.

| case | fp16 | bf16 |
| --- | --- | --- |
| random block bytes, NaN-aware compare (n = 1 … 4097 blocks) | exact, 0 finite mismatches | exact, 0 finite mismatches |
| `d` forced to a normal exponent | exact | exact |

So the open question from `p7-e2e-blocked-and-quant-map.md` — whether the bf16 mismatch was an
interpreter artifact or a real bug — is closed: **there was no mismatch**; it was the
comparison primitive. The reference is this plugin's own `dequantize_blocks_Q6_K`, so the
guarantee is against production code, not a reimplementation.

## In-situ A/B/A through `dequantize()`

Flag toggled on the module attribute so both arms share one process, one clock state, and the
real dispatch (`qtype` selection, block reshape, output shape). Interleaved off/on/off/on/off,
median of 7, best-of reported:

| tensor shape | torch off | Triton on | speedup | A/A band | peak VRAM off → on | fell_back |
| --- | --- | --- | --- | --- | --- | --- |
| 24576×4096 | 10.642 ms | 1.411 ms | **7.54×** | 0.186 ms | 615 → 303 MiB | 0 |
| 4096×12288 | 4.881 ms | 0.681 ms | **7.17×** | 0.040 ms | 324 → 168 MiB | 0 |
| 4096×4096 | 1.531 ms | 0.109 ms | **14.05×** | 0.007 ms | 130 → 78 MiB | 0 |

Bitwise equal, shape and dtype match in every case (`integrated-path exactness failures: none`).
Peak VRAM goes *down* by roughly half in all three cases, because the torch path materializes
the intermediate broadcast/int8 tensors that the fused kernel never creates — which is the
reason weight caching was rejected earlier (OPT-036) and why this path helps the memory story.

## What that is worth per diffusion step

Counting from the GGUF header (recorded in `p7-e2e-blocked-and-quant-map.md`), the Q6_K
tensors in production are **all 128 attention projections** (4096×4096) plus **16 of the 32
down-projections** (4096×12288):

```
128 × (1.531 - 0.109) ms = 181.9 ms
 16 × (4.881 - 0.681) ms =  67.2 ms
                             ------
                            ~249 ms per step
```

Combined with the shipped Q5_K kernel (~378 ms/step), the two fused dequants together address
roughly **627 ms/step**, about 15% of a ~4.3 s step, and they are the only two measured wins
in this program that are *not* gated off by an inadmissible end-to-end gate — they are wired
into the plugin and active whenever `GGUF_TRITON_DEQUANT=1`.

## Not claimed

* No whole-step timing: the end-to-end image gate remains contract-inadmissible at production
  geometry (`p7-e2e-blocked-and-quant-map.md` / task "image gate"), so 249 ms/step is the sum of
  measured per-tensor deltas, not a measured step delta. It also assumes dequant happens once
  per tensor per step, which is what the header scan and `cast_bias_weight` imply but the step
  measurement would confirm.
* The 14.05× on 4096×4096 comes with a "438 GB/s" figure in the standalone bench that exceeds
  this card's ~268 GB/s peak, so treat the *ratio* as the reliable number and the absolute
  bandwidth as a modelling artifact of the small tensor (input partly resident across reps).
