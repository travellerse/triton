# End-to-end gate: BLOCKED, and the exact quant-type map that corrects my own sizing

Date 2026-10-19. Two things came out of one attempt.

## 1. End-to-end A/B of the two flags is contract-blocked, not merely awkward

Attempt: three full-lifecycle runs (`off / on / off`) of the production editing workflow at a reduced
512×768 latent, fixed seed 1234567890, 8 steps, driven through `evidence/harness/run.sh` (which is
itself an admission participant: `gpu_health` → `gpu_lock <tag> <budget>` held for the whole
lifecycle → release). No images were produced, so **the paired image gate for the repack is
`BLOCKED`, not passed and not failed**.

What actually happened, from `server.log`:

```
loaded completely; 11004.80 MB usable, 8917.48 MB loaded, full load: True   (text encoder)
Requested to load QwenImage21
loaded partially; 932.70 MB usable, 727.02 MB loaded, 4550.00 MB offloaded   (DiT)
  0%| | 0/8  Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
```

Kernel log confirms it: `amdgpu ... [gfxhub] page fault ... Process python pid 141008`, five times.

Two distinct causes, both mine, neither about the flags:

* **I used the base `launch.sh`**, which lacks the measurement flags earlier phases required
  (`--disable-async-offload --cache-none`, `reports/50:75`), and the text encoder was still resident
  (`0 models unloaded`) — so the DiT went into partial/low-vram streaming and faulted on step 1.
* The deeper point stands regardless of resolution: the resident quantised weights are **9.55 GB**
  and their fp16 expansion is **14.23 GB** (computed from the GGUF header, matching OPT-036's
  13.96 GB). The admissible budget ceiling on this card is ~9.8 GiB (`used + budget + 1200 ≤ 12272`
  with `used ≈ 1.27 GiB`), and shrinking the *image* only shrinks activations. So an honest peak for
  this workflow exceeds what the contract can admit **at any geometry** — which is precisely the
  limitation the contract already documents, now confirmed by an attempt rather than by reading.

Afterwards, further device work is refused by design: `state=BUSY`, reason `recent GPU fault lines: 5`.
That is the wrapper failing closed correctly. I did not retry in a loop, lower the fault window, reset
the GPU, or touch any threshold, and the device recovered on its own (VRAM back to 1.59 GB, no
lingering server).

## 2. Exact quant-type per role, which corrects my own step-share claim

Read from the model header rather than assumed:

| role | tensors | params | type | covered by the shipped kernel? |
|---|---|---|---|---|
| `img_mlp.gate_up` | 32 | 3.221 B (45.3%) | **Q5_K** | yes |
| `img_mlp.out` (down) | 16 | 0.805 B | Q5_K | yes |
| `img_mlp.out` (down) | 16 | 0.805 B | Q6_K | no |
| `attn.to_q/to_k/to_v/to_out` | 4×32 | 2.148 B (30.2%) | **Q6_K** | no |
| `modulation` | 1 | 0.067 B | Q8_0 | no |
| embeddings/norms | 72 | 0.069 B | BF16 | trivial |

My earlier "in-situ ~650 ms/step, 6.8% captured" line was wrong in attribution: it applied the
measured Q5_K speedups to call shapes that are, for the attention projections, **Q6_K tensors**.
Restated with the real map and the measured per-type numbers:

* captured by the shipped Q5_K kernel: gate_up 32 × 9.42 ms + half the down projections
  16 × 4.83 ms ≈ **378 ms/step ≈ 3.9%**;
* still on the torch path, and the target of `q6k_triton_dequant.py`: 128 projections × ~1.5 ms +
  16 down ≈ **~270 ms/step (2.8%)** more, plus Q8_0 and the BF16 group.

So the Q5_K kernel gets a bit over half the available prize, the Q6_K kernel is worth roughly the
other 2.8% of step, and neither figure is a claim about end-to-end wall time — that remains blocked
as above and must be labelled projected until someone can run the workflow under a raised ceiling.

## 3. Q6_K kernel status

Written (`docs/rdna2/performance/q6k_triton_dequant.py`) with the block layout derived from
`dequantize_blocks_Q6_K` — 210 bytes/256 values, `ql[0:128] | qh[128:192] | scales[192:208] int8 |
d[208:210] fp16` — and torch's two-step rounding reproduced. **Not yet validated**: its acceptance is
bit-exactness over several block counts including non-multiples of the tile, fp16 and bf16, then the
same in-situ A/B/A the Q5_K kernel passed. The GPU is currently in the fault-window refusal, so that
verification is queued, not skipped.
