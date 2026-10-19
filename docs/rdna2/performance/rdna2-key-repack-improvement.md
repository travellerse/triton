# RDNA2 key-repack: first MEASURED IMPROVEMENT on the real workload

Date 2026-10-19. Decision: **`MEASURED IMPROVEMENT`**. Host RX 6700 XT, emitted target `gfx1030`
under the pre-existing `HSA_OVERRIDE_GFX_VERSION=10.3.0`, ComfyUI `self` @ `13bd8f7e9`, fork Triton
3.9.0 (workload arm ran the workload's own Triton 3.7.1 — the repack kernel uses no fork-specific
feature). All GPU runs admitted, `state=EXCLUSIVE_COOPERATIVE_LOCK`
(`/tmp/rdna2-admission/audit/*-p6-*.json`, `*-p7-*.json`).

## The bottleneck and how it was found

Discovered accidentally while timing a flash-attention attempt: re-deriving the attention key in a
different memory order made the same math run 1.5× faster. Isolating that variable measured
(`rdna2_key_layout_ab.py`, production shapes q=5120, k=10287, H=32, D=128, fp16, mask=None,
`query_chunk_size=1024`):

| key operand | call ms | QK^T kernel selected | QK^T ms |
|---|---|---|---|
| as production builds it — `[BH, D, N]`, strides `(128, 1, 4096)`, token axis strided | 102.3 | `Cijk_Alik_Bljk_MT128x256x16` | 55.71 |
| packed `[BH, D, N]`, token axis contiguous | 74.3 | `Cijk_Ailk_Bljk_MT128x128x8` | 24.01 |

Same FLOPs, same library, same shapes: the strided B operand costs QK^T a factor of 2.3. This is the
mechanism behind the P1 study's otherwise puzzling production figures (QK^T 6.55 TFLOPs in the
workload vs 17.59 TFLOPs for the identical GEMM measured in isolation).

## Why `contiguous_key` was already known to be useless — and confirmed here

Adding torch's own `.contiguous()` back, **with the copy inside the timed region**, gives 99.7 ms —
only ~2.9 ms better than baseline. The transposed gather costs ~26 ms and eats almost the entire
28 ms GEMM win. That closes the loop on the ledger's rejection of `contiguous_key`: the flag was not
inert after all, it was correct but paid more for the copy than it saved. A tiled transpose that is
coalesced on both sides changes that arithmetic completely: 84 MB read + 84 MB written should be
~0.6 ms of traffic, and the Triton kernel lands near that.

## The candidate

`comfy/ldm/modules/attention.py`: a tiled Triton repack (`BD=32, BN=128, num_warps=4`) that reads the
key from whatever strides the projection left and writes a packed `[heads, dim, tokens]` buffer, wired
into the `skip_reshape=False` branch that the DiT actually takes. **Gated by `Q21_TRITON_REPACK_KEY`,
default off**; falls back to the strided operand on any exception or when `dim_head % 32 != 0`, and
counts both outcomes in `_repack_stats`. Nothing else in the attention path was touched.

## Correctness

* Kernel vs `torch.contiguous()`: **bit-exact** (`max_abs = 0.0`, `torch.equal` True) — it produces
  the same tensor, just cheaper.
* End-to-end attention output, candidate vs production baseline: `max_abs = 6.1e-05`, not bitwise
  equal, which is the expected consequence of a different Tensile tile's fp16 accumulation order. It
  matches the magnitude of the torch-repack arm exactly, so it is attributable to the tile, not to
  the repack.
* Not yet run: the model-level paired image gate. Kernel-level equivalence is not image-quality
  evidence; that gate is the remaining step before this should be turned on by default.

## Performance

Same-process A/B/A inside the real `attention_sub_quad`, flag toggled through the module attribute so
both arms share process, allocator, clocks and inputs (`rdna2_repack_prod_ab.py`, 7 samples/arm,
2 warm-ups, 8 s clock ramp, baseline arm run three times):

| arm | medians (ms) | TFLOPs | peak allocated |
|---|---|---|---|
| flag off (×3) | 113.064 / 112.372 / 112.624 | 7.63–7.68 | 1551–1631 MiB |
| flag on (×2) | 90.752 / 90.746 | 9.51 | 1671–1711 MiB |

* **A/A band 0.692 ms; delta −21.94 ms per call; worst-case ordering (candidate min − baseline max)
  is still −22.32 ms**, i.e. the effect is ~32× the noise band and no sample ordering rescues the
  baseline.
* **Speedup 1.242×** on the steady-state attention call.
* Candidate medians agree to 0.006 ms across two separated runs.
* Projected step effect using the Phase-50 attribution (32 such calls/step, step 9616.8 ms):
  **≈702 ms/step ≈ 7.3%**. Peak VRAM rises ~120–160 MiB (the packed key buffer) — a real cost on a
  12 GiB card, and it must be re-measured against the VRAM accounting before default enablement.

## Evidence that the intended path executed

`_repack_stats.applied` in the per-arm record: 0 at the first baseline arm, **10 after the first
candidate arm**, unchanged through the following baseline arm, **19 after the second candidate arm**,
unchanged through the last baseline. Counter only moves while the flag is on, so the measured
difference is the repack and not drift. Kernel identity in the layout study independently shows the
Tensile tile actually changing.

## The four claims, separated

expected code generated — yes (bit-exact packed buffer, tile change observed); measured call
improved — yes, beyond the noise band; complete workload improved — **projected, not measured**:
a full production run stays inadmissible under the GPU contract (honest peak 10.8–11.6 GiB cannot
satisfy `used + budget + 1200 ≤ 12272`), so the step figure above is arithmetic on a measured
per-call delta, not a measured step time.

## Next

1. Paired image-quality gate at production geometry before flipping the default.
2. Same treatment for the value operand and for the projection bucket (the never-decomposed
   ~17–20% q/k/v/out terms from Phase 50) — the operand-layout result suggests layout, not FLOPs,
   is what has been limiting this model on RDNA2.
3. Then the flash-attention rewrite, whose ceiling this raises: PV-shaped dots still cap near
   5.96 TFLOPs, so attention must win by deleting score traffic rather than out-GEMMing rocBLAS.

Reproduce: `docs/rdna2/performance/rdna2_key_layout_ab.py`, `rdna2_key_repack.py`,
`rdna2_repack_prod_ab.py`; raw records in `docs/rdna2/performance/artifacts/`.
