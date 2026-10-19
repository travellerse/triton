# RDNA2 fast attention is register-bound, not LDS-bound

Status: **CONFIRMED** statically (emitted `amdhsa` metadata + pass source). The runtime cost
was subsequently **MEASURED** on device -- see
[rdna2-spill-cost-measured.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-spill-cost-measured.md),
whose Result-1 correction to this file's recommendation is applied below.
Date: 2026-10-19. Triton `work/rdna2-gfx1031-m0`, `triton-opt`/`libtriton` built from this tree.
Harness: [rdna2_dot_spill_map.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_dot_spill_map.py),
[rdna2_inthread_transpose_probe.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_inthread_transpose_probe.py),
raw output: [artifacts/dot_spill_map_2026-10-19.txt](file:///home/trave/code/triton/docs/rdna2/performance/artifacts/dot_spill_map_2026-10-19.txt).

All of this was produced **without touching the GPU** — `triton.compile(ASTSource, target=GPUTarget("hip", arch, 32))`
runs the whole AMD pipeline including codegen, so `.amdgcn` and its metadata are available
statically. That is what makes this kind of audit possible during a BUSY window.

## 1. Upstream's flash-attention LDS trick cannot reach RDNA2

`InThreadTranspose` rewrites `tt.load → local_alloc → local_load → dot operand` so the
register tile is transposed *inside* a thread before the shared store, avoiding scalar
16-bit LDS traffic. Upstream #10390 (commit `0418ee6c1e`, 2026-05-27) enabled it for
gfx942 / gfx110x / gfx115x / gfx120x, reporting 512 scalar `ds_load_u16` → 144
`ds_load_b128` and **+3.8% median** on AITER `flash_attn_2.varlen_fwd` at a ViT prefill
shape. The gate is `is_in_thread_transpose_enabled()`
([third_party/amd/backend/compiler.py:173](file:///home/trave/code/triton/third_party/amd/backend/compiler.py#L173)) —
an explicit arch allow-list that does not contain gfx103x.

The gate is *not* the only blocker. `matchInThreadTransposePattern()` requires the operand's
`DotOperandEncodingAttr` parent to be MFMA or WMMA
([third_party/amd/lib/TritonAMDGPUTransforms/InThreadTranspose.cpp:592](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUTransforms/InThreadTranspose.cpp#L592)),
and RDNA2 has neither: it runs the FMA path, so its dot operands carry
`#ttg.dot_op<{opIdx = .., parent = #blocked}>`.

Probe result, with `knobs.amd.use_in_thread_transpose = True` (i.e. gate bypassed) and
realistic divisibility hints:

```
gfx942    pass_fired=YES  dot_op_parents=['#mma']   loads: global_load_dwordx4 x4
gfx1100   pass_fired=YES  dot_op_parents=['#mma']   loads: global_load_b128 x8, ds_load_b128 x32
gfx1030   pass_fired= no  dot_op_parents=['#blocked'] loads: global_load_dwordx4 x8, buffer_load_dword x40
gfx1031   pass_fired= no  dot_op_parents=['#blocked'] (identical)
```

With hints supplied, RDNA2's operand traffic is already fully vectorized
(`global_load_dwordx4`, `ds_write_b128`), so there is nothing for the pass to fix on the
memory side in this shape either. **Conclusion: this lever is structurally unavailable on
RDNA2; RDNA2 fast attention has to come from elsewhere.**

### Harness trap worth recording

Without `tt.divisibility = 16` in `ASTSource.attrs` (the JIT adds it at launch time for
16-byte-aligned pointers, the static path does not), *every* arch degenerates:
`global_load_ushort` ×64, `ds_write_b16` ×64 on gfx1030. I initially read that as an RDNA2
pathology; it is a harness artifact (`--no-hints` reproduces it). Any static codegen
comparison must pass the hints.

## 2. What actually limits RDNA2: VGPR spills in the FMA dot

The `buffer_load_dword` ×40 / `buffer_store_dword` ×40 above are **scratch register spills**
(`buffer_load_dword vN, off, s[8:11], 0 offset:144`, marked `16-byte Folded Reload`, and the
metadata says `.vgpr_spill_count: 40`, `.private_segment_fixed_size: 164`). Scratch lives in
VRAM, so a spilling attention tile puts per-element register traffic on the global path.

Sweeping tile shape × head_dim × warps (single dot, `num_stages=1`, 64 threads/warp):

| tile BM×BN | D | warps | gfx1030 spills | gfx1100 spills |
| --- | --- | --- | --- | --- |
| 32×32 | 64 | 2 | 0 | 0 |
| 32×32 | 128 | 2 | 56 | 0 |
| 32×32 | 128 | 4 | 0 | 0 |
| 64×64 | 64 | 4 | 40 | 0 |
| 64×64 | 64 | 8 | 0 | 0 |
| 64×64 | 128 | 4 | 304 | 0 |
| 64×64 | 128 | 8 | 40 | 0 |
| 128×64 | 128 | 8 | 301 | 0 |
| 128×128 | 64 | 8 | 444 | 0 |
| 128×128 | 128 | 8 | 816 | 88 |

RDNA2 is spill-free only in a narrow band — 32×32 (≥2 warps at D=64, ≥4 at D=128) and
64×64×64 at 8 warps — and every 128-wide tile spills at every warp count. RDNA3's WMMA
path is spill-free at all of the rows above except 128×128×128, where it still spills 9×
less. Reading the register cost: the FMA path keeps the f32 accumulator **and** both fp16
operand tiles in registers, so registers scale as `BM*BN + 2*K*(BM+BN)/2` spread over
`32*warps` lanes; RDNA2 therefore needs roughly **twice the warps of RDNA3 for the same
tile**, which in turn shrinks per-warp ILP and raises the wave's LDS/barrier pressure.

Caveats, stated as such:

* These are optimistic. A real flash-attention inner loop also keeps `m`, `l`, the softmax
  tile and the running `acc` alive, so its spill count is ≥ the table above at the same tile.
* On RDNA2 **all 34** spilling rows report `next_free_vgpr = 256`, the gfx10 per-thread
  ceiling, so the allocator is genuinely out of registers rather than mis-preferring spills.
  RDNA3 is the same story but milder (13 of its 20 spilling rows sit at 256, the rest at
  251–255).
* Nothing here is measured on hardware; the *cost* of 40 spills (a 640 B/lane scratch
  round trip) versus a smaller, spill-free tile is exactly the on-device question this
  table is meant to ask.

## 3. Consequences for the RDNA2 fast-attention work

1. **Design constraint, not a tuning knob:** an RDNA2 flash-attention kernel must be built
   to a VGPR budget. Candidate configs that are provably spill-free from this table:
   `BM=BN=32, D=64, num_warps=4` and `BM=BN=64, D=64, num_warps=8`; for D=128 the only
   spill-free shape found is `32×32` at 4-8 warps.
   **Correction after measurement** (see
   [rdna2-spill-cost-measured.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-spill-cost-measured.md)):
   the spill-free points are *not* the fastest ones. `32×32, D=128, num_warps=8` has 0 spills
   and runs 17% **slower** than the 947-spill `64×64/w4` baseline, and at `D=64` it is 1.81×
   slower than the 9-spill `w4` arm. The measured rule is "minimize spills subject to keeping
   enough per-lane work", which lands on `32×32 × num_warps=4` (222 spills, 1.495× faster than
   baseline, 4.82 TFLOPs = 81% of the PV-shaped ceiling).2. **The structural fix is operand-stationary FMA**, i.e. consuming one dot operand straight
   from LDS (or in a narrower `kWidth`) instead of materializing both operand tiles in
   registers. `kWidth = 4` for the blocked parent is where to start looking — RDNA3's WMMA
   operand uses `[4, 8]` sizePerThread while RDNA2's FMA dot picked `[4, 4]`.
3. **`num_stages` is not a lever** (checked statically, same timed kernel, `gfx1030`):

   | config | stages 1 | 2 | 3 | 4 |
   | --- | --- | --- | --- | --- |
   | 64×64, D=128, W=4 | 947 | 846 | 845 | 1794 |
   | 64×64, D=64, W=4 | 395 | 413 | 568 | 723 |
   | 32×32, D=128, W=4 | 222 | 223 | 441 | 440 |

   Pipelining never removes the spills; deeper pipelines add live buffers and make them
   worse. `next_free_vgpr` stays pinned at 256 in every row. So the pressure comes from the
   tile itself, not from scheduling.
4. **Spill cost arbitration — done.** Two admitted runs compared `64×64/D=128/w4` against
   lower-spill tiles; the result, including the finding that the spill-free point is a
   regression, is in `rdna2-spill-cost-measured.md`.
5. This supersedes the LDS-vectorization hypothesis as the target for RDNA2 fast attention:
   reads and writes are already `b128` here (P5 restored the vec-128 tile, which is what
   made `ds_read_b128`/`ds_write_b128` possible).
