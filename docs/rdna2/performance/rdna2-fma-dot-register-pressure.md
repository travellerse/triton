# The FMA dot lowering is the RDNA2 performance problem — and it is not RDNA2's code

Status: **CONFIRMED** statically on two architectures, with the runtime cost already measured
(`rdna2-spill-cost-measured.md`, 1.495× / 4.82 TFLOPs). Date: 2026-10-19.
Tool: [rdna2_pressure_decomposition.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_pressure_decomposition.py)

## Why this is the headline item

Everything RDNA2-specific I have audited so far is either correct or a documented small gap
([lowering parity sweep](file:///home/trave/code/triton/docs/rdna2/lowering-parity-sweep-rdna2.md):
175 arch-rewritten lit lines, zero unexplained differences). The thing that actually costs
performance is that Triton's **FMA `tt.dot` path spills VGPRs to VRAM**, and RDNA2 has no
tensor cores, so *every* matmul on this card goes through it.

## Evidence 1 — the softmax is not what overflows the register file

`BM=32, BN=32, head_dim=128, num_warps=4` on gfx1030, one construct removed at a time
(`vgpr_spill_count` / `next_free_vgpr` from the emitted `amdhsa` metadata):

| formulation | spills | next_free_vgpr |
| --- | --- | --- |
| full FA step (masked key tail + online softmax) | 226 | 256 |
| without the masked tail (`tl.where` + masked loads removed) | 222 | 256 |
| without online rescaling (no `m_i` / `alpha`) | 212 | 256 |
| **one dot only, QK shape** | 38 | 256 |
| **one dot only, PV shape, no softmax at all** | 17 | 256 |

On gfx1100 the same five rows are `0 / 0 / 0 / 0 / 0` spills with 115–240 registers — but that
is WMMA, so it is not the comparison that settles where the defect lives.

## Evidence 2 — attribution: force RDNA3 onto the same path

Making the operands fp32 removes WMMA and MFMA as options, so gfx1100 must use the *same*
core FMA lowering as gfx1030 (32×32×128, 4 warps):

| arch, fp32 operands, FMA forced | spills | next_free_vgpr |
| --- | --- | --- |
| gfx1030 | 81 | 256 |
| gfx1100 | 72 | 256 |

Both saturate the register file and spill. **The behaviour belongs to the shared FMA dot
lowering, not to RDNA2-specific code.** RDNA2 is simply the family that can never opt out of
it, and it is *better* than RDNA3 doing the same thing (17 spills with fp16 operands vs 72
with fp32), which is the expected direction.

## Evidence 3 — the demand is modest, so the spills look avoidable

For the "PV dot only" case (one dot, probabilities loaded from memory, nothing else), the
per-lane data is roughly: acc `BM×HD` f32 = 32 regs, p `BM×BN` f16 = 4 regs, v `BN×HD` f16 = 16
regs — call it ~60 registers of live data against a 256-register budget. The kernel still uses
all 256 and spills. Two further observations constrain any explanation:

* spill counts are **not monotone in per-lane demand**: at `HD=128, 32×32`, going 1 → 2 → 4
  warps gives `0 → 114 → 17` spills, i.e. the configuration with the *least* data per lane
  spills more than the one with the most;
* `ds_read2_b64` (paired LDS reads) appears only in the spilling warp counts — at `num_warps=1`
  the same kernel has no paired reads, 128 `ds_read_b128` and zero spills.

So the pressure is created by the lowering's *live-range shape* (operand tiles materialized
whole, paired LDS reads, and the f32→f16 copy of the operand) rather than by the volume of
data. That is the part worth fixing, and it is in shared code:
`lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp` and `FMADotUtility.cpp`.

## What a fix would have to do

Constrained by the measurements above, the plausible directions are:

1. **Chunk the FMA contraction over K** so operand registers are recycled per chunk instead of
   the whole `BN×HD` / `BM×BN` tiles being live at once. This is what makes the demand ~60
   registers yet the peak much higher.
2. **Feed one operand from LDS** instead of materializing both tiles in registers (the reason
   `InThreadTranspose` cannot help here is orthogonal — it can't match a blocked parent at all).
3. Failing those, **expose a target-aware cap**: if the FMA path knows the register budget is
   reached, it should prefer a smaller operand tile / more K iterations rather than emit VRAM
   traffic. Today that trade is only reachable by the user changing `BLOCK_M/BLOCK_N/num_warps`,
   which is exactly what the measured 1.495× came from.

None of these is RDNA2-specific work, which is why this is the most valuable Triton-side
optimization found in the whole RDNA2 program — it helps every FMA-fallback target.

## Measured consequence on the real workload

* Attention inner loop: 42.66 → 28.53 ms (1.495×) purely by choosing a tile the current
  lowering can hold, at identical peak VRAM; the fused-attention candidate still loses to the
  library path (182 vs 92 ms) for the same reason — see
  [rdna2-fused-fa-loses-on-rdna2.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-fused-fa-loses-on-rdna2.md).
* `num_stages` cannot fix it (947 → 846 → 845 → 1794 spills), so it is not a scheduling flag.

## Not established

* I have not proven *which* values the allocator is keeping live; that needs `-debug-only=regalloc`
  on the emitted LLVM IR, which I have not run.
* Upstream issue search was unavailable from this authentication context (`gh search issues`
  returns "Invalid search query / cannot be searched"), so I cannot say whether this is already
  filed. Before proposing a change upstream, that check should be redone from a working session.
* No fix has been implemented; this report localizes the defect and quantifies it.
