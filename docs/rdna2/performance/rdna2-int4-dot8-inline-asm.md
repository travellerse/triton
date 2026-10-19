# RDNA2 int4 `v_dot8`: value-correct through inline asm, 1.98x `v_dot4`

Status: **MEASURED** on native gfx1031 (RX 6700 XT). Date: 2026-10-20.
Probe: [rdna2_int4_dot8_probe.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_int4_dot8_probe.py).

## Question

The gate map called `V_DOT8_I32_I4`/`V_DOT8_U32_U4` the one RDNA2 capability "Triton cannot express
at all", because `tl.dot` has no int4 operand type. The instruction itself is a VOP3 op with three
VGPR operands, so it should be reachable through `tl.inline_asm_elementwise`. This probe checks the
values and what the extra four products per instruction are worth.

## Method

Both arms run the same structure: four independent accumulators (so the dots do not serialise on one
dependency chain) and a runtime loop that re-reads operands from eight small buffers indexed by the
loop counter (so the inline asm cannot be hoisted). `dot8` uses `v_dot8_i32_i4`, `dot4` uses
`v_dot4_i32_i8`; both execute 32 dot instructions in the emitted kernel (LLVM unrolled 8×), which the
probe asserts from `.amdgcn`. Values are compared against a nibble/byte-unpacked reference.

## Results

| arm | value check | time | MACs/s | A/A |
| --- | --- | --- | --- | --- |
| `v_dot8_i32_i4` | exact | 111.50 µs | **75.2 GMAC/s** | 0.65% |
| `v_dot4_i32_i8` | exact | 110.42 µs | 38.0 GMAC/s | 0.33% |

At the same dot-instruction count and essentially the same time, dot8 does exactly twice the MACs —
the ISA's eight int4 products versus four int8 products. The inline-asm expression is therefore
value-correct and delivers the throughput the instruction promises.

## What this does and does not close

* It closes the "inexpressible at all" wording: an int4 kernel can use `v_dot8` today through
  `tl.inline_asm_elementwise` (and the unsigned form assembles too, though only the signed form is
  value-checked here).
* It does not add a first-class `tl.dot` int4 path. A real int4 GEMV still has to pack activations
  into nibbles, choose a lane/K layout around the raw instruction, and feed an accumulator; that
  kernel and its end-to-end measurement are not done.
* This is a compute-only microbenchmark: the operands stay in L1/L2 and there is no dequantised
  weight traffic, so it ranks instruction throughput, not a whole kernel.

## Why it matters

`v_dot8` is the one place where an int4 kernel can beat the dequant-then-`v_dot4` path rather than
just match it, which makes it relevant to Q4-class inference. The probe shows the hardware and the
inline-asm route are real; the remaining work is a kernel, not a capability gate.
