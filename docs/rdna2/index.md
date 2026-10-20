# RDNA2 (gfx103x)

What Triton's AMD backend does on GFX10.3 — Radeon RX 6000 — and what the silicon cannot do. The
target is a consumer card without matrix units, so most of what follows is about the consequences
of that one absence.

## The hardware in one paragraph

RDNA2 has no MFMA and no WMMA. `tl.dot` therefore lowers to per-lane FMA, and the throughput of
every kernel is bounded by the FMA pipe rather than by a tensor core. It also has no tensor
descriptors, no direct-to-LDS load, no bulk async copy and no multicast, so the pipelining
patterns written for CDNA do not apply. What it does have is a full 32-lane integer and fp16 dot
product, buffer atomics over the integer set, and 64 KiB of LDS at 32 banks per wave32.

## Status

`Hardware` says whether the instruction exists. `Triton` says what the backend does with it.
Where a test is named, that test asserts the emitted form.

### Target and codegen

| Capability | Hardware | Triton | Test |
|---|---|---|---|
| ISA family classification (gfx1030–gfx1036) | — | supported | `test/Conversion/amd/rdna2_target_guards.mlir` |
| Buffer resource descriptors | RDNA layout | supported | `test/Conversion/amd/buffer_load_store.mlir` |
| LDS load/store tiles (vec-128) | yes | supported, same lowering as RDNA3 | `test/Conversion/amd/tritongpu_to_llvm_rdna.mlir` |
| Cross-lane reduce | DPP `permlanex16` | supported via `permlanex16` + `readlane` | same file |

### Math

| Capability | Hardware | Triton | Test |
|---|---|---|---|
| fp16 dot | `V_DOT2C_F32_F16` | reaches the packed intrinsic | `tritongpu_to_llvm_rdna.mlir` |
| int8 dot | `V_DOT4C_I32_I8` | reaches the packed intrinsic | same |
| int4 dot | `V_DOT8_I32_I4` | inline asm only; no `tl.dot` int4 path | — |
| Matrix units (MFMA / WMMA) | **absent** | n/a — FMA lowering | — |
| `tl.dot_scaled` (MXFP8 / MXFP4) | no FP8 hardware | software decomposition | `accelerate-amd-matmul-rdna-no-matrix.mlir` |
| fp8 software casts | — | NaN and Inf preserved (E4M3FN, E5M2) | `tritongpu_to_llvm_rdna.mlir` |

### Memory and atomics

| Capability | Hardware | Triton | Test |
|---|---|---|---|
| Direct-to-LDS load, bulk copy | **absent** | n/a | — |
| Tensor descriptors (TDM) | **absent** | decomposed to masked load/store | `accelerate-amd-matmul-rdna-no-matrix.mlir` |
| Integer buffer atomics, incl. masked lanes | yes | supported | `amd-convert-buffer-ops-atomic-rdna2-int.mlir` |
| f32 buffer atomic FMIN/FMAX | yes | not emitted by the frontend | — |
| f32 buffer atomic FADD | **no such opcode** | reported, not silently lowered | — |
| Cache modifiers `.cg/.cs/.cv/.wt` | yes | encoded per the RDNA2 field table | `amd-buffer-cache-modifiers-rdna2.mlir` |

## Measured ceilings

Square and contraction-shaped fp16 GEMM, fp32 accumulate, RX 6700 XT:

| shape | Triton FMA dot | rocBLAS |
|---|---:|---:|
| square 2048³ | 15.19 TFLOPs | 17.59 |
| QK-like (1024×8192×128) | 13.77 | — |
| PV-like (1024×128×8192) | **5.96** | 17.16 |

The PV-shaped contraction is the binding constraint, and it is why a fused attention kernel loses
here: at 1.97× slower than the library path it cannot be won back by removing the softmax round
trip, even though it uses 9.3× less transient memory.

## Known gaps

Separated from the rows above on purpose — the silicon can do these and the software does not:

- **No first-class int4 `tl.dot`.** The instruction is reachable through inline asm at roughly
  twice `V_DOT4`; packing, lane layout and accumulator integration are not done.
- **`eviction_policy` is inert on every AMD target**, not just RDNA2. RDNA2 has no "keep line"
  encoding, so the hint cannot be honoured; whether it should be a no-op or an error upstream is
  an open question.
- **Cache-policy timing is unmeasured.** The encoding is verified in machine code; whether it
  moves wall time is not established.

## Scope

One card: RX 6700 XT (gfx1031 silicon, reported as gfx1030 under `HSA_OVERRIDE_GFX_VERSION=10.3.0`),
ROCm 7.2.53211, torch 2.13.0+rocm7.2. The numbers are per-shape and per-tile. None of them
extrapolate to another RDNA2 part, another resolution or another driver without re-measuring.
