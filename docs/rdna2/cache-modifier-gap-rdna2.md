# Cache modifiers are silently dropped on RDNA2 (GFX10)

**RESOLVED 2026-10-19** — RDNA2 now emits the pre-GFX12 `SC0/NT/SC1` values; see
[performance/rdna2-cache-modifiers-enabled.md](file:///home/trave/code/triton/docs/rdna2/performance/rdna2-cache-modifiers-enabled.md)
for the machine-code proof, the bitwise-identical device results, and the honest statement that no
performance effect was measurable. This file stays as the diagnosis and as the record of why the
RDNA3 table was the wrong one to copy.

Status at diagnosis time: **CONFIRMED** (static, device-free). The hardware effect of a hypothetical
GFX10 mapping was **NOT_VERIFIED** and deliberately not guessed at that point.
Date: 2026-10-19. Triton `work/rdna2-gfx1031-m0` @ 2122e24efc + local commits.

## Finding

`tl.load(..., cache_modifier=".cg"/".cs"/".cv")` and `tl.store(..., ".cs"/".wt")`
compile on gfx1030/gfx1031 and are numerically correct, but **none of the qualifiers
reach the instruction**: every lowered `rocdl.raw.ptr.buffer.load/store` gets the
default cache-policy aux operand `0`. RDNA3 emits a real encoding for the same source.

`getCtrlBitsForCacheModifierOnTarget()`
([third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:723](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp#L723))
dispatches per `ISAFamily`:

| family | helper | gfx1030/gfx1031 |
| --- | --- | --- |
| CDNA3, CDNA4 | `..._CDNA3_CDNA4` (SC0/NT/SC1) | no |
| RDNA3, RDNA4m | `...OnRDNA3` (GLC/SLC/DLC) | no |
| RDNA4, CDNA5 | `..._GFX12` (SCOPE/TH) | no |
| **RDNA1, RDNA2, Unknown** | `getDefaultCtrlBitsForCacheModifier` → always `0` | **yes** |

So the drop is not new to this branch. Classifying gfx103x as `ISAFamily::RDNA2`
did not change it either: before that fix the arch was `Unknown`, which also landed
in `default:`. Upstream added the RDNA3 mapping when someone with RDNA3 hardware
needed it (`a357e1a77b`); no GFX10 mapping has ever existed, and the pre-CDNA2
cleanup (`4a5147d919`) removed nothing related to it.

Every call site goes through this one helper — `BufferOpsEmitter.cpp:288`,
`MaskedOpsToLLVM.cpp:59`, and five places in `LoadStoreOpToLLVM.cpp` (1105, 1260,
1370, 1439, 1517) — so the gap covers buffer ops, global ops, and masked ops alike.
The helpers switch only on the `cache_modifier` field of `#tt.cache_policy`, so the
emitted aux never depends on `eviction_policy` either (that is an all-AMD
observation, not RDNA2-specific).

## Evidence

Same pipeline, four arches, on the RDNA3 lit input
(`test/TritonGPU/amd/amd-buffer-cache-modifiers-rdna3.mlir`), reading the final
operand of each lowered buffer op. Cases in order: `load.cg load.cs load.cv
store.cs store.wt`, four accesses each (128 f32 at `sizePerThread = 4`):

```
gfx1100   aux: 1 1 1 1 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
gfx1150   aux: 1 1 1 1 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
gfx1030   aux: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
gfx1031   aux: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

Raw capture: [conformance/cache_mod_aux.txt](file:///home/trave/code/triton/docs/rdna2/conformance/cache_mod_aux.txt).

This is also why the earlier on-device conformance probe (`rdna2_opclass_probe.py`,
43/43 at native gfx1031) could not see it: that probe compares results against
torch, and a dropped cache hint produces identical results. Only inspecting the
lowering reveals the gap.

## New test

`test/TritonGPU/amd/amd-buffer-cache-modifiers-rdna2.mlir` pins the current
behaviour for `gfx-arch=gfx1030` and `gfx-arch=gfx1031`. Each case asserts
`CHECK-COUNT-4` lowered accesses with `aux = 0`. Because `aux` is the last operand,
requiring `0` on every access is simultaneously the negative assertion — no nonzero
encoding can match — so adding a GFX10 mapping forces this file to be updated on
purpose.

Verified with the freshly built `triton-opt` (lit `PASS` on both arches) and
mutation-checked to prove it is not decoration:

| mutation applied to the lowering output | FileCheck result |
| --- | --- |
| none (baseline gfx1030 / gfx1031) | exit 0 / exit 0 |
| `load ... , 0 :` → `load ... , 7 :` | **exit 1** (aux value is pinned) |
| one lowered `buffer.load` deleted | **exit 1** (access count is pinned) |

A first draft of the test used `CHECK-NOT` between `CHECK-LABEL` and the positive
`CHECK`; that region is empty in this lowering, so the directive passed even when
inverted — it was dropped in favour of the count/value pin above.

## Impact on the real workload

None measured. The Qwen-Image-2.1 attention and GGUF-dequant kernels identified in
`performance/` pass no cache modifiers, so the gap costs nothing today. It is a
compatibility gap, not a performance regression: a user who writes `.cg`/`.cs`/`.wt`
on RDNA2 gets default caching while believing they asked for bypass or streaming,
which is the kind of silent no-op that makes later perf work unreadable.

## What closing it requires (deliberately not attempted here)

1. A GFX10 `SLCCNTL` semantic mapping: which of the popup control bits gfx10
   decodes, and their positions. Note LLVM's `CPol` for all pre-GFX12 parts names
   them `GLC=0x1 SLC=0x2 DLC=0x4 SWZ=0x8 SCC=0x10` (RDNA3 names), while Triton's
   CDNA helper uses the same positions under `SC0=0x1 NT=0x2 SC1=0x10`; which names
   gfx10 hardware honours, and what `.cg`/`.cs`/`.cv`/`.wt` should map to, is a
   hardware question. Static ISA probing here (assembler operand syntax, LLVM test
   mining) reached no authoritative answer, and I will not ship invented bits: an
   aux operand changes memory semantics, so a wrong guess is a silent-correctness
   bug replacing a silent-no-op.
2. On-device arbitration that the mapping is both honoured and not a slowdown
   (an admitted A/B against `aux = 0`), plus a numerics run.
3. Only then flip the test to the new encodings.
