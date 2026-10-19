# RDNA2 cache modifiers: encoding derived from the RDNA2 ISA, then verified in machine code

Status: **CONFIRMED** for the aux → instruction-bit mapping (through Triton's own compiler, on both
gfx1030 and gfx1031). **NOT ESTABLISHED** for any performance effect: the timing arms below are
null and my earlier bandwidth-saturation explanation is withdrawn.
Date: 2026-10-19 (post-review rewrite). Tags `cmod3`. Device: RX 6700 XT (`gcn=gfx1030`, native
part gfx1031).
Supersedes the values in the previous version of this file and the "known gap" state in
[cache-modifier-gap-rdna2.md](file:///home/trave/code/triton/docs/rdna2/cache-modifier-gap-rdna2.md).

## Why this version exists

A review of the first patch rejected it: it proved that *nonzero* aux values were emitted but not
that `.cg/.cs/.cv/.wt` meant what RDNA2 thinks they mean, it copied a table from another family,
and the lit test would have frozen an unverified mapping. Three specific defects, all mine:

1. The first patch routed RDNA2 to the **CDNA3/CDNA4** table (3/3/17/3/17). On GFX10 that is wrong
   by construction: CDNA's third field is bit 4, and the GFX10 buffer encoding has no bit 4.
2. The second version routed RDNA2 to the **RDNA3** table (1/7/7). Its "DLC = MALL no-allocate"
   reading is a GFX11+ meaning; on GFX10 DLC does something else entirely.
3. "Evidence 1" in the old version printed `DC308000 / DC328000 / DC319000` as if those were buffer
   instructions. `0xDC…` is a *global* (flat) load encoding; buffer ops on this part are `0xE038…`.
   I read the wrong instruction class, and the aux-to-field claims drawn from it are retracted.

## The encoding, from RDNA2's own documentation

AMD "RDNA 2" Instruction Set Architecture Reference Guide (Dec 2020), section 13.6.1 (MTBUF field
table) gives the three cache fields of a buffer instruction: **GLC = bit 14, DLC = bit 15,
SLC = bit 54**. Section 8.1.10 tables 38 and 39 give what the combinations *do*:

| SLC | DLC | load: L2 / L1 | store: L2 |
| --- | --- | --- | --- |
| 0 | 0 | LRU / hit-LRU | LRU |
| 0 | 1 | LRU / **miss-evict** | **bypass** |
| 1 | 0 | stream / hit-LRU | stream (hit does not reset age) |
| 1 | 1 | hit-no-allocate / miss-evict | hit-no-allocate |

GLC is a separate, near-cache control: for a load it "intentionally misses the GPU L0 and reads
from L2", invalidating a matching L0 line; for a store it removes L0 persistence across wavefronts.
Stores always miss-evict L0 and L1 is bypassed-but-coherent for stores.

That yields the RDNA2 table now in
[Utility.cpp](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp):

| request | aux | fields | RDNA2 behaviour | exact? |
| --- | --- | --- | --- | --- |
| load `.ca` | 0 | — | L0 hit, L1 hit-LRU, L2 LRU | yes |
| load `.cg` | 5 | GLC\|DLC | L0 bypassed, L1 miss-evict, L2 still LRU | yes |
| load `.cs` | 6 | SLC\|DLC | L1 miss-evict, L2 hit-no-allocate | yes |
| load `.cv` | 5 | GLC\|DLC | same as `.cg` | **no** — see below |
| store `.wb` | 0 | — | L2 LRU | yes |
| store `.cg` | 0 | — | L2 LRU; L1 already bypassed for stores | yes |
| store `.cs` | 2 | SLC | L2 stream | yes |
| store `.wt` | 5 | GLC\|DLC | L2 bypass | **approximate** |

Two limits are ISA facts, not choices: no load encoding can bypass L2, so `.cv` and `.cg` collapse
onto the same bits and the remaining half of `.cv` (system-scope visibility) is a `s_waitcnt` that
an aux value cannot express; and "L2 bypass" for a store is write-*around*, not architectural
write-through to system memory.

## Evidence 1 — aux really lands on those bits, through Triton's own pipeline

[rdna2_cache_policy_end_to_end.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_cache_policy_end_to_end.py)
compiles a real `@triton.jit` copy kernel per modifier (`triton.compile`, explicit `GPUTarget`, no
GPU touched), then decodes the emitted **hsaco instruction words** at bits 14/15/54. Each case also
asserts the *other* access in the same kernel — requested with no modifier — still encodes nothing,
which is what catches a hint leaking across ops or being dropped:

```
cpu=gfx1030   triton=3.9.0 at /home/trave/code/triton/python/triton
op    cm     aux want                 text     machine code  verdict
load  none     0 -                 4/0            2+2/0 bad  OK
load  .cg      5 dlc,glc           4/0            2+2/0 bad  OK
load  .cs      6 dlc,slc           4/0            2+2/0 bad  OK
load  .cv      5 dlc,glc           4/0            2+2/0 bad  OK
store .cs      2 slc               4/0            2+2/0 bad  OK
store .wt      5 dlc,glc           4/0            2+2/0 bad  OK      (10 cases, all OK)
```

`.amdgcn` text and instruction bits agree for every case, on `gfx1030` and `gfx1031`. Example word
for a `.cg` load: `E038D000 80010008` → 0x4000 (bit 14) + 0x8000 (bit 15) set, word-1 bit 22 clear.

The harness has been mutation-checked: setting the expected aux for `.cg` to 0 and for `.wt` to 2
makes both the text and the word decode report MISMATCH (exit 1), so the OKs are not decoration.

## Evidence 2 — the mapping survives an LLVM change

* [test/TritonGPU/amd/amd-buffer-cache-modifiers-rdna2.mlir](file:///home/trave/code/triton/test/TritonGPU/amd/amd-buffer-cache-modifiers-rdna2.mlir)
  pins the aux constant Triton emits per modifier (RUN lines for gfx1030 **and** gfx1031,
  `CHECK-COUNT-4` on the value, plus `.ca` load and `.cg` store rows pinned at 0 so "everything
  became 5" cannot pass).
* [test/LLVMIR/rdna2-buffer-load-cache-policy.ll](file:///home/trave/code/triton/test/LLVMIR/rdna2-buffer-load-cache-policy.ll)
  pins the *other* half — what LLVM encodes from each aux on gfx10 — by running `llc` directly:
  single-field cases establish bit 0→glc, 1→slc, 2→dlc; `aux=16` must encode identically to
  `aux=0` (GFX10 has no field there even though LLVM's pre-GFX12 mask admits SCC); and the three
  values Triton actually emits (5, 6, 2) are each checked to survive. Verified across **all 32**
  aux values on both parts: bits ≥3 are discarded, so 8..15 alias 0..7 and 16..31 alias 0..15.
  Mutation-checked: aux 1→2, 7→3, 16→17, 4→6 and the store 7→6 each make FileCheck exit 1.

A first draft of that second test failed for a dumb reason worth recording: the kernel names
contained `glc`/`slc`/`dlc`, so the `CHECK-NOT`s matched the neighbouring `.globl` symbol lines.
Names now carry no field substrings.

## Evidence 3 — no performance effect, and the saturation story was wrong

`rdna2_cache_modifier_honour.py` on device, median of bracketed launch batches, baseline arm
repeated for the A/A band (run `cmod`, superseded values 1/7/7):

| arm | median | vs baseline |
| --- | --- | --- |
| resident re-read, no modifier | 9.78 µs | — |
| `.cg` | 9.91 µs | ×1.013 |
| `.cs` | 10.48 µs | ×1.071 |
| `.cv` | 9.94 µs | ×1.017 |
| streaming read (134 MB) | 0.898 ms | 149.5 GB/s read-side |
| copy | 0.916 ms | 146.6 GB/s each direction (293 GB/s summed) |

The old text claimed the card "is at ~99% of its ~288 GB/s spec". Both halves were wrong: the RX
6700 XT spec is **384 GB/s** (16 Gbps × 192-bit), and 293 GB/s was *summed* read+write traffic,
which cannot be compared against a unidirectional figure. Measured read-only bandwidth is **149.5
GB/s ≈ 39% of spec**. So the null result is *not* explained by saturation — there is headroom, and
these hints did not use it. That is a real open question, not a closed one.

What is now verified is the mapping; what is still unverified is whether GLC/SLC/DLC change
measurable traffic on this part, because the timing arms are within or near the 2.37% A/A band and
the two runs disagreed in sign. The discriminator is counters, not timing: `GL2C_HIT`/`GL2C_MISS`
are collectible on this host (`TCP_UTCL1_REQUEST` is not), and the concrete next attempt is to
profile a **pre-compiled** launch so rocprofv3's interception does not overlap Triton's in-process
LLVM compilation, or to use the AMD metrics API directly.

## Collateral

`/usr/bin/lit` over `test/TritonGPU/amd`, `test/Conversion/amd`, `test/LLVMIR`: **135/136 pass**.
The single failure is `test/LLVMIR/nvptx-vectorizer.ll`, which is unrelated and environmental: this
build's `triton-llvm-opt` does not register the NVIDIA passes at all
(`triton-llvm-opt: Unknown command line argument '-nvptx-vectorize'`), so its stdin is empty.
A full-tree `ninja` still fails in `third_party/proton` (`RocprofSDKProfiler.cpp`) on host
libstdc++ 16 headers (`/usr/include/c++/16/format`, `-Wtemplate-body`), independent of this change;
`triton-opt` and `libtriton.so` build fine with `-j6`.
