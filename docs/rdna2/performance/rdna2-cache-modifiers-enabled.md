# RDNA2 cache modifiers are now emitted (was: silently dropped)

Status: **CONFIRMED** — encoding verified in machine code, **bitwise-identical results measured on
device**; performance effect **NOT ESTABLISHED** (below the noise this benchmark can resolve).
Date: 2026-10-19. Tags `cmod`, `cmod2`, `state=EXCLUSIVE_COOPERATIVE_LOCK`. Device: RX 6700 XT
(`gcn=gfx1030`, native part is gfx1031).
Supersedes the "known gap" state recorded in
[cache-modifier-gap-rdna2.md](file:///home/trave/code/triton/docs/rdna2/cache-modifier-gap-rdna2.md).

## The change

`getCtrlBitsForCacheModifierOnTarget`
([third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:723](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp#L723))
fell through to `default:` for `ISAFamily::RDNA2`, so **every** cache modifier lowered to `aux = 0`
and was discarded, while RDNA3 emitted 1/7 and CDNA3/4 emitted their SC0/NT/SC1 values. RDNA2 now
uses the CDNA3/CDNA4 table:

| request | aux on RDNA2 now | aux before |
| --- | --- | --- |
| load `.cg` | 3 | 0 (dropped) |
| load `.cs` | 3 | 0 |
| load `.cv` | 17 | 0 |
| store `.cs` | 3 | 0 |
| store `.wt` | 17 | 0 |

Why this table and not RDNA3's: LLVM masks the cache-policy operand with `ALL_pregfx12` for every
pre-GFX12 part, RDNA included (`SIISelLowering.cpp`, `lowerGlobalLoadStore`), so the aux positions
CDNA uses are the valid ones for gfx10; RDNA3's table sets bit 2, whose gfx10 meaning I could not
establish, so it stays unused. `test/TritonGPU/amd/amd-buffer-cache-modifiers-rdna2.mlir` now pins
the 3/3/17/3/17 values for both `gfx1030` and `gfx1031` (previously it pinned the *absence*).

## Evidence 1 — the bits really reach the machine code

`llvm-objdump -d --mcpu=gfx1030` on the hsaco Triton produced for the same kernel:

| variant | instruction bytes | printed cache field |
| --- | --- | --- |
| no modifier | `DC308000` | — |
| `.cg` / `.cs` (aux 3) | `DC328000` | `slc` |
| `.cv` (aux 17) | `DC319000` | `glc dlc` |

So the encoder is not discarding the operand. Note LLVM's *names* for these gfx10 fields
(`glc`/`slc`/`dlc`) are the RDNA names, and the name→bit correspondence is not 1:1 with the CDNA
table's `SC0/NT/SC1` labels — aux 3 encodes as `slc` alone while aux 17 encodes as `glc dlc`. I am
reporting that rather than smoothing it over: what is verified is that distinct, nonzero cache
fields are emitted, not a claim about which ISA field each Triton modifier lands in.

## Evidence 2 — results are unchanged (the safety gate)

Every case compared bitwise against the same kernel with no modifier, and against torch:

```
resident re-read  matches_none=True for none/cg/cs/cv
streaming read    bitwise=True      for none/cg/cs/cv
copy w/ store cm  bitwise=True      for none/cs/wt
```

Cache hints cannot change results on these paths, and they did not.

## Evidence 3 — no measurable performance effect, and my earlier claim retracted

| arm | run `cmod` | run `cmod2` |
| --- | --- | --- |
| resident, no modifier | 9.78 µs | 10.84 µs (repeat: 10.59) |
| `.cg` | 9.91 (x1.013) | 10.51 (x0.970) |
| `.cs` | 10.48 (**x1.071**) | 10.38 (**x0.958**) |
| `.cv` | 9.94 (x1.017) | 10.50 (x0.969) |

The two runs disagree **in sign** for every modifier, the within-run A/A band is 2.37%, and the
same configuration drifted 10.8% between runs. So the `.cs` "7% slower, therefore SC0 is honoured"
reading from the first run was noise, and I am withdrawing it. On the streaming case the card is at
~99% of its ~288 GB/s spec regardless of modifier (147.5–149.5 GB/s read-side accounting,
293–304 GB/s for the copy), which is why a hint has nowhere to go there.

Honest summary: the fix makes Triton honour what the user asked for and removes a silent no-op;
it is not a performance win on this workload, and proving the hardware interpretation of each
bit would need a counter-based test (`rocprofv3` is installed on this host) rather than timing.

## Attempted follow-up: hardware counters, blocked on this host

Timing could not settle the mechanism question, so I tried counters with
[rdna2_cache_modifier_pmc.py](file:///home/trave/code/triton/docs/rdna2/performance/rdna2_cache_modifier_pmc.py)
(one process per modifier, since the launches share a kernel name and would aggregate):

* `--pmc TCP_UTCL1_REQUEST GL2C_HIT GL2C_MISS` → rocprofv3 reported `Missing:
  [TCP_UTCL1_REQUEST]` for agent gpu-0 (gfx1031); `GL2C_HIT`/`GL2C_MISS` are collectible. So the
  L1-bypass hypothesis cannot be tested by counting L1 requests directly with this counter set.
* Retrying with only the collectible pair (L2 hits/misses, which should multiply if repeats stop
  being served from L1) **hung**: after `HSA version 1.18.0 initialized` the process produced no
  CSV and no further log for >5 minutes, and had to be stopped. No counter rows were collected,
  for any variant.

So this remains genuinely open, and the concrete next attempt is identified rather than vague:
profile a *pre-compiled* launch (populate Triton's cache first, then run a script that only loads
and launches the cached kernel) so rocprofv3's interception does not overlap Triton's in-process
LLVM compilation — or use the AMD metrics API directly instead of the CLI.

## Collateral

`/usr/bin/lit` over `test/Conversion/amd`, `test/TritonGPU/amd`, `test/TritonAMDGPUTransforms`:
**129/129 pass**, including the RDNA3 and CDNA cache-modifier tests, which the change does not
touch (it only adds a case before the RDNA3 arm).
