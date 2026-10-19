# RDNA2 masked buffer atomics: the predicate-off sentinel is dropped

Status: **CONFIRMED** on gfx1031 (RX 6700 XT, native target, not the gfx1030 override) for the
integer buffer-atomic subset that `ConvertToBufferOps` takes on RDNA2.
Date: 2026-10-20.

## The question

The buffer lowering sends a lane whose predicate is false to byte offset `0x8000_0000`
([BufferOpsEmitter.cpp:308-311](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp#L308)),
against a descriptor with `numRecords = 0x7FFF_FFFE`
([BufferOpsEmitter.cpp:81](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp#L81)).
Every masked buffer load and store on RDNA2 already depends on that lane being dropped. The gate
added with the RDNA2 atomic subset (`7ee1cdac78`) nevertheless kept *masked atomics* on the
generic path, because the ISA guide states the range check but not the comparator's signedness:

* OOB_SELECT=3 (the mode this descriptor uses), stride = 0: `OOB = (offset + payload > NumRecords)`
  (RDNA2 ISA 8.1, Table 35).
* Note 3: "Load/store-format-\* instruction and atomics are range-checked all or nothing."

`0x8000_0000` only exceeds `0x7FFF_FFFE` under an unsigned compare; a signed compare would read it
as negative and let the atomic land at `base + 2 GiB`. Both the user mask and the redundant-thread
predicate reach the same `pred`
([LoadStoreOpToLLVM.cpp:1895](file:///home/trave/code/triton/third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp#L1895)),
so both shapes were in question.

## What the probe does

[docs/rdna2/conformance/rdna2_masked_buffer_atomic_probe.py](file:///home/trave/code/triton/docs/rdna2/conformance/rdna2_masked_buffer_atomic_probe.py)
makes a leak harmless instead of unmapped: it allocates an i32 tensor of `0x2000_0010` elements
(2 GiB + 64 B), writes every page with `zeros()`, and uses `base + 0x8000_0000` as a canary. If the
sentinel were treated as in-bounds, the atomic would land on the canary rather than 2 GiB outside
the allocation.

* `masked_or_kernel`: 256 elements over 128 threads, alternating mask. Active lanes target real
  addresses in the first block; masked lanes carry the same real addresses in the tensor, so a mask
  that was silently treated as all-true is caught as well.
* `redundant_or_kernel`: 4 elements over 4 warps, no mask. 124 threads have no element of their
  own, so the only predicate source is `emitRedundantThreadPredicateNonNull`.
* Each arm runs in its own process with its own `TRITON_CACHE_DIR`; the knob
  `AMDGCN_ALLOW_MASKED_BUFFER_ATOMICS` (pass option `allow-masked-buffer-atomics`) selects the path.
* Teeth: after the checks, the probe plants `0xDEAD` at the canary and confirms the same check
  reports it dirty; controls run the block with mask all-true and all-false.

## Result (44/44 checks, both arms)

| arm | static gfx1030 (masked / redundant) | runtime gfx1031 (masked / redundant) | canary at `base + 2 GiB` |
| --- | --- | --- | --- |
| generic (`AMDGCN_ALLOW_MASKED_BUFFER_ATOMICS=0`) | 0 / 0 `buffer_atomic_or`, 2 / 1 `global_atomic` | 0 / 0 buffer, 2 / 1 global | clean |
| buffer (option on) | 2 / 1 `buffer_atomic_or`, 2 / 1× `0x80000000`, 0 global | 2 / 1 buffer, 0 global | clean |

* Active lanes are OR'd with the value; masked lanes' real targets and the canary are untouched.
* Redundant lanes: the 4 real elements are written, the other 124 threads' sentinel is dropped,
  canary clean, and the runtime kernel really emitted `buffer_atomic_or` (1 instruction, 1 sentinel).
* Old values: masked lanes return 0, not whatever sits at the canary (`0xDEAD` when planted).
* A/B: the generic and buffer arms agree on every value check.
* The canary check fails when a write is planted (proved, then restored).

## What this does not say

* Only `or` was exercised on the card. The range check is performed by the memory pipeline before
  the opcode's read-modify-write (ISA note 3 says atomics are checked "all or nothing"), so it is
  opcode-independent; the `and/xor/umin/umax/smin/smax` arms share the descriptor and the same
  `fillCommonArgsAtomics`. The 64-bit `_X2` forms were validated unmasked earlier and were not
  re-run masked here.
* It measures the drop, not the comparator. Whether the hardware forms the check as unsigned or as
  a wider add, the observable is the same: the lane does not access memory.
* One allocation at whatever VA the allocator returned. The point is that `base + 2 GiB` was
  *mapped* and still not written; it is not a claim about unmapped addresses.

## Decision

Inactive lanes now take the buffer path by default (`allow-masked-buffer-atomics = true`).
`AMDGCN_ALLOW_MASKED_BUFFER_ATOMICS=0` restores the previous generic lowering; the lit test pins
both arms, and both directions were mutation-checked (the default arm fails with the option off,
the conservative arm fails with it on).

## Reproduce

```
python3 docs/rdna2/admission/gpu_admission.py --tag masked-atomics --budget-mib 4096 \
    --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \
    PYTHONPATH=/home/trave/code/triton/python \
    python3 docs/rdna2/conformance/rdna2_masked_buffer_atomic_probe.py
```

The probe's parent spawns one child per arm. Admission reported
`EXCLUSIVE_COOPERATIVE_LOCK`, free VRAM 10940 MiB; the peak allocation is the 2 GiB canary buffer.
