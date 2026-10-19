"""Does a masked buffer atomic really get suppressed on RDNA2?

Background this test exists to settle
-------------------------------------
The M1 record kept RDNA2's integer buffer atomics unmasked-only because GFX10's atomic out-of-bounds
suppression was treated as undocumented. Two things have since been established statically:

* the RDNA2 ISA guide does document it -- out-of-range "writes are ignored (dropped)", and note 3
  of the Range Checking section says load/store-format instructions *and atomics* are range-checked
  "all or nothing";
* Triton already relies on the mechanism in shipped code: on RDNA/RDNA4 the descriptor is built with
  OOB_select=3, stride 0, numRecords INT_MAX-1, and masked-off lanes are sent offset 0x80000000
  (BufferOpsEmitter.cpp). RDNA3 and RDNA4 use the same flags and DO have buffer atomics enabled, so
  masked buffer atomics already run through this path on shipping targets; what is unproven is only
  whether the *comparison* behaves the same on gfx10.

What this probe adds over the static evidence
---------------------------------------------
A guard region cannot discriminate the open question: a lane that escaped suppression writes at
base + 0x80000000, or base - 2 GiB if the offset compared signed -- outside the tensor either way,
so in-buffer sentinels stay clean under both readings. The observable is a memory violation, since
base - 2 GiB is essentially never mapped on a 12 GiB card. So this is a fault-or-pass test.

The signed-comparison branch is already largely falsified without it: RDNA2's masked *buffer loads*
use the same 0x80000000 sentinel against the same descriptor field, and under a signed reading they
would return garbage instead of the hardware's out-of-range zero, which the on-device conformance
probe (43/43, masked loads with other=0 included) would have caught. This probe makes the atomics
themselves a measured fact instead of an inference, and it is the last item between the current gate
and enabling the integer subset.

For contrast, on gfx1100 the same source lowers to `buffer_atomic_and_b32` with that sentinel-fed
address, so the mechanism is live on a shipping target today; on gfx1030 it lowers to
EXEC-masked branching plus a CAS loop, which is why the gate currently costs 8 atomics where RDNA3
spends 4.

Run under the GPU admission wrapper. Requires buffer atomics to actually be emitted on RDNA2, which
today they are not -- see the note at the bottom; on gfx1100 the same script exercises the live path.
"""

import argparse
import sys

import torch
import triton
import triton.language as tl

N = 4096
GUARD = 1024


@triton.jit
def masked_atomic_and(P, X, n, BLOCK: tl.constexpr):
    """Half the lanes masked off. Integer AND is inside the M1M candidate subset, so this is the
    op that would be lowered to a buffer atomic once RDNA2's gate is widened."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(X + offs, mask=m, other=0)
    tl.atomic_and(P + offs, v, mask=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="informational only; the device decides")
    args = ap.parse_args()

    torch.manual_seed(0)
    total = N + GUARD
    p = torch.full((total,), -1, device="cuda", dtype=torch.int32)
    x = torch.randint(-2**31, 2**31 - 1, (total,), device="cuda", dtype=torch.int32)
    expected = p[:N].cpu().clone()
    guard_before = p[N:].cpu().clone()

    masked_atomic_and[(1,)](p, x, N // 2, BLOCK=N, num_warps=4)
    torch.cuda.synchronize()

    after = p[:N].cpu()
    touched = (after != expected)
    guard_moved = bool((p[N:].cpu() != guard_before).any())
    print(f"arch={getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '?')}")
    print(f"lanes written: {int(touched.sum())} (expected {N // 2} = the unmasked half)")
    print(f"masked-off lanes unchanged: "
          f"{bool((~touched)[N // 2:].all())}; guard region untouched: {not guard_moved}")
    print("If this prints at all, the kernel did not fault: under a signed offset comparison it "
          "would have written base-2GiB and raised a memory violation. Pair it with the ISA note "
          "that out-of-range writes are dropped to conclude suppression is real on this part.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
