"""Hardware-counter evidence for whether GFX10 honours the RDNA2 cache-policy aux bits.

Timing could not answer this: two admitted runs of the same resident-read benchmark disagreed in
sign and the identical config drifted 10.8% between runs. The counters can, because they count
requests rather than nanoseconds:

  * TCP_UTCL1_REQUEST  -- requests that actually reached the L1 (GL1) pipe
  * GL2C_HIT / GL2C_MISS -- what the L2 saw

If the aux value that Triton now emits for .cg/.cs really bypasses L1, the L1 request count for
that variant must collapse relative to the unmodified variant, while L2 traffic stays the same
(both read the same bytes the same number of times). If the counts are identical, the bits are
encoded but inert on this part, and the honest summary is "honours the request at the ISA level,
hardware effect unproven".

One process per modifier, because the four launches share a kernel name and would aggregate.

Driven from the shell, one rocprofv3 invocation per --cm value; run under the admission wrapper.
"""

import argparse
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdna2_cache_modifier_honour import resident_read  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", required=True, choices=["none", "cg", "cs", "cv"])
    ap.add_argument("--launches", type=int, default=3)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    BLOCK, REP = 1024, 64
    x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    o = torch.zeros(BLOCK, device="cuda", dtype=torch.float32)
    for _ in range(args.launches):
        resident_read[(1,)](x, o, N=BLOCK, BLOCK=BLOCK, REP=REP, CM=args.cm, num_warps=1)
    torch.cuda.synchronize()
    ok = bool(torch.allclose(o, x, rtol=1e-3, atol=1e-4))
    print(f"cm={args.cm} result_matches_input={ok}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
