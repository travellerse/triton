"""Do the RDNA2 cache-policy aux bits change actual cache behaviour? Counter version.

Timing could not answer this: the two admitted runs of the resident-read benchmark disagreed in
sign and the same config drifted 10.8% between runs. Counters count requests instead of
nanoseconds, so they can:

  arm     aux  ISA behaviour (tables 38/39)            expected GL2C traffic
  none     0  L0 hit, L1 hit-LRU, L2 LRU               lowest  (re-reads served by L1/L0)
  cg       5  L0 bypassed, L1 miss-evict, L2 LRU       ~REP x higher (every re-read goes to L2)
  cv       5  same bits as cg                          identical to cg, by construction
  cs       2  L1 hit-LRU, L2 stream                    same request count as none (policy, not bypass)

So this is a *mechanism* test with a falsifiable prediction: if GL2C traffic does not rise for the
bypass arms, the bits are encoded but inert on this part, and the honest verdict is "honours the
request at the ISA level, hardware effect unproven". If cv and cg count identically, that confirms
the two rows really are the same encoding.

Why this is a separate script from rdna2_cache_modifier_pmc.py: that attempt hung under rocprofv3,
because Triton was still running LLVM compilation inside the profiled process. Here the kernels are
compiled and launched FIRST (arm --populate), so the profiled process (--profile) only loads a
cached hsaco and submits it.

Driven from the shell, one rocprofv3 invocation per arm, under the GPU admission wrapper:

  python3 rdna2_cache_policy_pmc.py --populate
  for cm in none cg cs cv; do
    rocprofv3 --pmc GL2C_REQUEST GL2C_HIT GL2C_MISS -f csv -o /tmp/pmc_$cm -- \
      python3 rdna2_cache_policy_pmc.py --profile $cm
  done
"""

import argparse
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdna2_cache_modifier_honour import resident_read  # noqa: E402

BLOCK, REP, WARPS = 1024, 64, 1
REF = "/tmp/rdna2_cmod_pmc_ref_none.pt"


def run_arm(cm, launches):
    """Launch the arm `launches` times; return its output as a CPU tensor.

    The kernel sums REP identical loads then divides by REP, so the value is only reproducible
    against the same math, not against the input. Every arm runs the same arithmetic, so the
    cross-arm invariant is bit-for-bit equality -- which is what a cache hint must satisfy.
    """
    x = torch.full((BLOCK,), 1.5, device="cuda", dtype=torch.float32)
    o = torch.zeros(BLOCK, device="cuda", dtype=torch.float32)
    for _ in range(launches):
        resident_read[(1,)](x, o, N=BLOCK, BLOCK=BLOCK, REP=REP, CM=cm, num_warps=WARPS)
    torch.cuda.synchronize()
    return o.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--populate", action="store_true",
                    help="compile and launch every arm so the disk cache is warm")
    ap.add_argument("--profile", choices=["none", "cg", "cs", "cv"],
                    help="launch a single arm, assuming its kernel is already cached")
    ap.add_argument("--launches", type=int, default=4)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no device"
    if args.populate:
        ref = run_arm("none", args.launches)
        torch.save(ref, REF)
        for cm in ["none", "cg", "cs", "cv"]:
            got = run_arm(cm, args.launches)
            print(f"populated cm={cm} bitwise_matches_none={bool(torch.equal(got, ref))}",
                  flush=True)
    elif args.profile:
        got = run_arm(args.profile, args.launches)
        ref = torch.load(REF, weights_only=True, map_location="cpu")
        print(f"profile cm={args.profile} launches={args.launches} "
              f"bitwise_matches_reference={bool(torch.equal(got, ref))}", flush=True)
    else:
        ap.error("pick --populate or --profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
