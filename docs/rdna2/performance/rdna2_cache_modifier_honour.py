"""Does GFX10 actually honour the cache-policy aux bits Triton now emits for RDNA2?

Context: `getCtrlBitsForCacheModifierOnTarget` used to return 0 for RDNA2, so every
`.cg/.cs/.cv/.wt` was silently dropped. The fix routes RDNA2 to its own table (aux 5 for `.cg` and
`.cv` loads, 2 for `.cs` loads and stores, 4 for `.wt` stores). The lit test proves the value
reaches the intrinsic; this script answers the two questions the test cannot:

1. NUMERICS. A cache hint must never change results. Every modifier variant is compared
   against the same kernel compiled with no modifier, and against torch.
2. HONOUR. If the hardware ignored the policy bits, an L1-resident re-read loop would time
   identically with and without a hint. If they are honoured, a bypass hint must make that loop
   measurably slower. The loop must use a runtime trip count: a constexpr Python range is fully
   unrolled and LLVM hoists the loop-invariant loads, which makes the hint unobservable by
   construction. That is a deterministic signal that does not need performance counters.

The third case is the opposite direction: a streaming read larger than L1, where a hint should
cost nothing (and NT should not hurt), to confirm the change is not a regression for the common
`.cs` use.

Run under the admission wrapper.
"""

import argparse
import json
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

MODS = ["none", "cg", "cs", "cv"]


@triton.jit
def resident_read(X, Out, rep, BLOCK: tl.constexpr, STRIPS: tl.constexpr,
                  CM: tl.constexpr):
    """Cycle over STRIPS small strips `rep` times. The load address depends on the runtime loop
    variable, so the loads cannot be hoisted or CSE'd: an earlier version used a constexpr trip
    count, was fully unrolled, and LLVM folded the repeats into one load per element, which made
    the cache hint unobservable by construction. With SC0 honoured, the repeated reads of this
    L1-sized set must come from L2 and cost measurably more than the unmodified variant."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(0, rep):
        strip = (i & (STRIPS - 1)) * BLOCK
        if CM == "none":
            acc += tl.load(X + strip + offs)
        elif CM == "cg":
            acc += tl.load(X + strip + offs, cache_modifier=".cg")
        elif CM == "cs":
            acc += tl.load(X + strip + offs, cache_modifier=".cs")
        else:
            acc += tl.load(X + strip + offs, cache_modifier=".cv")
    tl.store(Out + offs, acc / rep)


@triton.jit
def streaming_read(X, Out, n_elem, BLOCK: tl.constexpr, CM: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    if CM == "none":
        v = tl.load(X + offs, mask=m)
    elif CM == "cg":
        v = tl.load(X + offs, mask=m, cache_modifier=".cg")
    elif CM == "cs":
        v = tl.load(X + offs, mask=m, cache_modifier=".cs")
    else:
        v = tl.load(X + offs, mask=m, cache_modifier=".cv")
    tl.store(Out + offs, v, mask=m)


@triton.jit
def write_mod(X, Out, n_elem, BLOCK: tl.constexpr, CM: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    v = tl.load(X + offs, mask=m)
    if CM == "none":
        tl.store(Out + offs, v, mask=m)
    elif CM == "cs":
        tl.store(Out + offs, v, mask=m, cache_modifier=".cs")
    else:
        tl.store(Out + offs, v, mask=m, cache_modifier=".wt")


def timed(fn, batch, reps, ramp):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    time.sleep(ramp)
    out = []
    for _ in range(reps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(batch):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / batch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--ramp", type=float, default=8.0)
    ap.add_argument("--out", default="/tmp/cmod.json")
    # Datasheet peak, unidirectional (RX 6700 XT: 16 Gbps x 192-bit = 384 GB/s). Only one-way
    # traffic is comparable to it: a copy moves bytes twice, so its summed figure must never be
    # read against this number.
    ap.add_argument("--spec-gbps", type=float, default=384.0)
    # The resident re-read is the honour discriminator. Four strips of --resident-block elements
    # form an L1-sized set; --resident-rep bounds the runtime loop so the loads cannot be hoisted.
    # If a near-cache bypass is real, the repeated reads must come from L2 and the gap should be a
    # multiple, not a few percent.
    ap.add_argument("--resident-block", type=int, default=256,
                    help="elements per strip; 4 strips are cycled (default 4 KiB working set)")
    ap.add_argument("--resident-rep", type=int, default=512,
                    help="runtime loop iterations over the strips")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    print(f"device={torch.cuda.get_device_name(0)} gcn={props.gcnArchName} "
          f"L1={getattr(props, 'l1_cache_size', '?')} L2={getattr(props, 'L2_cache_size', '?')}",
          flush=True)

    res = {}

    # ---- case 1: L1-resident re-reads (the honour signal) ----
    # The arm order is none, cg, cs, cv, none again: the trailing repeat of the unmodified
    # variant sizes the noise band, without which a 7% delta is not evidence of anything.
    BLOCK, REP = args.resident_block, args.resident_rep
    STRIPS = 4
    print(f"  resident arm: {STRIPS} strips x BLOCK={BLOCK} "
          f"({STRIPS*BLOCK*4/1024:.0f} KiB working set), {REP} runtime iterations", flush=True)
    x = torch.randn(STRIPS * BLOCK, device="cuda", dtype=torch.float32)
    o = torch.zeros(BLOCK, device="cuda", dtype=torch.float32)
    ref = None
    for arm, cm in enumerate(MODS + ["none_repeat"]):
        key = "none" if cm == "none_repeat" else cm
        o.zero_()
        resident_read[(1,)](x, o, REP, BLOCK=BLOCK, STRIPS=STRIPS, CM=key, num_warps=1)
        torch.cuda.synchronize()
        if ref is None:
            ref = o.clone()
            expect = x.reshape(STRIPS, BLOCK).mean(0)
            print(f"  numerics baseline (none) matches x: "
                  f"{torch.allclose(o, expect, rtol=1e-3, atol=1e-4)}", flush=True)
        agree = bool(torch.allclose(o, ref, rtol=1e-3, atol=1e-4))
        ms = statistics.median(timed(
            lambda key=key: resident_read[(1,)](x, o, REP, BLOCK=BLOCK, STRIPS=STRIPS,
                                                CM=key, num_warps=1),
            args.batch, args.reps, args.ramp))
        res[f"resident_{arm}_{cm}"] = {"median_ms": ms, "matches_none": agree}
        print(f"resident re-read  {cm:11s} median {ms*1000:8.2f} us  matches_none={agree}",
              flush=True)

    # ---- case 2: streaming read far bigger than L1 ----
    N = 1 << 25                                  # 128 MiB fp32
    xs = torch.randn(N, device="cuda", dtype=torch.float32)
    os_ = torch.empty(N, device="cuda", dtype=torch.float32)
    blk = 1024
    grid = (triton.cdiv(N, blk),)
    ref2 = None
    for cm in MODS:
        os_.zero_()
        streaming_read[grid](xs, os_, N, BLOCK=blk, CM=cm, num_warps=4)
        torch.cuda.synchronize()
        if ref2 is None:
            ref2 = xs.clone()
        agree = bool(torch.equal(os_, ref2))
        ms = statistics.median(timed(
            lambda cm=cm: streaming_read[grid](xs, os_, N, BLOCK=blk, CM=cm, num_warps=4),
            max(1, args.batch // 10), args.reps, args.ramp))
        gbps = (N * 4 / 1e9) / (ms / 1000)
        res[f"stream_{cm}"] = {"median_ms": ms, "gbps_oneway": gbps,
                               "pct_of_spec": 100 * gbps / args.spec_gbps,
                               "matches_none": agree}
        print(f"streaming read    CM={cm:5s} median {ms:8.3f} ms  {gbps:6.1f} GB/s one-way "
              f"({100 * gbps / args.spec_gbps:4.1f}% of {args.spec_gbps:g})  bitwise={agree}",
              flush=True)
    del xs, os_
    torch.cuda.empty_cache()

    # ---- case 3: store modifiers ----
    xd = torch.randn(N, device="cuda", dtype=torch.float32)
    od = torch.empty(N, device="cuda", dtype=torch.float32)
    ref3 = None
    for cm in ["none", "cs", "wt"]:
        od.zero_()
        write_mod[grid](xd, od, N, BLOCK=blk, CM=cm, num_warps=4)
        torch.cuda.synchronize()
        if ref3 is None:
            ref3 = xd.clone()
        agree = bool(torch.equal(od, ref3))
        ms = statistics.median(timed(
            lambda cm=cm: write_mod[grid](xd, od, N, BLOCK=blk, CM=cm, num_warps=4),
            max(1, args.batch // 10), args.reps, args.ramp))
        oneway = (N * 4 / 1e9) / (ms / 1000)
        res[f"store_{cm}"] = {"median_ms": ms, "gbps_oneway": oneway,
                              "gbps_summed": 2 * oneway,
                              "pct_of_spec": 100 * oneway / args.spec_gbps,
                              "matches_none": agree}
        print(f"copy w/ store cm  CM={cm:5s} median {ms:8.3f} ms  {oneway:6.1f} GB/s per "
              f"direction ({100 * oneway / args.spec_gbps:4.1f}% of {args.spec_gbps:g}), "
              f"{2 * oneway:6.1f} summed  bitwise={agree}", flush=True)
    del xd, od
    torch.cuda.empty_cache()

    first = res["resident_0_none"]["median_ms"]
    last = res["resident_4_none_repeat"]["median_ms"]
    band = abs(first - last) / min(first, last)
    print(f"\nresident A/A band (unmodified arm measured first and last): {first*1000:.2f} vs "
          f"{last*1000:.2f} us -> {band*100:.2f}%", flush=True)
    print("== resident re-read ratios vs the unmodified variant ==")
    for cm in MODS:
        m = res[f"resident_{MODS.index(cm)}_{cm}"]["median_ms"]
        flag = "  (within A/A)" if abs(m - first) / first <= band else "  (outside A/A)"
        print(f"  {cm:5s} {m*1000:8.2f} us   x{m/first:.3f}{flag}")
    json.dump({"args": vars(args), "res": res,
               "device": torch.cuda.get_device_name(0),
               "gcn": props.gcnArchName}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
