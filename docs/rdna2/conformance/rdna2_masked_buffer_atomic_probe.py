#!/usr/bin/env python3
"""Settle whether RDNA2 masked buffer atomics can use the predicate-off sentinel.

The buffer lowering turns a false predicate into byte offset 0x8000_0000 and relies on the
descriptor's range check (num_records = 0x7FFF_FFFE) to drop the lane. Every masked buffer
load and store on this target already works that way. For atomics the guide states the same
all-or-nothing range check (RDNA2 ISA 8.1, Table 35 note 3; OOB_SELECT=3 compares
offset + payload against numRecords), but the compare's signedness is not written down, so
`ConvertToBufferOps` keeps masked atomics on the generic path unless the opt-in
`AMDGCN_ALLOW_MASKED_BUFFER_ATOMICS` (pass option `allow-masked-buffer-atomics`) is set.

This probe emits that exact instruction and makes a leak harmless: the base of a >= 2 GiB
allocation is passed as the pointer, so a sentinel that is *not* dropped lands at
base + 0x8000_0000 -- the canary inside the allocation -- instead of 2 GiB out of bounds.
The masked lanes also carry real target addresses inside the first block, so a mask that was
silently treated as all-true is caught too.

It replaces an earlier fault-or-pass draft of this file, which relied on a small tensor and a
memory violation to prove suppression. A signed comparison does not subtract 2 GiB: the address
is base plus the raw 32-bit byte offset, so the leaked access would land at base + 2 GiB either
way. Mapping that address and canarying it discriminates the same branches without asking the
device to fault.

Run through the admission wrapper (the parent spawns one child per arm, each with its own
Triton cache dir):
  python3 docs/rdna2/admission/gpu_admission.py --tag masked-atomics --budget-mib 4096 \\
      --acquire run -- env -u HSA_OVERRIDE_GFX_VERSION \\
      PYTHONPATH=/home/trave/code/triton/python \\
      python3 docs/rdna2/conformance/rdna2_masked_buffer_atomic_probe.py
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import torch
import triton
import triton.language as tl

BLOCK = 256
VALUE = 0xDEAD
CANARY_ELEM = 0x8000_0000 // 4  # sentinel byte offset in i32 elements
BUF_ELEMS = CANARY_ELEM + 16
CACHE_ROOT = "/tmp/rdna2-masked-buffer-atomic"


@triton.jit
def masked_or_kernel(ptr, mask_ptr, out_ptr, BLOCK: tl.constexpr, VALUE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = tl.load(mask_ptr + offs) != 0
    old = tl.atomic_or(ptr + offs, VALUE, mask=m)
    tl.store(out_ptr + offs, old)


@triton.jit
def redundant_or_kernel(ptr, BLOCK: tl.constexpr, VALUE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_or(ptr + offs, VALUE)


def asm_counts(asm):
    return {
        "target": (re.search(r"\.amdgcn_target\s+\"([^\"]+)\"", asm) or [None, "?"])[1],
        "buffer_atomic_or": len(re.findall(r"buffer_atomic_or", asm)),
        "global_atomic": len(re.findall(r"global_atomic", asm)),
        "sentinel": len(re.findall(r"0x80000000", asm)),
    }


def compile_static(arch):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(
        fn=masked_or_kernel,
        signature={"ptr": "*i32", "mask_ptr": "*i8", "out_ptr": "*i32",
                   "BLOCK": "constexpr", "VALUE": "constexpr"},
        constexprs={"BLOCK": BLOCK, "VALUE": VALUE},
    )
    k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                       options={"num_warps": 4})
    return asm_counts(k.asm["amdgcn"])


def compile_static_redundant(arch):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(
        fn=redundant_or_kernel,
        signature={"ptr": "*i32", "BLOCK": "constexpr", "VALUE": "constexpr"},
        constexprs={"BLOCK": 4, "VALUE": VALUE},
    )
    k = triton.compile(src, target=GPUTarget("hip", arch, 32),
                       options={"num_warps": 4})
    return asm_counts(k.asm["amdgcn"])


def device_checks(arm):
    assert torch.cuda.is_available(), "no CUDA/HIP device"
    dev = torch.cuda.current_device()
    print(f"  device: {torch.cuda.get_device_name(dev)}", flush=True)

    mask = torch.tensor([(i % 2) == 0 for i in range(BLOCK)], dtype=torch.uint8,
                        device="cuda")
    out = torch.full((BLOCK,), -1, dtype=torch.int32, device="cuda")

    # zeros() writes every page, so a leaked sentinel hits committed memory.
    buf = torch.zeros(BUF_ELEMS, dtype=torch.int32, device="cuda")
    masked_or_kernel[(1,)](buf, mask, out, BLOCK=BLOCK, VALUE=VALUE, num_warps=4)
    torch.cuda.synchronize()

    kernel_cache = masked_or_kernel.device_caches[dev][0]
    kern = next(iter(kernel_cache.values()))
    counts = asm_counts(kern.asm["amdgcn"])
    del kernel_cache

    active = mask.bool()
    first = buf[:BLOCK]
    canary = buf[CANARY_ELEM:CANARY_ELEM + 16]
    checks = [
        ("active_lanes_written", bool((first[active] == VALUE).all()),
         f"n={int(active.sum())}"),
        ("inactive_lanes_untouched", bool((first[~active] == 0).all()),
         f"n={int((~active).sum())}"),
        ("masked_lane_targets_untouched", bool((first[~active] == 0).all()),
         "same block, real addresses"),
        ("sentinel_canary_clean", bool((canary == 0).all()),
         f"canary @byte 0x80000000, head={canary[:4].tolist()}"),
        ("old_value_active", bool((out[active] == 0).all()), ""),
        ("old_value_inactive", bool((out[~active] == 0).all()),
         "masked lanes return 0, not the canary's old value"),
    ]

    # Teeth: the same canary check must notice a write at that address.
    buf[CANARY_ELEM] = VALUE
    torch.cuda.synchronize()
    dirty = buf[CANARY_ELEM:CANARY_ELEM + 16]
    checks.append(("canary_check_has_teeth", bool((dirty != 0).any()),
                   f"planted 0x{VALUE:x} at the canary, saw {dirty[:4].tolist()}"))
    buf[CANARY_ELEM] = 0
    torch.cuda.synchronize()
    canary = buf[CANARY_ELEM:CANARY_ELEM + 16]
    checks.append(("canary_restored", bool((canary == 0).all()), ""))

    # Mask controls: all-true must write every lane, all-false must write none.
    first.zero_()
    out.fill_(-1)
    torch.cuda.synchronize()
    all_on = torch.ones(BLOCK, dtype=torch.uint8, device="cuda")
    masked_or_kernel[(1,)](buf, all_on, out, BLOCK=BLOCK, VALUE=VALUE, num_warps=4)
    torch.cuda.synchronize()
    checks.append(("control_mask_all_true", bool((buf[:BLOCK] == VALUE).all()), ""))

    first.zero_()
    torch.cuda.synchronize()
    all_off = torch.zeros(BLOCK, dtype=torch.uint8, device="cuda")
    masked_or_kernel[(1,)](buf, all_off, out, BLOCK=BLOCK, VALUE=VALUE, num_warps=4)
    torch.cuda.synchronize()
    checks.append(("control_mask_all_false", bool((buf[:BLOCK] == 0).all()), ""))

    # Redundant lanes: 4 elements over 4 warps leaves 124 threads with no element of their
    # own. The emitter predicates them off with the same sentinel, which is what the pass's
    # free-variable check guards; the mask is absent here so only that check can refuse.
    first.zero_()
    torch.cuda.synchronize()
    redundant_or_kernel[(1,)](buf, BLOCK=4, VALUE=VALUE, num_warps=4)
    torch.cuda.synchronize()
    red_cache = redundant_or_kernel.device_caches[dev][0]
    red_counts = asm_counts(next(iter(red_cache.values())).asm["amdgcn"])
    del red_cache
    checks.append(("redundant_lanes_written_4",
                   bool((buf[:4] == VALUE).all()), f"head={buf[:4].tolist()}"))
    checks.append(("redundant_lanes_no_extra_elements",
                   bool((buf[4:BLOCK] == 0).all()), ""))
    canary = buf[CANARY_ELEM:CANARY_ELEM + 16]
    checks.append(("redundant_sentinel_canary_clean",
                   bool((canary == 0).all()), "predicated-off lanes"))

    canary = buf[CANARY_ELEM:CANARY_ELEM + 16]
    checks.append(("sentinel_canary_clean_after_controls",
                   bool((canary == 0).all()), ""))

    del buf
    torch.cuda.empty_cache()
    return {"masked": counts, "redundant": red_counts}, checks


def run_child(arm):
    from triton import knobs

    allow = arm == "buffer"
    knobs.amd.allow_masked_buffer_atomics = allow
    print(f"arm={arm} allow_masked_buffer_atomics={allow}", flush=True)

    static = {arch: compile_static(arch) for arch in ("gfx1030", "gfx1031")}
    static_red = {arch: compile_static_redundant(arch)
                  for arch in ("gfx1030", "gfx1031")}
    for arch, c in static.items():
        print(f"  static {arch}: {c}", flush=True)
        print(f"  static {arch} redundant: {static_red[arch]}", flush=True)
    counts, checks = device_checks(arm)
    print(f"  runtime: {counts}", flush=True)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {arm}/{name}"
              + (f"  {detail}" if detail else ""), flush=True)

    want_buffer = allow
    checks.append(("static_gfx1030_instruction_choice",
                   (static["gfx1030"]["buffer_atomic_or"] > 0) == want_buffer,
                   (f"buffer_atomic_or={static['gfx1030']['buffer_atomic_or']} "
                    f"global_atomic={static['gfx1030']['global_atomic']}")))
    checks.append(("static_gfx1031_instruction_choice",
                   (static["gfx1031"]["buffer_atomic_or"] > 0) == want_buffer,
                   (f"buffer_atomic_or={static['gfx1031']['buffer_atomic_or']} "
                    f"global_atomic={static['gfx1031']['global_atomic']}")))
    checks.append(("runtime_instruction_choice",
                   (counts["masked"]["buffer_atomic_or"] > 0) == want_buffer,
                   (f"buffer_atomic_or={counts['masked']['buffer_atomic_or']} "
                    f"global_atomic={counts['masked']['global_atomic']}")))
    checks.append(("static_redundant_instruction_choice",
                   (static_red["gfx1030"]["buffer_atomic_or"] > 0) == want_buffer,
                   (f"buffer_atomic_or={static_red['gfx1030']['buffer_atomic_or']} "
                    f"global_atomic={static_red['gfx1030']['global_atomic']}")))
    checks.append(("runtime_redundant_instruction_choice",
                   (counts["redundant"]["buffer_atomic_or"] > 0) == want_buffer,
                   (f"buffer_atomic_or={counts['redundant']['buffer_atomic_or']} "
                    f"global_atomic={counts['redundant']['global_atomic']}")))
    if allow:
        checks.append(("runtime_sentinel_materialised",
                       counts["masked"]["sentinel"] > 0,
                       f"0x80000000 occurrences={counts['masked']['sentinel']}"))
        checks.append(("runtime_redundant_sentinel_materialised",
                       counts["redundant"]["sentinel"] > 0,
                       f"0x80000000 occurrences={counts['redundant']['sentinel']}"))
    else:
        # Guard against the atomic being optimized away: the generic arm must still
        # contain a global atomic, just not a buffer one.
        checks.append(("static_generic_atomic_present",
                       static["gfx1030"]["global_atomic"] > 0,
                       f"global_atomic={static['gfx1030']['global_atomic']}"))
        checks.append(("runtime_generic_atomic_present",
                       counts["masked"]["global_atomic"] > 0,
                       f"global_atomic={counts['masked']['global_atomic']}"))
        checks.append(("runtime_redundant_generic_atomic_present",
                       counts["redundant"]["global_atomic"] > 0,
                       f"global_atomic={counts['redundant']['global_atomic']}"))

    print("RESULT " + json.dumps({
        "arm": arm,
        "static": static,
        "static_redundant": static_red,
        "runtime": counts,
        "checks": [[n, bool(ok), d] for n, ok, d in checks],
    }))
    return 0 if all(ok for _, ok, _ in checks) else 1


def run_parent():
    results = {}
    status = 0
    for arm in ("generic", "buffer"):
        cache = os.path.join(CACHE_ROOT, arm)
        shutil.rmtree(cache, ignore_errors=True)
        env = dict(os.environ, TRITON_CACHE_DIR=cache)
        print(f"=== child arm={arm} cache={cache} ===", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", arm],
                           check=False, env=env, capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        if p.stderr.strip():
            sys.stdout.write(p.stderr[-4000:])
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"[FAIL] {arm}/child produced no RESULT (exit {p.returncode})")
            status = 1
            continue
        results[arm] = json.loads(line[-1][len("RESULT "):])
        status |= p.returncode

    if len(results) == 2:
        g, b = results["generic"], results["buffer"]
        gd = {n: ok for n, ok, _ in g["checks"]}
        bd = {n: ok for n, ok, _ in b["checks"]}
        same = (g["checks"] and b["checks"] and
                all(gd.get(n) == bd.get(n) for n in
                    ("active_lanes_written", "inactive_lanes_untouched",
                     "sentinel_canary_clean")))
        ab = same and gd.get("sentinel_canary_clean") and bd.get("sentinel_canary_clean")
        print(f"[{'PASS' if ab else 'FAIL'}] a_b_agreeing_semantics"
              "  generic and buffer arms agree on masked lanes and canary")
        status |= 0 if ab else 1
        print(f"[INFO] generic buffer_atomic_or="
              f"{g['runtime']['masked']['buffer_atomic_or']} "
              f"buffer buffer_atomic_or={b['runtime']['masked']['buffer_atomic_or']}")

    total = sum(len(r["checks"]) for r in results.values()) + (1 if len(results) == 2 else 0)
    bad = sum(1 for r in results.values() for _, ok, _ in r["checks"] if not ok)
    print(f"\n{total - bad}/{total} checks passed across {len(results)} arms")
    return 1 if status or bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=("generic", "buffer"))
    args = ap.parse_args()
    if args.child:
        return run_child(args.child)
    return run_parent()


if __name__ == "__main__":
    sys.exit(main())
