#!/usr/bin/env python3
"""RDNA2 conformance matrix: run bounded op-class subsets of the Triton suite on the real
gfx1031 card and record, per class, what actually passed.

Why this exists: the M0 audit left the runtime correctness of reduce (L5) and of atomics,
cache modifiers, fp8 conversion and local_alloc/swizzle (L6) unevidenced on RDNA2. Reading
the code cannot close those. This script spends one admitted GPU window on the op classes
that matter instead of on a pass count.

Guards, learned the hard way on this host:
  * the system python3 has a distro `triton 3.5.1` that is NOT this repository. The script
    aborts unless `triton` and `triton._C.libtriton` both resolve inside the source tree.
  * results must be labelled by the emitted target, not the physical part: HSA_OVERRIDE_GFX_VERSION
    makes Triton see gfx1030 on a gfx1031 card. Both are recorded.
  * each Triton specialization costs several seconds to compile, so selections stay narrow.

Invoke it through the admission wrapper, e.g.
  python3 docs/rdna2/admission/gpu_admission.py --tag p2-reduce --budget-mib 3072 \
      --mem-floor-mib 3500 --acquire run -- python3 docs/rdna2/rdna2_conformance_matrix.py
"""
import json
import os
import subprocess
import sys
import time

REPO = "/home/trave/code/triton"
PKG = os.path.join(REPO, "python")
sys.path.insert(0, PKG)

GROUPS = [
    # (label, pytest file, -k expression)
    ("reduce_wave32", "python/test/unit/language/test_core.py", "test_reduce1d"),
    ("atomics", "python/test/unit/language/test_core.py", "test_atomic_add or test_atomic_min or test_atomic_max or test_atomic_cas or test_atomic_xchg"),
    ("dot_fma_path", "python/test/unit/language/test_core.py", "test_dot and not ieee and not tf32"),
    ("fp8_conversion", "python/test/unit/language/test_core.py", "test_cast_fp8 or test_fp8"),
    ("layout_swizzle", "python/test/unit/language/test_core.py", "test_trans"),
    ("masked_ops", "python/test/unit/language/test_core.py", "test_load_store or test_mask_load_split or test_mask_store_split"),
]


def require_source_tree():
    import triton
    import triton._C.libtriton as L
    tp = os.path.dirname(triton.__file__)
    if not tp.startswith(REPO):
        raise SystemExit(f"ABORT: triton resolves to {tp}, not the source tree {PKG}")
    if PKG not in (getattr(L, "__file__", "") or ""):
        raise SystemExit(f"ABORT: libtriton resolves to {L.__file__}")
    return triton.__version__, L.__file__


def emitted_target():
    """What .amdgcn_target the compiler will actually emit for this device."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _probe(x_ptr):
        tl.store(x_ptr, 1.0)

    try:
        src = triton.compiler.ASTSource(fn=_probe, signature={"x_ptr": "*fp32"})
        compiled = triton.compile(src)
        asm = compiled.asm.get("amdgcn", "")
        tgt = [ln for ln in asm.splitlines() if ".amdgcn_target" in ln]
        return tgt[0].strip() if tgt else "no .amdgcn_target line"
    except Exception as exc:
        return f"probe failed: {exc!r}"


def run_group(label, path, kexpr, workdir):
    cmd = [sys.executable, "-m", "pytest", path, "-k", kexpr, "-q", "--tb=line",
           "-s", "-p", "no:cacheprovider", "--no-header"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    out = r.stdout + r.stderr
    last = [ln for ln in out.splitlines() if ln.strip()]
    return {
        "label": label,
        "k": kexpr,
        "rc": r.returncode,
        "seconds": round(time.time() - t0, 1),
        "summary": next((ln for ln in reversed(last)
                         if " passed" in ln or " failed" in ln or " no tests ran" in ln), last[-1] if last else ""),
        "failures": [ln for ln in last if ln.startswith("FAILED") or "E   " in ln][:12],
    }


def main():
    ver, libpath = require_source_tree()
    import torch
    report = {
        "triton_version": ver,
        "triton_package": PKG,
        "libtriton": libpath,
        "libtriton_mtime": time.strftime("%F %T", time.localtime(os.path.getmtime(libpath))),
        "physical_device": torch.cuda.get_device_name(0),
        "gcnArchName": torch.cuda.get_device_properties(0).gcnArchName,
        "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "emitted_target": emitted_target(),
        "torch": torch.__version__,
        "groups": [],
    }
    for label, path, kexpr in GROUPS:
        only = os.environ.get("Q21_GROUPS")
        if only and label not in only.split(","):
            continue
        print(f"[matrix] {label} ...", flush=True)
        g = run_group(label, path, kexpr, REPO)
        report["groups"].append(g)
        print(f"[matrix] {label}: rc={g['rc']} {g['seconds']}s :: {g['summary']}", flush=True)

    out = "/tmp/q21-p1/rdna2_conformance_matrix.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
