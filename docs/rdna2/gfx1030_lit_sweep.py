"""Re-run every AMD lit RUN line under gfx-arch=gfx1030 and report the deltas.

Purpose: find places where RDNA2 lowering differs from, or is broken relative to, the
arches upstream actually exercises (gfx942/gfx950/gfx1010-era/gfx110x/gfx115x/gfx1250...).
Purely static: triton-opt needs no GPU.

Each RUN line is executed verbatim, with only the gfx-arch= value rewritten, so a
"FAIL" means *this test's own* expectations do not hold for RDNA2 -- which is either a
gap to fix or a legitimately arch-specific test. Both outcomes get recorded.
"""

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--build", default="/home/trave/code/triton/build/cmake.linux-x86_64-cpython-3.14")
_ap.add_argument("--target", default="gfx1030",
                 help="arch to substitute into every gfx-arch= knob (gfx1031 for the "
                      "native ISA behind HSA_OVERRIDE_GFX_VERSION)")
_args = _ap.parse_args()
BUILD = pathlib.Path(_args.build)
SRC = pathlib.Path("/home/trave/code/triton/test")
TARGET = _args.target
DIRS = ["Conversion", "TritonGPU", "TritonAMDGPUTransforms", "Triton", "Gluon", "Proton"]

RUN = re.compile(r"^//\s*RUN:\s*(.*)$")
APPLY = re.compile(r"triton-opt\b")
ARCH = re.compile(r"gfx-arch=gfx[\w-]+")


def sh(cmd):
    # shell=True only because lit RUN lines contain a pipe; the command text comes from
    # this repository's own .mlir RUN lines, never from external or user data.
    env = dict(os.environ, PATH=f"{BUILD / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(cmd, shell=True, cwd=BUILD, capture_output=True, text=True,
                          env=env)


def main():
    total = rewritten = passed = 0
    recs = []
    for d in DIRS:
        base = SRC / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.mlir")):
            text = f.read_text()
            runs = [m.group(1) for line in text.splitlines()
                    if (m := RUN.match(line))]
            if not runs or not APPLY.search(" ".join(runs)):
                continue
            rel = str(f)
            for r in runs:
                if "FileCheck" not in r:
                    continue
                total += 1
                if not ARCH.search(r):
                    continue
                origin = ARCH.search(r).group(0).split("=")[1]
                new = ARCH.sub(f"gfx-arch={TARGET}", r)
                if new == r:
                    continue
                rewritten += 1
                cmd = new.replace("%s", shlex.quote(rel))
                res = sh(cmd)
                if res.returncode == 0:
                    passed += 1
                    recs.append((rel, origin, True, ""))
                else:
                    err = next((l for l in res.stderr.splitlines()
                                if "error" in l), "?")
                    if err == "?":
                        err = (res.stderr.strip().splitlines() or ["no stderr"])[:1][0]
                    recs.append((rel, origin, False, err[:200]))

    print(f"RUN lines rewritten to {TARGET}: {rewritten} of {total} arch-aware runs; "
          f"pass={passed} fail={rewritten - passed}")

    def short(p):
        return p.split("/")[-1]

    # A test that only ever runs on RDNA3 is the interesting comparison: RDNA2 shares the
    # generic lowering there, so a failure is a real difference rather than a test written
    # for another ISA.
    rdna3_origin = [r for r in recs if not r[2] and r[1].startswith(("gfx110", "gfx115",
                                                                    "gfx117", "gfx101"))]
    # Files whose names announce an ISA feature RDNA2 does not have (async copy / buffer
    # atomics / transposed LDS / warp id) cannot be expected to pass; they get their own
    # bucket so the signal list stays readable.
    gated = re.compile(r"atomic|async|ds_transpose|warp_id|tdm|mbarrier|cluster|pingpong",
                       re.I)
    known = [r for r in rdna3_origin if gated.search(r[0])]
    signal = [r for r in rdna3_origin if r not in known]
    other = [r for r in recs if not r[2] and r not in rdna3_origin]
    print(f"\n--- rewritten from RDNA3/RDNA1 origins and NOT explained by a known "
          f"RDNA2-absent feature ({len(signal)})")
    for rel, origin, _, err in signal:
        print(f"  {short(rel)} (origin {origin}): {err}")
    print(f"\n--- rewritten from RDNA3/RDNA1, feature-gated files ({len(known)})")
    for rel, origin, _, _ in known:
        print(f"  {short(rel)} (origin {origin})")
    print(f"\n--- failures rewritten from other arches ({len(other)}) "
          f"= mostly ISA-specific tests, listed by origin")
    by_origin = {}
    for rel, origin, _, _ in other:
        by_origin.setdefault(origin, []).append(short(rel))
    for origin in sorted(by_origin):
        print(f"  {origin}: {' '.join(sorted(set(by_origin[origin])))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
