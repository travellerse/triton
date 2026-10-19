#!/usr/bin/env python3
"""P5 probe: does RDNA2 get the vec-128 LDS tile, and what changes in the lowering?

Recipe taken from m1-implementation-report.md section 14.2, which is the only shape family
that produced a discriminating result there: 128-bit contiguous register groups on BOTH
sides (sizePerThread=[1,4], f32) and different warpsPerCTA, so the conversion must move data
across warps through LDS. Anything that only rearranges data within a warp is diverted to
transferWithinWarp, emits no LDS access at all, and reads as a false "no difference".

Prints, per arch: ttg.shared bytes, lowering line count, and the widths of the LDS accesses.
Run it from the triton repo root; it only needs the built triton-opt.
"""
import subprocess
import sys
import os
import re

BUILD_DIR = "build/cmake.linux-x86_64-cpython-3.14"
OPT = os.path.join(BUILD_DIR, "bin", "triton-opt")

TEMPLATE = """
#src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @xwarp_vec128(%arg0: tensor<128x128xf32, #src>) -> tensor<128x128xf32, #dst> {
    %0 = ttg.convert_layout %arg0 : tensor<128x128xf32, #src> -> tensor<128x128xf32, #dst>
    tt.return %0 : tensor<128x128xf32, #dst>
  }
}
"""

ARCHES = ["gfx1030", "gfx1031", "gfx1100", "gfx1170", "gfx942", "gfx90a"]


def lower(arch):
    path = f"/tmp/q21-p1/p5_{arch}.mlir"
    with open(path, "w") as f:
        f.write(TEMPLATE)
    cmd = [OPT, path, "-split-input-file", "--allocate-shared-memory",
           f"--convert-triton-amdgpu-to-llvm=gfx-arch={arch}",
           "--convert-builtin-func-to-llvm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def summarize(arch, out):
    shared = re.findall(r'ttg\.shared" = (\d+)', out)
    # LDS access widths show up as vector loads/stores from the shared address space
    vecs = sorted(set(re.findall(r'\bv(?:(?:8|16|32))?\s*:?\s*vector<(\d+)x[fbi]\d+>', out)))
    b128 = ("vector<4xf32>" in out or "vector<8xf16>" in out or
            "vector<4xi32>" in out or "ds_read_b128" in out or "ds_write_b128" in out)
    return {
        "arch": arch,
        "lines": len(out.splitlines()),
        "ttg_shared": shared[0] if shared else "n/a",
        "has_vec128_lds_access": b128,
        "vec_widths": vecs,
    }


def main():
    if not os.path.exists(OPT):
        print(f"missing {OPT}: run ninja first", file=sys.stderr)
        return 2
    rows, outputs = [], {}
    for arch in ARCHES:
        r = lower(arch)
        if r.returncode != 0:
            print(f"{arch}: triton-opt rc={r.returncode}: {r.stderr.strip().splitlines()[:2]}")
            continue
        outputs[arch] = r.stdout
        rows.append(summarize(arch, r.stdout))

    print(f"{'arch':9s} {'lines':>6s} {'ttg.shared':>11s} {'vec128':>7s}  vec_widths")
    for r in rows:
        print(f"{r['arch']:9s} {r['lines']:6d} {r['ttg_shared']:>11s} "
              f"{str(r['has_vec128_lds_access']):>7s}  {r['vec_widths']}")

    def get(a, k):
        for r in rows:
            if r["arch"] == a:
                return r[k]
        return None

    print()
    for pair in [("gfx1030", "gfx1031"), ("gfx1100", "gfx1170")]:
        x, y = pair
        same = get(x, "lines") == get(y, "lines") and get(x, "vec_widths") == get(y, "vec_widths")
        print(f"{x} {'==' if same else '!='} {y}")
    print(f"P5 claim under test: gfx1030/gfx1031 should now MATCH gfx1100 (RDNA3 tile group)")
    ok = get("gfx1030", "lines") == get("gfx1100", "lines")
    print(f"result: {'P5 RESTORED - RDNA2 lowers like RDNA3' if ok else 'P5 STILL ABSENT - differs from RDNA3'}")
    for arch in ("gfx1030", "gfx1100"):
        if arch in outputs:
            with open(f"/tmp/q21-p1/p5_out_{arch}.mlir", "w") as f:
                f.write(outputs[arch])
    return 0


if __name__ == "__main__":
    sys.exit(main())
