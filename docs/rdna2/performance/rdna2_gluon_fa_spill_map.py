"""Does a hand-pinned Gluon layout spill less than Triton's auto-layout on RDNA2?

RDNA2 fast attention is register-bound: at useful tiles the Triton frontend spills hundreds of VGPRs,
and the measured consequence was that the spill-free configs are the slow ones
(performance/rdna2-spill-cost-measured.md). Gluon is the layer where layouts are chosen by hand, and
it compiles for gfx1030 (docs/rdna2/gluon-can-express-rdna2-without-matrix-units.md), so the narrow
static question is: at equal geometry AND equal math, does pinning the layouts reduce spills?

Both arms are the same attention inner step -- QK^T, exp2 with a fixed softmax shift, PV -- and
neither carries the online softmax. That is a limitation of MY Gluon kernel, not of the target:
gl.max/gl.sum along an axis do compile for AMD through GluonASTSource (verified on gfx1100 and
gfx1030), but folding a row vector produced by a reduction over the score tile into the
accumulator's layout needs conversions I have not got right yet. The Triton arm drops its
reductions as well, so both sides do identical work -- which is why each arm's emitted dot count is
printed.

Static only: `vgpr_spill_count` says a kernel fits the register file; it says nothing about speed,
so this selects candidates for hardware, it never ranks them.

Usage: PYTHONPATH=./python python docs/rdna2/performance/rdna2_gluon_fa_spill_map.py [--arch gfx1030]
"""

import argparse
import re
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental import gluon
from triton.experimental.gluon import GluonASTSource, language as gl

SPILL = re.compile(r"\.vgpr_spill_count:\s*(\d+)")
NGPR = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")
DOTS = re.compile(r"\bv_dot\w*")
SK = 1024
THREADS = [8, 4]


@triton.jit
def fa_plain(Q, K, V, O, SQ, SKC, sm,
             BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    """Triton arm: auto-chosen layouts, identical math to the Gluon arm."""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * SQ + offs_d[None, :])
    acc = tl.zeros([BM, HD], tl.float32)
    for start in range(0, SKC, BN):
        n = start + tl.arange(0, BN)
        k = tl.load(K + n[None, :] * SQ + offs_d[:, None])
        s = tl.dot(q, k) * sm
        p = tl.math.exp2(s - 8.0)
        v = tl.load(V + n[:, None] * SQ + offs_d[None, :])
        acc = tl.dot(p.to(tl.float16), v, acc)
    tl.store(O + offs_m[:, None] * HD + offs_d[None, :], acc / SKC)


def blocked(m, n, warps, threads=(None, None), cta_axis=0, order=(1, 0)):
    """Blocked layout for an m x n tile, or None when it cannot tile the shape exactly.

    cta_axis picks which dimension the warps spread over; threads overrides the [8,4] default so
    the sweep can probe operand shapes rather than one hand-picked layout.
    """
    tw = list(THREADS if threads == (None, None) else threads)
    per_cta = [warps, 1] if cta_axis == 0 else [1, warps]
    spm = m // (tw[0] * per_cta[0])
    spn = n // (tw[1] * per_cta[1])
    if spm < 1 or spn < 1 or spm * tw[0] * per_cta[0] != m or spn * tw[1] * per_cta[1] != n:
        return None
    return gl.BlockedLayout(size_per_thread=[spm, spn], threads_per_warp=tw,
                            warps_per_cta=per_cta, order=list(order))


VARIANTS = [
    ("threads 8,4 warps@M", dict(threads=(8, 4), cta_axis=0)),
    ("threads 4,8 warps@M", dict(threads=(4, 8), cta_axis=0)),
    ("threads 4,8 warps@N", dict(threads=(4, 8), cta_axis=1)),
    ("threads 2,16 warps@M", dict(threads=(2, 16), cta_axis=0)),
    ("threads 16,2 warps@M", dict(threads=(16, 2), cta_axis=0)),
    ("threads 8,4 order01", dict(threads=(8, 4), cta_axis=0, order=(0, 1))),
]


def gluon_layouts_named(bm, bn, hd, warps, kw):
    la = blocked(bm, hd, warps, **kw)
    ls = blocked(bm, bn, warps, **kw)
    lq = blocked(bm, hd, warps, **kw)
    lv = blocked(bn, hd, warps, **kw)
    lk = blocked(hd, bn, warps, **kw)
    if None in (la, ls, lq, lv, lk):
        return None
    return dict(LQ=lq, LK=lk, LV=lv, LA=la, LS=ls,
                A0=gl.DotOperandLayout(operand_index=0, parent=ls, k_width=0),
                B1=gl.DotOperandLayout(operand_index=1, parent=ls, k_width=0),
                P0=gl.DotOperandLayout(operand_index=0, parent=la, k_width=0),
                V1=gl.DotOperandLayout(operand_index=1, parent=la, k_width=0))


@gluon.jit
def fa_gluon(Q, K, V, O, SQ, SKC, sm,
             BM: gl.constexpr, BN: gl.constexpr, HD: gl.constexpr,
             LQ: gl.constexpr, LK: gl.constexpr, LV: gl.constexpr,
             LA: gl.constexpr, LS: gl.constexpr,
             A0: gl.constexpr, B1: gl.constexpr, P0: gl.constexpr, V1: gl.constexpr):
    """Gluon arm: every layout pinned by the caller. QK^T comes from loading K already transposed
    through indexing (no transpose op); both matmuls use dot_fma, which RDNA2 lowers to packed
    v_dot2* instructions."""
    pid = gl.program_id(0)
    q_rows = gl.arange(0, BM, layout=gl.SliceLayout(1, LQ))
    q_ds = gl.arange(0, HD, layout=gl.SliceLayout(0, LQ))
    q = gl.load(Q + (pid * BM + q_rows)[:, None] * SQ + q_ds[None, :])

    k_ds = gl.arange(0, HD, layout=gl.SliceLayout(1, LK))
    k_n = gl.arange(0, BN, layout=gl.SliceLayout(0, LK))
    v_n = gl.arange(0, BN, layout=gl.SliceLayout(1, LV))
    v_ds = gl.arange(0, HD, layout=gl.SliceLayout(0, LV))

    acc = gl.zeros([BM, HD], gl.float32, layout=LA)
    for start in range(0, SKC, BN):
        k = gl.load(K + (start + k_n)[None, :] * SQ + k_ds[:, None])
        s = gl.dot_fma(gl.convert_layout(q, A0, assert_trivial=False),
                       gl.convert_layout(k, B1, assert_trivial=False),
                       gl.zeros([BM, BN], gl.float32, layout=LS)) * sm
        p = gl.exp2(s - 8.0)
        v = gl.load(V + (start + v_n)[:, None] * SQ + v_ds[None, :])
        acc = gl.dot_fma(gl.convert_layout(p.to(gl.float16), P0, assert_trivial=False),
                         gl.convert_layout(v, V1, assert_trivial=False), acc)
    o_rows = gl.arange(0, BM, layout=gl.SliceLayout(1, LA))
    o_ds = gl.arange(0, HD, layout=gl.SliceLayout(0, LA))
    gl.store(O + (pid * BM + o_rows)[:, None] * HD + o_ds[None, :], acc / SKC)


def gluon_layouts(bm, bn, hd, warps):
    la, ls = blocked(bm, hd, warps), blocked(bm, bn, warps)
    lq, lv, lk = blocked(bm, hd, warps), blocked(bn, hd, warps), blocked(hd, bn, warps)
    if None in (la, ls, lq, lv, lk):
        return None
    return dict(LQ=lq, LK=lk, LV=lv, LA=la, LS=ls,
                A0=gl.DotOperandLayout(operand_index=0, parent=ls, k_width=0),
                B1=gl.DotOperandLayout(operand_index=1, parent=ls, k_width=0),
                P0=gl.DotOperandLayout(operand_index=0, parent=la, k_width=0),
                V1=gl.DotOperandLayout(operand_index=1, parent=la, k_width=0))


def _stats(k):
    asm = k.asm["amdgcn"]
    sp, ng = SPILL.search(asm), NGPR.search(asm)
    return (int(sp.group(1)) if sp else 0), (int(ng.group(1)) if ng else 0), len(DOTS.findall(asm))


def triton_measure(arch, bm, bn, hd, warps):
    src = ASTSource(
        fn=fa_plain,
        signature={"Q": "*fp16", "K": "*fp16", "V": "*fp16", "O": "*fp16",
                   "SQ": "i32", "SKC": "i32", "sm": "fp32"},
        constexprs={"BM": bm, "BN": bn, "HD": hd},
        attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]],
               (2,): [["tt.divisibility", 16]], (3,): [["tt.divisibility", 16]]},
    )
    return _stats(triton.compile(src, target=GPUTarget("hip", arch, 32),
                                 options={"num_warps": warps, "num_stages": 1}))


def gluon_measure(arch, bm, bn, hd, warps, kw=None):
    layouts = (gluon_layouts_named(bm, bn, hd, warps, kw) if kw
               else gluon_layouts(bm, bn, hd, warps))
    if layouts is None:
        raise ValueError("layout does not tile the shape")
    src = GluonASTSource(
        fn=fa_gluon,
        signature={"Q": "*fp16", "K": "*fp16", "V": "*fp16", "O": "*fp16",
                   "SQ": "i32", "SKC": "i32", "sm": "fp32"},
        constexprs={"BM": bm, "BN": bn, "HD": hd, **layouts},
        attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]],
               (2,): [["tt.divisibility", 16]], (3,): [["tt.divisibility", 16]]},
    )
    return _stats(triton.compile(src, target=GPUTarget("hip", arch, 32),
                                 options={"num_warps": warps, "num_stages": 1}))


def sweep_layouts(arch, configs):
    """Best pinned layout per tile vs the compiler's own choice: sizes the gap a heuristic fix
    would close, and shows whether one hand-picked layout is enough."""
    print(f"arch={arch}  best of {len(VARIANTS)} pinned layouts per tile vs auto")
    print(f"{'BMxBN':>9} {'HD':>4} {'W':>2} | {'auto':>6} | {'best':>6} {'gain%':>7}  best variant")
    for bm, bn, hd, w in configs:
        try:
            tsp, _, tdots = triton_measure(arch, bm, bn, hd, w)
        except Exception as e:  # noqa: BLE001
            print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2} | auto FAILS {' '.join(str(e).split())[-60:]}")
            continue
        best, best_name, best_dots = tsp, "auto", tdots
        per_variant = []
        for name, kw in VARIANTS:
            try:
                gsp, _, gdots = gluon_measure(arch, bm, bn, hd, w, kw)
            except Exception:  # noqa: BLE001 - variant may not tile the shape
                continue
            if gdots != tdots:
                continue  # not the same work: cannot be compared
            per_variant.append((gsp, name, gdots))
            if gsp < best:
                best, best_name = gsp, name
        gain = 100.0 * (tsp - best) / tsp if tsp else 0.0
        detail = "  ".join(f"{n}={v}" for v, n, _ in sorted(per_variant))
        print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2} | {tsp:>6} | {best:>6} {gain:>6.1f}%  "
              f"{best_name}   [{detail}]")
    return 0


def time_both(bm, bn, hd, warps, threads, reps=5, batch=40, ramp=8.0):
    """Measure the pinned-vs-auto prediction on real hardware.

    Same math, same tile, only the layout differs. Small kernels here are launch-latency bound,
    so each timing bracket holds `batch` launches behind a clock ramp, and the auto arm runs twice
    to size the A/A band -- without that a few percent is not evidence of anything.
    """
    import statistics
    import time

    import torch

    global THREADS
    THREADS = list(threads)

    sk = 4096
    q = torch.randn(bm, hd, device="cuda", dtype=torch.float16)
    k = torch.randn(sk, hd, device="cuda", dtype=torch.float16)
    v = torch.randn(sk, hd, device="cuda", dtype=torch.float16)

    def bench(fn, grid, args, num_warps):
        fn[grid](*args, num_warps=num_warps)
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(batch):
                fn[grid](*args, num_warps=num_warps)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) / batch * 1e3)
        return statistics.median(out)

    def triton_run():
        o = torch.empty(bm, hd, device="cuda", dtype=torch.float32)
        args = (q, k, v, o, hd, sk, 0.125)
        return bench(fa_plain, (1,), args + (bm, bn, hd), warps), o

    def gluon_run():
        L = gluon_layouts(bm, bn, hd, warps)
        o = torch.empty(bm, hd, device="cuda", dtype=torch.float32)
        args = (q, k, v, o, hd, sk, 0.125)
        return bench(fa_gluon, (1,), args + (bm, bn, hd, *L.values()), warps), o

    t_ramp = time.perf_counter()
    while time.perf_counter() - t_ramp < ramp:
        triton_run()
    auto_ms, auto_out = triton_run()
    glu_ms, glu_out = gluon_run()
    auto2_ms, _ = triton_run()
    band = abs(auto2_ms - auto_ms) / auto_ms * 100
    same = bool(torch.allclose(auto_out, glu_out, rtol=1e-2, atol=1e-2))
    print(f"  {bm}x{bn} hd={hd} w={warps}: auto {auto_ms:.3f} ms | gluon {glu_ms:.3f} ms "
          f"({100 * (glu_ms - auto_ms) / auto_ms:+.1f}%) | A/A {band:.2f}% | results_agree={same}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true",
                    help="run the pinned-vs-auto A/B on the GPU (needs an admitted device)")
    ap.add_argument("--sweep-layouts", action="store_true",
                    help="best of a layout family per tile vs auto, instead of one fixed layout")
    ap.add_argument("--arch", default="gfx1030")
    ap.add_argument("--threads", default="8,4",
                    help="threads_per_warp for the pinned layouts, e.g. 8,4 or 4,8")
    args = ap.parse_args()
    global THREADS
    THREADS = [int(x) for x in args.threads.split(",")]

    if args.time:
        import torch
        assert torch.cuda.is_available(), "--time needs a GPU and an admitted lock"
        for bm, bn, hd, w, th in [(64, 64, 128, 2, [4, 8]), (64, 64, 128, 1, [4, 8]),
                                  (64, 32, 128, 1, [4, 8]), (64, 64, 64, 2, [4, 8])]:
            time_both(bm, bn, hd, w, th)
        print("Spill-count predictions are candidates only: this card has already shown a "
              "spill-free config running slower than a spilling one.")
        return 0

    if args.sweep_layouts:
        configs = [(bm, bn, hd, w)
                   for bm, bn in [(64, 64), (64, 32)]
                   for hd in [128]
                   for w in [1, 2] if bm >= w * 32]
        return sweep_layouts(args.arch, configs)

    print(f"arch={args.arch}  pinned threads_per_warp={THREADS}  both arms: same math (fixed shift, no reductions), SK={SK}")
    print("static register pressure only -- selects candidates, never ranks speed\n")
    print(f"{'BMxBN':>9} {'HD':>4} {'W':>2} | {'t_spill':>7} {'t_vdot':>6} {'t_vgpr':>6} | "
          f"{'g_spill':>7} {'g_vdot':>6} {'g_vgpr':>6}  verdict")
    tally = {}
    for bm, bn in [(32, 32), (64, 32), (64, 64)]:
        for hd in [64, 128]:
            for w in [1, 2]:
                if bm < w * 32:
                    continue
                try:
                    tsp, tngp, tdots = triton_measure(args.arch, bm, bn, hd, w)
                except Exception as e:  # noqa: BLE001
                    tally["triton fails"] = tally.get("triton fails", 0) + 1
                    print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2} | triton FAILS "
                          f"{' '.join(str(e).split())[-70:]}")
                    continue
                try:
                    gsp, gngp, gdots = gluon_measure(args.arch, bm, bn, hd, w)
                except Exception as e:  # noqa: BLE001
                    msg = " ".join(str(e).split())
                    key = "layout skipped" if "does not tile" in msg else "gluon fails"
                    tally[key] = tally.get(key, 0) + 1
                    print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2} | {tsp:>7} {tdots:>6} {tngp:>6} | "
                          f"{key:>10}  {msg[-60:]}")
                    continue
                note = ("not comparable" if tdots != gdots else
                        "GLUON BETTER" if gsp < tsp else
                        "tie" if gsp == tsp else "triton better")
                tally[note] = tally.get(note, 0) + 1
                print(f"{bm:>4}x{bn:<4} {hd:>4} {w:>2} | {tsp:>7} {tdots:>6} {tngp:>6} | "
                      f"{gsp:>7} {gdots:>6} {gngp:>6}  {note}")
    print("\ntally:", "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print("Rows marked 'not comparable' have different dot counts between arms, so no spill "
          "conclusion can be drawn from them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
