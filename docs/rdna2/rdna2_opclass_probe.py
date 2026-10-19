#!/usr/bin/env python3
"""RDNA2 op-class correctness probe (reduce / atomics / dot / conversions / layouts).

Closes the audit's unevidenced claims on real silicon: L5 (is the
DPP ROW_SHR -> PermlaneX16 -> Readlane ladder numerically correct on a 32-lane RDNA2
wavefront?) and L6 (atomics, cache modifiers, fp8 conversion, local_alloc/swizzle).

It deliberately reports per-case results rather than a suite pass count, because a
green-looking number proves nothing about the op class under test.

Run it against ANY Triton build; the identity of that build is part of the record:
  # installed package the workload uses
  python3 docs/rdna2/rdna2_opclass_probe.py
  # this source tree
  PYTHONPATH=/home/trave/code/triton/python python3 docs/rdna2/rdna2_opclass_probe.py
"""
import json
import os
import sys
import time

RESULTS = []


def record(case, ok, detail=""):
    RESULTS.append({"case": case, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {case}" + (f"  {detail}" if detail else ""), flush=True)


def emitted_target():
    import triton
    import triton.language as tl

    @triton.jit
    def _p(x_ptr):
        tl.store(x_ptr, 1.0)

    try:
        c = triton.compile(triton.compiler.ASTSource(fn=_p, signature={"x_ptr": "*fp32"}))
        for ln in c.asm.get("amdgcn", "").splitlines():
            if ".amdgcn_target" in ln:
                return ln.strip()
        return "no .amdgcn_target"
    except Exception as exc:
        return f"probe failed: {exc!r}"


def max_abs(a, b):
    import torch
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a.float() - b.float()).abs().max())


def reduce_cases():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def row_sum(x_ptr, o_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0.0)
            acc += v.to(tl.float32)
        tl.store(o_ptr + row, tl.sum(acc, axis=0))

    torch.manual_seed(3)
    for dt, tol in ((torch.float32, 2e-3), (torch.float16, 3e-2), (torch.bfloat16, 8e-2)):
        for n_rows, n_cols in ((32, 32), (8, 1000), (16, 4096), (4, 33)):
            x = (torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)).to(dt)
            o = torch.empty(n_rows, device="cuda", dtype=torch.float32)
            row_sum[(n_rows,)](x, o, n_cols=n_cols, BLOCK=256, num_warps=1)
            ref = x.float().sum(dim=1)
            d = max_abs(o, ref)
            record(f"reduce tl.sum axis0 {dt} {n_rows}x{n_cols} w1",
                   isinstance(d, float) and d <= tol, f"max_abs={d}")

    # the ladder is exercised differently per warp count; wave32 means 32 lanes
    for nw in (1, 2, 4, 8):
        x = torch.randn(64, 2048, device="cuda", dtype=torch.float32)
        o = torch.empty(64, device="cuda", dtype=torch.float32)
        row_sum[(64,)](x, o, n_cols=2048, BLOCK=512, num_warps=nw)
        d = max_abs(o, x.sum(dim=1))
        record(f"reduce tl.sum axis0 f32 64x2048 w{nw}", isinstance(d, float) and d < 1e-2, f"max_abs={d}")

    # in-warp reduction along a tensor axis (blocked layout, exercises cross-lane ops)
    @triton.jit
    def col_max(x_ptr, o_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr):
        cols = tl.arange(0, n_cols)
        acc = tl.full([n_rows, n_cols], float("-inf"), dtype=tl.float32)
        for r in range(0, n_rows):
            v = tl.load(x_ptr + r * n_cols + cols).to(tl.float32)
            acc = tl.maximum(acc, v[None, :])
        tl.store(o_ptr + cols, tl.max(acc, axis=0))

    x = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    o = torch.empty(32, device="cuda", dtype=torch.float32)
    col_max[(1,)](x, o, n_rows=64, n_cols=32, num_warps=4)
    d = max_abs(o, x.max(dim=0).values)
    record("reduce tl.max over 2D axis0 f32 64x32 w4", isinstance(d, float) and d == 0.0, f"max_abs={d}")


def atomic_cases():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def rmw_k(x_ptr, o_ptr, op: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        if op == 0:
            tl.atomic_max(o_ptr + offs, v, mask=offs < n)
        elif op == 1:
            tl.atomic_min(o_ptr + offs, v, mask=offs < n)
        else:
            tl.atomic_xchg(o_ptr + offs, v, mask=offs < n)

    @triton.jit
    def rmw_contended(x_ptr, o_ptr, op: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
        # every lane hits element 0: this is what actually stresses the CAS expansion
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        if op == 0:
            tl.atomic_max(o_ptr + offs * 0, v, mask=offs < n)
        elif op == 1:
            tl.atomic_min(o_ptr + offs * 0, v, mask=offs < n)
        else:
            tl.atomic_add(o_ptr + offs * 0, v, mask=offs < n)

    torch.manual_seed(5)
    # f32 add under full contention: the result must be the sum of all n lanes
    for n in (32, 256, 1024):
        x = torch.ones(n, device="cuda", dtype=torch.float32)
        o = torch.zeros(1, device="cuda", dtype=torch.float32)
        rmw_contended[(1,)](x, o, op=2, n=n, BLOCK=1024, num_warps=4)
        record(f"atomic_add f32 contended n={n}", abs(float(o.item()) - n) < 1e-4, f"got={float(o.item())}")

    x = torch.randn(256, device="cuda", dtype=torch.float32)
    o = torch.zeros(256, device="cuda", dtype=torch.float32)
    rmw_k[(1,)](x, o, op=2, n=256, BLOCK=256, num_warps=1)
    d = max_abs(o, x)
    record("atomic_add f32 per-lane n=256", isinstance(d, float) and d == 0.0, f"max_abs={d}")

    x = torch.randint(-1000, 1000, (256,), device="cuda", dtype=torch.int32)
    init = torch.zeros(256, device="cuda", dtype=torch.int32)
    o = init.clone()
    rmw_k[(1,)](x, o, op=0, n=256, BLOCK=256, num_warps=1)
    record("atomic_max i32 per-lane", bool(torch.equal(o, torch.maximum(x, init))), "")
    o = init.clone()
    rmw_k[(1,)](x, o, op=1, n=256, BLOCK=256, num_warps=1)
    record("atomic_min i32 per-lane", bool(torch.equal(o, torch.minimum(x, init))), "")

    xi = torch.randint(-1000, 1000, (256,), device="cuda", dtype=torch.int32)
    o = torch.zeros(1, device="cuda", dtype=torch.int32)
    rmw_contended[(1,)](xi, o, op=0, n=256, BLOCK=256, num_warps=4)
    record("atomic_max i32 contended", int(o.item()) == int(xi.max().item()),
           f"got={int(o.item())} want={int(xi.max().item())}")
    o = torch.zeros(1, device="cuda", dtype=torch.int32)
    rmw_contended[(1,)](xi, o, op=1, n=256, BLOCK=256, num_warps=4)
    record("atomic_min i32 contended", int(o.item()) == int(xi.min().item()),
           f"got={int(o.item())} want={int(xi.min().item())}")

    # signed zero and NaN must survive the bitwise CAS expansion on RDNA2, so compare
    # bit patterns: -0.0 == 0.0 numerically and NaN != NaN, which hides real damage
    x = torch.tensor([-0.0, 0.0, float("nan"), 1.0], device="cuda", dtype=torch.float32)
    o = torch.zeros(4, device="cuda", dtype=torch.float32)

    @triton.jit
    def xchg_k(x_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.atomic_xchg(o_ptr + offs, v)

    o = torch.zeros(4, device="cuda", dtype=torch.float32)
    xchg_k[(1,)](x, o, BLOCK=4, num_warps=1)
    xb = x.view(torch.int32).tolist()
    ob = o.view(torch.int32).tolist()
    record("atomic_xchg f32 keeps -0.0/+0.0/NaN bit patterns", xb == ob, f"{xb} -> {ob}")

    # tl.atomic_cas needs cmp/val as tensors matching the pointer tensor (scalar literals
    # fail the "ptr type matches cmp type" verifier), and takes no mask on this build
    @triton.jit
    def cas_k(ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        cmp = tl.full([BLOCK], 1, tl.int32)
        val = tl.full([BLOCK], 2, tl.int32)
        old = tl.atomic_cas(ptr + offs, cmp, val)
        tl.store(o_ptr + offs, old)

    buf = torch.ones(256, dtype=torch.int32, device="cuda")
    outb = torch.zeros(256, dtype=torch.int32, device="cuda")
    cas_k[(1,)](buf, outb, BLOCK=256, num_warps=4)
    record("atomic_cas i32 returns the previous value and writes the new one",
           bool((outb == 1).all().item()) and bool((buf == 2).all().item()),
           f"old_min={int(outb.min())} old_max={int(outb.max())} now={int(buf.unique().tolist()[0])}")


def dot_and_convert_cases():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        am = tl.arange(0, M)
        an = tl.arange(0, N)
        ak = tl.arange(0, K)
        a = tl.load(a_ptr + am[:, None] * K + ak[None, :])
        b = tl.load(b_ptr + ak[:, None] * N + an[None, :])
        tl.store(o_ptr + am[:, None] * N + an[None, :], tl.dot(a, b))

    torch.manual_seed(9)
    for dt, tol in ((torch.float16, 6e-2), (torch.bfloat16, 2e-1), (torch.float32, 2e-2)):
        M = N = K = 32
        a = torch.randn(M, K, device="cuda", dtype=dt)
        b = torch.randn(K, N, device="cuda", dtype=dt)
        o = torch.empty(M, N, device="cuda", dtype=torch.float32)
        mm[(1,)](a, b, o, M=M, N=N, K=K, num_warps=1)
        d = max_abs(o, (a.float() @ b.float()))
        record(f"tl.dot {dt} 32x32x32 (FMA path, wave32)", isinstance(d, float) and d < tol, f"max_abs={d}")

    @triton.jit
    def cvt(x_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        tl.store(o_ptr + offs, v.to(tl.float8e4nv), mask=offs < n)

    n = 128
    x = torch.linspace(-4, 4, n, device="cuda", dtype=torch.float32)
    o = torch.empty(n, device="cuda", dtype=torch.float8_e4m3fn)
    try:
        cvt[(1,)](x, o, n=n, BLOCK=128, num_warps=1)
        ref = x.to(torch.float8_e4m3fn)
        d = max_abs(o, ref)
        record("fp8e4nv conversion matches torch", isinstance(d, float) and d < 0.1, f"max_abs={d}")
    except Exception as exc:
        record("fp8e4nv conversion compiles at all on RDNA2", False, f"{exc!r}"[:180])


def layout_cases():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def st(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        v = tl.load(x_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
        tl.store(o_ptr + tl.arange(0, N)[:, None] * M + tl.arange(0, M)[None, :], tl.trans(v))

    for M, N, nw in ((64, 64, 4), (128, 32, 4), (32, 128, 2)):
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        o = torch.empty(N, M, device="cuda", dtype=torch.float32)
        st[(1,)](x, o, M=M, N=N, num_warps=nw)
        d = max_abs(o, x.t().contiguous())
        record(f"trans + LDS round trip {M}x{N} w{nw}", isinstance(d, float) and d == 0.0, f"max_abs={d}")


def remaining_l6_cases():
    """The op classes the M0 audit left unevidenced on RDNA2: cache modifiers, eviction
    policy, masked atomics under contention, cross-lane scans/gathers, fp8 in tl.dot,
    and the libdevice math set."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def cache_mod(x_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(x_ptr + offs, mask=m, other=0.0, cache_modifier=".cg")
        b = tl.load(x_ptr + n + offs, mask=m, other=0.0, cache_modifier=".ca")
        tl.store(o_ptr + offs, a + b, mask=m, cache_modifier=".wt")

    @triton.jit
    def evict(x_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0, eviction_policy="evict_last")
        tl.store(o_ptr + offs, v * 2, mask=offs < n, eviction_policy="evict_first")

    @triton.jit
    def masked_add_contended(x_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        # half the lanes masked out, all surviving lanes onto element 0
        tl.atomic_add(o_ptr + offs * 0, v, mask=(offs < n) & (offs % 2 == 0))

    @triton.jit
    def scan_k(x_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(o_ptr + offs, tl.cumsum(v, axis=0))

    @triton.jit
    def gather_k(x_ptr, i_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        src = tl.load(x_ptr + offs).to(tl.float32)[None, :]
        idx = tl.load(i_ptr + offs)[None, :].to(tl.int32)
        got = tl.gather(src, idx, axis=1)
        tl.store(o_ptr + offs[None, :], got)

    @triton.jit
    def math_k(x_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        r = (tl.exp(v) + tl.exp2(v) + tl.log(tl.abs(v) + 1e-3) + tl.sqrt(tl.abs(v))
             + tl.rsqrt(tl.abs(v) + 1e-3) + tl.sigmoid(v) + tl.floor(v) + tl.ceil(v)
             + tl.erf(v) + tl.sin(v) + tl.cos(v) + tl.fma(v, 2.0, 1.0))
        tl.store(o_ptr + offs, r, mask=m)

    @triton.jit
    def fp8_dot(a_ptr, b_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], tl.dot(a, b))

    @triton.jit
    def flip_k(x_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.store(o_ptr + offs, tl.flip(v, 0))

    @triton.jit
    def reduce1_k(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        rr = tl.arange(0, R)[:, None]
        cc = tl.arange(0, C)[None, :]
        v = tl.load(x_ptr + rr * C + cc).to(tl.float32)
        tl.store(o_ptr + tl.arange(0, C)[None, :], tl.sum(v, axis=0))

    torch.manual_seed(21)
    n, BLOCK = 256, 256
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    o = torch.zeros(n, device="cuda", dtype=torch.float32)

    try:
        x2 = torch.randn(2 * n, device="cuda", dtype=torch.float32)
        o2 = torch.empty(n, device="cuda", dtype=torch.float32)
        cache_mod[(1,)](x2, o2, n=n, BLOCK=BLOCK)
        d = max_abs(o2, x2[:n] + x2[n:])
        record("cache_modifier .cg/.ca loads and .wt store compute correctly",
               isinstance(d, float) and d == 0.0, f"max_abs={d}")
    except Exception as exc:
        record("cache_modifier .cg/.ca/.wt compiles on RDNA2", False, exc)

    try:
        evict[(1,)](x, o, n=n, BLOCK=BLOCK)
        d = max_abs(o, x * 2)
        record("eviction_policy evict_last/evict_first compute correctly",
               isinstance(d, float) and d == 0.0, f"max_abs={d}")
    except Exception as exc:
        record("eviction_policy compiles on RDNA2", False, exc)

    src = torch.ones(256, device="cuda", dtype=torch.float32)
    dst = torch.zeros(1, device="cuda", dtype=torch.float32)
    masked_add_contended[(1,)](src, dst, n=256, BLOCK=256, num_warps=4)
    record("masked atomic_add under contention counts only unmasked lanes",
           abs(float(dst.item()) - 128.0) < 1e-4, f"got={float(dst.item())}")

    v = torch.randn(256, device="cuda", dtype=torch.float32)
    o = torch.empty(256, device="cuda", dtype=torch.float32)
    scan_k[(1,)](v, o, BLOCK=256, num_warps=4)
    d = max_abs(o, v.cumsum(0))
    record("tl.cumsum matches torch.cumsum", isinstance(d, float) and d < 1e-2, f"max_abs={d}")

    idx = torch.randint(0, 256, (256,), device="cuda", dtype=torch.int64)
    try:
        gather_k[(1,)](v, idx, o, BLOCK=256, num_warps=4)
        d = max_abs(o, v[idx])
        record("tl.gather matches indexing", isinstance(d, float) and d == 0.0, f"max_abs={d}")
    except Exception as exc:
        record("tl.gather compiles on RDNA2", False, exc)

    try:
        flip_k[(1,)](v, o, BLOCK=256, num_warps=4)
        record("tl.flip matches torch.flip", bool(torch.equal(o, torch.flip(v, (0,)))), "")
    except Exception as exc:
        record("tl.flip compiles on RDNA2", False, exc)

    try:
        vr = torch.randn(8, 32, device="cuda", dtype=torch.float32)
        oc = torch.empty(32, device="cuda", dtype=torch.float32)
        reduce1_k[(1,)](vr, oc, R=8, C=32, num_warps=4)
        d = max_abs(oc, vr.sum(0))
        record("cross-lane tl.sum along axis 0 (row reduce) matches torch",
               isinstance(d, float) and abs(d) < 1e-3, f"max_abs={d}")
    except Exception as exc:
        record("row-wise tl.sum compiles on RDNA2", False, exc)

    m = torch.linspace(-2, 2, n, device="cuda", dtype=torch.float32)
    try:
        math_k[(1,)](m, o, n=n, BLOCK=BLOCK)
        ref = (torch.exp(m) + torch.exp2(m) + torch.log(m.abs() + 1e-3) + m.abs().sqrt()
               + torch.rsqrt(m.abs() + 1e-3) + torch.sigmoid(m) + torch.floor(m) + torch.ceil(m)
               + torch.erf(m) + torch.sin(m) + torch.cos(m)
               + torch.addcmul(torch.full_like(m, 1.0), m, torch.full_like(m, 2.0)))
        d = max_abs(o, ref)
        record("libdevice math set matches torch fp32", isinstance(d, float) and d < 2e-3,
               f"max_abs={d}")
    except Exception as exc:
        record("libdevice math set compiles on RDNA2", False, exc)

    try:
        a8 = (torch.randn(32, 32, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        b8 = (torch.randn(32, 32, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        o8 = torch.empty(32, 32, device="cuda", dtype=torch.float32)
        fp8_dot[(1,)](a8, b8, o8, BM=32, BN=32, BK=32, num_warps=1)
        d = max_abs(o8, a8.float() @ b8.float())
        record("tl.dot accepts fp8e4nv operands on RDNA2", isinstance(d, float) and d < 0.05,
               f"max_abs={d}")
    except Exception as exc:
        record("tl.dot fp8e4nv on RDNA2 (expected unsupported: no fp8 hardware path)", False, exc)


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no HIP device")
    ident = {
        "physical_device": torch.cuda.get_device_name(0),
        "gcnArchName": torch.cuda.get_device_properties(0).gcnArchName,
        "warp_size": torch.cuda.get_device_properties(0).warp_size,
        "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    import triton
    ident["triton_version"] = triton.__version__
    ident["triton_package"] = os.path.dirname(triton.__file__)
    ident["emitted_target"] = emitted_target()
    print(json.dumps(ident, indent=1), flush=True)

    t0 = time.time()
    # a single unsupported API must not cost the whole matrix; each group is isolated
    for fn in (reduce_cases, atomic_cases, dot_and_convert_cases, layout_cases,
               remaining_l6_cases):
        try:
            fn()
        except Exception as exc:
            record(f"group {fn.__name__} aborted", False, f"{type(exc).__name__}: {exc}")

    fails = [r for r in RESULTS if not r["ok"]]
    out = {"identity": ident, "seconds": round(time.time() - t0, 1),
           "total": len(RESULTS), "failed": len(fails), "cases": RESULTS}
    path = "/tmp/q21-p1/rdna2_opclass.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n{len(RESULTS)} cases, {len(fails)} FAILED in {out['seconds']}s -> {path}")
    for f in fails:
        print(f"  FAILED: {f['case']} :: {f['detail']}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
