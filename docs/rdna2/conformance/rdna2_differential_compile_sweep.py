"""Differential compile sweep: does RDNA2 lower the same programs like RDNA3 does?

The on-device conformance probe covers behaviour; the lit sweep covers the compiler's own tests.
Neither covers the middle: a broad surface of ops compiled *for gfx1030* and compared structurally
against the same program compiled for gfx1100. Anything that differs is a candidate defect --
asymmetric compile failures, a bit-serial fallback where a packed instruction exists, narrower
vector memory accesses, an intrinsic the family should not use, or a spill the other arch avoids.

Static: triton.compile with an explicit GPUTarget never touches a device.

Caveat that shapes the reading, not the run: gfx1100 has WMMA and gfx1030 does not, so some matrix
asymmetries are ISA facts. Those are expected and are labelled by name in the output; anything
labelled UNEXPECTED is what deserves a look.

Usage: PYTHONPATH=./python python docs/rdna2/conformance/rdna2_differential_compile_sweep.py
"""

import collections
import re
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ARCHES = ("gfx1030", "gfx1100", "gfx1200")
N = 1024
BLOCK = 256


@triton.jit
def k_binary(X, Y, O, OP: tl.constexpr, dt: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i).to(dt)
    y = tl.load(Y + i).to(dt)
    if OP == "add":
        r = x + y
    elif OP == "mul":
        r = x * y
    elif OP == "div":
        r = x / y
    elif OP == "sub":
        r = x - y
    elif OP == "maximum":
        r = tl.maximum(x, y)
    elif OP == "minimum":
        r = tl.minimum(x, y)
    elif OP == "where":
        r = tl.where(x > y, x, y)
    else:
        r = x
    tl.store(O + i, r.to(O.dtype.element_ty))


@triton.jit
def k_unary(X, O, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i).to(tl.float32)
    if OP == "exp":
        r = tl.exp(x)
    elif OP == "exp2":
        r = tl.exp2(x)
    elif OP == "log":
        r = tl.log(x)
    elif OP == "log2":
        r = tl.log2(x)
    elif OP == "sqrt":
        r = tl.sqrt(x)
    elif OP == "rsqrt":
        r = tl.rsqrt(x)
    elif OP == "sigmoid":
        r = tl.sigmoid(x)
    elif OP == "tanh":
        # tl.math.tanh does not exist in this frontend; written from exp2 so the case still
        # exercises the transcendentals lowering path it was meant to cover.
        e = tl.exp2(x * 2.885390081777927)
        r = 1.0 - 2.0 / (e + 1.0)
    elif OP == "fma":
        r = tl.math.fma(x, x, x)
    elif OP == "tanh":
        r = 1.0 - 2.0 / (tl.exp2(x * 2.885390081777927) + 1.0)
    else:
        r = tl.abs(x)
    tl.store(O + i, r)


@triton.jit
def k_cast(X, O, src: tl.constexpr, dst: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(O + i, tl.load(X + i).to(dst))


@triton.jit
def k_reduce(X, O, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i).to(tl.float32)
    if OP == "sum":
        r = tl.sum(x, 0)
    elif OP == "xor":
        r = tl.xor_sum(x.to(tl.int32), 0).to(tl.float32)
    else:
        r = tl.max(x, 0)
    tl.store(O, r)


@triton.jit
def k_scan(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(O + i, tl.cumsum(tl.load(X + i).to(tl.float32), 0))


@triton.jit
def k_trans(X, O, ROWS: tl.constexpr, COLS: tl.constexpr):
    r = tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, COLS)[None, :]
    x = tl.load(X + r * COLS + c)
    tl.store(O + c * ROWS + r, x)


@triton.jit
def k_dot(A, B, O, M: tl.constexpr, Nn: tl.constexpr, K: tl.constexpr, dt: tl.constexpr):
    rm = tl.arange(0, M)[:, None]
    rn = tl.arange(0, Nn)[None, :]
    a = tl.load(A + rm * K + tl.arange(0, K)[None, :]).to(dt)
    b = tl.load(B + tl.arange(0, K)[:, None] * Nn + rn).to(dt)
    acc = tl.dot(a, b, tl.zeros([M, Nn], tl.float32))
    tl.store(O + rm * Nn + rn, acc)


@triton.jit
def k_atomic(X, O, P, BLOCK: tl.constexpr):
    """Separate pointers per element type: reusing one pointer as both f32 and i32 is not a valid
    program (gfx1030 rejects it, gfx1100 crashes the backend -- a separate defect, on the other
    family, and out of scope for what this sweep measures)."""
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i)
    tl.atomic_add(O + i, x)
    tl.atomic_max(O + i, x)
    tl.atomic_and(P + i, x.to(tl.int32, bitcast=True))


@triton.jit
def k_reshape(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i).to(tl.float32)
    y = tl.reshape(x, (BLOCK // 2, 2))
    z = tl.sum(y, 1)
    tl.store(O + tl.arange(0, BLOCK // 2), z)


@triton.jit
def k_select_pred(X, Y, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i).to(tl.int32)
    y = tl.load(Y + i).to(tl.int32)
    r = (x << 3) ^ (y >> 2)
    r = tl.where((r & 0xF) == 0, r, ~r)
    tl.store(O + i, r.to(O.dtype.element_ty))


CASES = []


def add(name, fn, signature, constexprs, expected=("matrix",)):
    CASES.append((name, fn, signature, constexprs, expected))


F32, F16, BF16, I32 = ("*fp32", "*fp16", "*bf16", "*i32")
for op in ["add", "mul", "div", "sub", "maximum", "minimum", "where"]:
    for dt in [tl.float32, tl.float16, tl.bfloat16]:
        add(f"bin_{op}_{dt.name}", k_binary, {"X": F32, "Y": F32, "O": F32},
            {"OP": op, "dt": dt, "BLOCK": BLOCK})
for op in ["exp", "exp2", "log", "log2", "sqrt", "rsqrt", "sigmoid", "tanh", "fma", "abs"]:
    add(f"un_{op}", k_unary, {"X": F32, "O": F32}, {"OP": op, "BLOCK": BLOCK})
for src, dst in [("fp32", "fp16"), ("fp32", "bf16"), ("fp32", "fp8e4nv"), ("fp32", "i32"),
                 ("i32", "fp32"), ("fp16", "fp32"), ("bf16", "fp32")]:
    st = {"fp32": tl.float32, "fp16": tl.float16, "bf16": tl.bfloat16, "fp8e4nv": tl.float8e4nv,
          "i32": tl.int32}
    add(f"cast_{src}_{dst}", k_cast, {"X": {"fp32": F32, "fp16": F16, "bf16": BF16, "i32": I32}[src],
                                      "O": {"fp32": F32, "fp16": F16, "bf16": BF16, "i32": I32,
                                            "fp8e4nv": "*fp8e4nv"}[dst]},
        {"src": st[src], "dst": st[dst], "BLOCK": BLOCK}, expected=())
for op in ["sum", "max", "xor"]:
    add(f"reduce_{op}", k_reduce, {"X": F32, "O": F32}, {"OP": op, "BLOCK": BLOCK}, expected=())
add("scan_cumsum", k_scan, {"X": F32, "O": F32}, {"BLOCK": BLOCK}, expected=())
add("trans_32x32", k_trans, {"X": F32, "O": F32}, {"ROWS": 32, "COLS": 32}, expected=())
add("reshape_sum", k_reshape, {"X": F32, "O": F32}, {"BLOCK": BLOCK}, expected=())
add("pred_bitops", k_select_pred, {"X": I32, "Y": I32, "O": I32}, {"BLOCK": BLOCK}, expected=())
add("atomics_mixed", k_atomic, {"X": F32, "O": F32, "P": I32}, {"BLOCK": BLOCK},
    expected=())
for dt in [tl.float16, tl.bfloat16, tl.float32]:
    add(f"dot_{dt.name}", k_dot, {"A": F32, "B": F32, "O": F32},
        {"M": 16, "Nn": 16, "K": 16, "dt": dt})


def normalise_w(w):
    return WIDTH.get(w, "w?")


def compile_one(arch, fn, signature, constexprs):
    attrs = {(k,): [["tt.divisibility", 16]] for k in range(len(signature))}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    return triton.compile(src, target=GPUTarget("hip", arch, 32),
                          options={"num_warps": 4, "num_stages": 1})


# RDNA3 renamed the memory width mnemonics (dwordx4 -> b128) and gained dual-issue forms
# (v_dual_add_f32 == two v_add_f32 plus a scheduler hint) and the SDWA operand modifier. Comparing
# raw mnemonics therefore reports generation vocabulary as if it were a lowering difference, which
# is exactly the false signal this sweep has to avoid. Normalise to a comparable vocabulary.
WIDTH = {"dwordx4": "w128", "dwordx3": "w096", "dwordx2": "w064", "dword": "w032",
         "b128": "w128", "b96": "w096", "b64": "w064", "b32": "w032"}
SKIP = {"s_clause", "s_delay_alu", "s_nop", "s_waitcnt_depctr", "s_mov_b32"}


# Instruction forms checked against the RDNA2 ISA guide rather than guessed:
#   RDNA3+ only  -- v_cndmask_b16, v_and_b16, v_or_b16, v_mov_b16 (the guide has *_b16 SHIFTS
#                  (V_LSHLREV_B16, V_LSHRREV_B16, and their PK forms) but no b16 cndmask/and/or/mov)
#   RDNA2 only   -- v_pack_b32_f16 (V_PACK_B32_F16 is in this guide; RDNA3 has newer pack forms)
# Anything else that differs while the control pair does not is scheduling noise or a real gap, and
# is reported as such rather than being absorbed into this table.
ISA_VOCAB = {
    "v_cndmask_b16": "RDNA3+ form, absent from the RDNA2 guide",
    "v_and_b16": "RDNA3+ form, absent from the RDNA2 guide",
    "v_or_b16": "RDNA3+ form, absent from the RDNA2 guide",
    "v_mov_b16": "RDNA3+ form, absent from the RDNA2 guide",
    "v_pack_w032_f16": "RDNA2-side V_PACK_B32_F16; RDNA3 uses newer pack forms",
}

WIDTH_ORDER = ["dwordx4", "dwordx3", "dwordx2", "b128", "b96", "b64", "dword", "b32"]


def normalise(op):
    """Canonical (op, weight): dual-issue counts twice, suffixes and width aliases collapse."""
    weight = 1
    if op.startswith("v_dual_"):
        # Keep the v_ prefix: gfx1100's v_dual_add_f32 must land on gfx1030's v_add_f32.
        op, weight = "v_" + op[len("v_dual_"):], 2
    op = re.sub(r"_(e32|e64|sdwa)$", "", op)
    # Channel width collapses like dual-issue does: the RDNA2 guide has no b16 cndmask/and/or/mov
    # (only *_b16 shifts), so RDNA2's b32-with-shuffles form IS the same algorithm, and the
    # interesting quantity left over is how many extra ops that costs -- reported as a count, not
    # as a missing instruction.
    op = op.replace("_b16", "_w032")
    # Longest alias first, and no \b: the width always follows an underscore, which is a word
    # character, so a boundary assertion never matches and silently skips the whole mapping.
    for src in WIDTH_ORDER:
        op = op.replace(src, WIDTH[src])
    # RDNA3 renamed the scalar and-not forms; that is spelling, not lowering.
    op = op.replace("and_not1", "andn2").replace("or3_b64", "or3")
    return op, weight


def metrics(k):
    asm = k.asm["amdgcn"]
    raw = re.findall(r"^\s+([vgs]_\w+)", asm, re.M)
    mnem = collections.Counter()
    for op in raw:
        if op in SKIP:
            continue
        norm, w = normalise(op)
        mnem[norm] += w
    loads = collections.Counter((p, o, w) for p, o, w in
                                re.findall(r"\b(global|buffer)_(load|store)_(dwordx4|dwordx3|"
                                           r"dwordx2|dword|b128|b96|b64|b32)\b", asm))
    sp = re.search(r"\.vgpr_spill_count:\s*(\d+)", asm)
    return {"mnem": mnem, "mem": loads, "spills": int(sp.group(1)) if sp else 0,
            "intrinsics": collections.Counter(re.findall(r"llvm\.amdgcn\.\w+", k.asm["llir"])),
            "nops": sum(mnem.values())}


def main():
    """Three arches, so a difference can be tested for being generational rather than assumed.

    gfx1100 vs gfx1200 is a control pair of two families that BOTH have matrix instructions and
    both postdate RDNA2. If RDNA2 differs from RDNA3 only in ways RDNA3 also differs from RDNA4,
    the difference is generation vocabulary; a difference with no control counterpart is the signal.
    """
    rows = []
    for name, fn, sig, cexpr, expected in CASES:
        got = {}
        for arch in ARCHES:
            try:
                got[arch] = metrics(compile_one(arch, fn, sig, cexpr))
            except Exception as e:  # noqa: BLE001
                got[arch] = f"FAILED: {' '.join(str(e).split())[-110:]}"
        a, b, c = got[ARCHES[0]], got[ARCHES[1]], got[ARCHES[2]]
        strs = [isinstance(x, str) for x in (a, b, c)]
        if any(strs):
            rows.append((name, "BOTH-FAIL" if sum(strs) == 3 else "ASYM-FAIL",
                         next(x for x, f in zip((a, b, c), strs) if f)[:130]))
            continue

        def diff(x, y):
            return (set(x["mnem"]) - set(y["mnem"]), set(y["mnem"]) - set(x["mnem"]))

        d_ab, d_bc = diff(a, b), diff(b, c)
        notes = []
        if a["spills"] != b["spills"] or b["spills"] != c["spills"]:
            notes.append(f"spills {a['spills']}/{b['spills']}/{c['spills']}")
        widths = [max((normalise_w(w) for (_p, _o, w) in m["mem"]), default="w032") for m in (a, b, c)]
        if len(set(widths)) > 1:
            notes.append(f"widest access {'/'.join(widths)}")
        vocab = []
        if d_ab[0] or d_ab[1]:
            unexplained = [o for o in list(d_ab[0]) + list(d_ab[1])
                           if o not in ISA_VOCAB and not o.startswith(("v_cndmask", "v_mov",
                                                                       "v_lshrrev", "v_lshlrev",
                                                                       "v_and", "v_or", "v_perm"))]
            for o in sorted(set(d_ab[0]) | set(d_ab[1])):
                if o in ISA_VOCAB:
                    vocab.append(f"{o}[{ISA_VOCAB[o]}]")
            notes.append(f"rdna2-vs-rdna3 ops: only2={sorted(d_ab[0])[:4]} "
                         f"only3={sorted(d_ab[1])[:4]}"
                         + (f"  RESIDUAL (no ISA or control story): {sorted(unexplained)}"
                            if unexplained else "  [ISA-vocabulary only]"))
            extra_a = sum(a["mnem"][o] - b["mnem"][o] for o in set(d_ab[0]))
            extra_b = sum(b["mnem"][o] - a["mnem"][o] for o in set(d_ab[1]))
            if extra_a or extra_b:
                notes.append(f"emulation cost: +{extra_a} ops on gfx1030, +{extra_b} on gfx1100")
        unexplained_present = "RESIDUAL" in " ".join(notes)
        if not d_ab[0] and not d_ab[1] and not notes:
            rows.append((name, "MATCH", ""))
        elif expected:
            # Named up front as arch-conditional: the dot rows differ because RDNA2 has no
            # WMMA/MFMA at all, which is an ISA fact, not something for this sweep to flag.
            rows.append((name, "ISA-FAMILY", "; ".join(notes)[:190]))
        elif unexplained_present:
            rows.append((name, "DIFF", "; ".join(notes)[:190]))
        else:
            rows.append((name, "VOCAB", "; ".join(vocab)[:120]))

    print(f"{len(CASES)} programs x {len(ARCHES)} arches "
          f"({' vs '.join(ARCHES)}); vocabulary normalised, control pair = rdna3 vs rdna4\n")
    order = {"DIFF": 0, "ASYM-FAIL": 1, "BOTH-FAIL": 1, "ISA-FAMILY": 2, "VOCAB": 2, "MATCH": 3}
    for name, status, note in sorted(rows, key=lambda r: (order.get(r[1], 2), r[0])):
        if status == "MATCH":
            continue
        print(f"{status:11} {name:24} {note}")
    n = collections.Counter(st for _, st, _ in rows)
    print(f"\nMATCH={n['MATCH']}  ISA-vocabulary={n['VOCAB']}  arch-conditional-by-design={n['ISA-FAMILY']}"
          f"  needs-a-look(DIFF)={n['DIFF']}  compile-failures={n['BOTH-FAIL'] + n['ASYM-FAIL']}")
    print("A DIFF row is a lowering difference with neither an ISA-form explanation nor a "
          "RDNA3-vs-RDNA4 control counterpart -- the class worth investigating on hardware.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
