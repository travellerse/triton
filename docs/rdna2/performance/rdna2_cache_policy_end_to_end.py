#!/usr/bin/env python3
"""End-to-end check of RDNA2 cache modifiers: tl.load/tl.store qualifier -> instruction bits.

For every cache modifier, this compiles a real Triton kernel the way the AMD backend does
(``triton.compile`` with an explicit GPUTarget -- never touches the GPU) and then reads the cache
policy back out of the emitted ISA:

  * from the ``.amdgcn`` text, the modifier fields the encoder printed (glc / slc / dlc);
  * from the ``hsaco``, the same fields decoded out of the instruction words at the positions the
    AMD "RDNA 2" ISA Reference Guide gives for MTBUF (section 13.6.1): GLC = bit 14, DLC = bit 15,
    SLC = bit 54.

Bits, not names, are the assertion: the names come from the same disassembler that produced them.

Usage:
  python3 rdna2_cache_policy_end_to_end.py [--cpu gfx1030] [--objdump llvm-objdump]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

# (field name, aux bit Triton must set, instruction word, bit within that word) per ISA 13.6.1.
FIELDS = [("glc", 0, 0, 14), ("slc", 1, 1, 22), ("dlc", 2, 0, 15)]

# What getCtrlBitsForCacheModifierOnRDNA2 in Utility.cpp promises, keyed by (op, modifier).
EXPECTED_AUX = {
    ("load", "none"): 0, ("load", ".ca"): 0, ("load", ".cg"): 5, ("load", ".cs"): 6,
    ("load", ".cv"): 5,
    ("store", "none"): 0, ("store", ".wb"): 0, ("store", ".cg"): 0, ("store", ".cs"): 2,
    ("store", ".wt"): 5,
}


@triton.jit
def copy_kern(inp, outp, N: tl.constexpr,
              LOAD_CM: tl.constexpr, STORE_CM: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(inp + offs, cache_modifier=LOAD_CM)
    tl.store(outp + offs, x, cache_modifier=STORE_CM)


def compile_case(cpu, load_cm, store_cm, n=1024):
    src = ASTSource(
        fn=copy_kern,
        signature={"inp": "*fp32", "outp": "*fp32"},
        constexprs={"N": n, "LOAD_CM": load_cm, "STORE_CM": store_cm},
        attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]},
    )
    return triton.compile(src, target=GPUTarget("hip", cpu, 32),
                          options={"num_warps": 4, "num_stages": 1})


def fields_of_aux(aux):
    return sorted(name for name, bit, _, _ in FIELDS if (aux >> bit) & 1)


CACHE_WORDS = {"glc", "slc", "dlc"}


def split_inst(line):
    """Return (mnemonic, cache fields) for an ISA line, or None if it is not a memory op."""
    m = re.match(r"\s*(buffer_(?:load|store)_\w+)\b(.*)$", line)
    if not m:
        return None
    # Everything after the trailing immediate is addressing mode plus cache fields.
    tail = m.group(2)
    idx = tail.rfind(", 0")
    fields = sorted(set(tail[idx + 3 :].split()) & CACHE_WORDS) if idx >= 0 else []
    return m.group(1), fields


def in_text(asm_text, mnemonic, want):
    """Count `mnemonic` lines whose printed cache fields are exactly `want` / are not."""
    hits = mismatch = 0
    for line in asm_text.splitlines():
        got = split_inst(line)
        if not got or not got[0].startswith(mnemonic):
            continue
        if got[1] == want:
            hits += 1
        else:
            mismatch += 1
    return hits, mismatch


def in_words(objdump, blob, mnemonic, want):
    """Decode instruction words; returns (matching, mismatching) `mnemonic` counts."""
    path = Path(tempfile.mkstemp(suffix=".o")[1])
    path.write_bytes(blob)
    try:
        text = subprocess.run([objdump, "--triple=amdgcn", "-d", str(path)],
                              capture_output=True, text=True, check=True).stdout
    finally:
        path.unlink(missing_ok=True)
    hits = mismatch = 0
    pat = re.compile(r"\t(buffer_(?:load|store)_\w+).*?//\s*[0-9a-fA-F]+:\s*"
                     r"([0-9A-Fa-f]{8}) ([0-9A-Fa-f]{8})")
    for m in pat.finditer(text):
        if not m.group(1).startswith(mnemonic):
            continue
        words = (int(m.group(2), 16), int(m.group(3), 16))
        got = sorted(name for name, _, wi, bit in FIELDS if (words[wi] >> bit) & 1)
        if got == want:
            hits += 1
        else:
            mismatch += 1
    return hits, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default="gfx1030")
    ap.add_argument("--objdump", default="llvm-objdump")
    args = ap.parse_args()

    failures = []
    print(f"cpu={args.cpu}   triton={triton.__version__} at {Path(triton.__file__).parent}")
    print(f"{'op':5} {'cm':5} {'aux':>4} {'want':14} {'text':>10} {'machine code':>16}  verdict")
    for (op, cm), aux in EXPECTED_AUX.items():
        want = fields_of_aux(aux)
        as_arg = "" if cm == "none" else cm
        load_cm, store_cm = (as_arg, "") if op == "load" else ("", as_arg)
        k = compile_case(args.cpu, load_cm, store_cm)
        asm = k.asm["amdgcn"]
        target = f"buffer_{'load' if op == 'load' else 'store'}"
        other = "buffer_store" if target == "buffer_load" else "buffer_load"
        blob = k.asm.get("hsaco")
        have_blob = isinstance(blob, (bytes, bytearray))
        th, tm = in_text(asm, target, want)
        oh, om = in_text(asm, other, [])
        if have_blob:
            wh, wm = in_words(args.objdump, blob, target, want)
            woh, wom = in_words(args.objdump, blob, other, [])
            wcell = f"{wh}+{woh}/{wm + wom} bad"
        else:
            wh = wm = woh = wom = 0
            wcell = "no hsaco"
        # Two assertions per case: the access under test carries exactly `want`, and the other
        # access in the same kernel (requested with no modifier) still carries nothing -- that is
        # what would catch a policy leaking across ops or a hint being dropped on the floor.
        ok = (th > 0 and tm == 0 and oh > 0 and om == 0
              and (not have_blob or (wh > 0 and wm == 0 and woh > 0 and wom == 0)))
        if not ok:
            failures.append((op, cm, {"want": want, "text": (th, tm, oh, om),
                                      "words": (wh, wm, woh, wom)}))
        print(f"{op:5} {cm:5} {aux:>4} {','.join(want) or '-':14} "
              f"{th + oh:>4}/{tm + om:<5} {wcell:>16}  {'OK' if ok else 'MISMATCH'}")

    print("\nfailures:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
