#!/usr/bin/env python3
"""GPU runtime admission for RDNA2 tests (M1L).

Reuses the existing cross-project cooperative mutex instead of inventing one:
  lock file  /tmp/qwen21-vram/.gpu_lock          (flock(2), identity is the FD)
  contract   /home/trave/code/rdna2-attention/evidence/harness/gpulock.sh
             (byte-identical copy sourced by the ComfyUI instance launchers)

The lock is *cooperative*: it serialises only workloads that take it. Nothing here can
make an unrelated program abstain, so the strongest state this tool can report is
EXCLUSIVE_COOPERATIVE_LOCK, never "the GPU is ours".

Stdlib only. It never imports torch/triton and never runs a GPU command itself; it either
refuses, or execs the caller's target with the lock fd inherited (so the lock lives for the
whole child tree).

Classification is separated from collection on purpose: `classify()` takes a plain signals
mapping, which is what makes the fail-closed table testable without touching a device.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = "/home/trave/code/triton"
GPULOCK_REPLICA = "/home/trave/code/rdna2-attention/evidence/harness/gpulock.sh"
GPULOCK_RUNTIME = "/tmp/qwen21-vram/gpulock.sh"
LIVE_LOCK_FILE = "/tmp/qwen21-vram/.gpu_lock"
LIVE_HISTORY_LOG = os.environ.get("TRITON_ADMISSION_HISTORY_LOG",
                          "/tmp/qwen21-vram/gpu_lock_history.log")
AUDIT_DIR = os.environ.get("TRITON_ADMISSION_AUDIT_DIR", "/tmp/rdna2-admission/audit")

KFD_DEVICE = "/dev/kfd"
FLOOR_MIB = 1200  # gpulock.sh Q21_FLOOR_MIB
DEFAULT_BUDGET_MIB = 2048
DEFAULT_MEM_FLOOR_MIB = 3500
DEFAULT_FAULT_WINDOW_MIN = 60

STATES = (
    "EXCLUSIVE_COOPERATIVE_LOCK",
    "IDLE_OBSERVED_NO_EXCLUSIVITY",
    "BUSY",
    "CONFLICT",
    "LOCK_UNAVAILABLE",
    "ERROR",
)
ADMITTING = {"EXCLUSIVE_COOPERATIVE_LOCK"}

WORKLOAD_PATTERNS = (
    "main.py",           # ComfyUI server entry
    "--port 8188",
    "--enable-triton-backend",
    "q21-harness",
    "q21-r47",
    "qwen21-vram",
    "run_one.py",
    "aba_view.py",
    "triton_kernels",
)


def _run(argv: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return -1, "", f"missing tool: {argv[0]}"
    except Exception as exc:  # timeout, decode errors, ...
        return -1, "", f"{type(exc).__name__}: {exc}"


def _sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _vram_mib() -> tuple[int | None, int | None, str]:
    """(total_MiB, used_MiB, error). `rocm-smi --showmeminfo vram --csv` prints
    card0,<total_b>,<used_b> here, i.e. the same fields gpulock.sh reads as $2/$3."""
    rc, out, err = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if rc != 0:
        return None, None, err or f"rocm-smi rc={rc}"
    for line in out.splitlines():
        if line.startswith("card0"):
            f = line.split(",")
            if len(f) < 3:
                return None, None, f"unexpected csv field count in {line[:80]!r}"
            try:
                return int(float(f[1]) // 1048576), int(float(f[2]) // 1048576), ""
            except ValueError:
                return None, None, f"unparsable csv line: {line[:80]}"
    return None, None, "no card0 line in vram csv"


def _kfd_pids() -> tuple[list[int] | None, str]:
    """None means 'could not tell' (never treat that as idle)."""
    rc, out, err = _run(["rocm-smi", "--showpids"])
    if rc != 0 or not out.strip():
        return None, err or f"rocm-smi --showpids rc={rc}"
    if "No KFD PIDs currently running" in out:
        return [], ""
    pids = []
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)\s+\S", line)
        if m and int(m.group(1)) > 1:
            pids.append(int(m.group(1)))
    if not pids:
        return None, "showpids output has neither the idle marker nor a PID row"
    return pids, ""


def _pids_holding(path: str) -> list[int]:
    holders = set()
    for pid in (d for d in os.listdir("/proc") if d.isdigit()):
        fdd = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fdd):
                try:
                    if os.readlink(f"{fdd}/{fd}") == path:
                        holders.add(int(pid))
                        break
                except OSError:
                    pass
        except OSError:
            continue  # exited or not readable
    return sorted(holders)


def _proc_snapshot() -> list[str]:
    rc, out, _ = _run(["ps", "-eo", "pid=,etime=,args="])
    return out.splitlines() if rc == 0 else []


def _listener_on(port: int) -> tuple[bool, str]:
    rc, out, err = _run(["ss", "-ltnp"])
    if rc != 0:
        return False, err or f"ss rc={rc}"
    for line in out.splitlines():
        if f":{port}" in line:
            m = re.search(r"pid=(\d+)", line)
            return True, m.group(1) if m else "unattributed"
    return False, ""


def _recent_faults(window_min: int) -> tuple[list[str], str]:
    """Kernel GPU fault lines inside the window. Returns ([], '') when the log genuinely has
    none and (None, why) when the observation itself is unavailable -- gpulock.sh's own dmesg
    probe silently yields nothing under dmesg_restrict=1, so an unreadable fault channel must
    be a refusal here, not a pass."""
    if os.uname().sysname != "Linux":
        return [], "not linux"
    rc, out, err = _run(["journalctl", "-k", "--no-pager", "--since",
                         f"{window_min} min ago", "-o", "short-iso"])
    if rc != 0:
        return None, (err or "journalctl -k unavailable").splitlines()[0][:120]
    hits = [l for l in out.splitlines()
            if re.search(r"gfxhub|amdgpu.*(page fault|ring .*reset|GPU reset|timeout)"
                         r"|kfd:.*fault", l, re.I)]
    return hits[-5:], ""


def collect(cfg: argparse.Namespace) -> dict:
    sig: dict = {"collected_at": time.time(), "errors": [], "warnings": []}
    total, used, e = _vram_mib()
    sig["vram_total_mib"], sig["vram_used_mib"] = total, used
    if e:
        sig["errors"].append(f"vram: {e}")
    pids, e = _kfd_pids()
    sig["kfd_pids"] = pids
    if e:
        sig["errors"].append(f"showpids: {e}")

    lock = cfg.lock_file
    sig["lock_file"] = lock
    sig["lock_exists"] = os.path.exists(lock)
    sig["lock_content"] = ""
    if sig["lock_exists"]:
        try:
            with open(lock, encoding="utf-8", errors="replace") as fh:
                sig["lock_content"] = fh.read(120).strip()
        except OSError as exc:
            sig["errors"].append(f"lock read: {exc}")
    sig["lock_fd_holders"] = _pids_holding(lock) if sig["lock_exists"] else []

    sig["kfd_device_holders"] = _pids_holding(KFD_DEVICE)
    procs = _proc_snapshot()
    sig["processes_readable"] = bool(procs)
    sig["workload_matches"] = sorted({p.strip()[:100] for p in procs
                                      for pat in WORKLOAD_PATTERNS if pat in p})
    sig["listener"], sig["listener_pid"] = _listener_on(cfg.port)
    faults, ferr = _recent_faults(cfg.fault_window_min)
    sig["faults"] = faults or []
    if ferr:
        sig["errors"].append(f"fault log: {ferr}")
    sig["lock_dir_writable"] = os.access(os.path.dirname(lock) or ".", os.W_OK)
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            sig["mem_available_mib"] = int(next(
                int(l.split()[1]) // 1024 for l in fh if l.startswith("MemAvailable:")))
    except (OSError, StopIteration, ValueError) as exc:
        sig["mem_available_mib"] = None
        sig["errors"].append(f"meminfo: {exc}")

    a, b = _sha256(GPULOCK_REPLICA), _sha256(GPULOCK_RUNTIME)
    sig["gpulock_sha_replica"], sig["gpulock_sha_runtime"] = a, b
    sig["gpulock_agrees"] = bool(a and a == b)
    sig["self_pid"] = os.getpid()
    return sig


def classify(sig: dict, cfg: argparse.Namespace) -> tuple[str, list[str]]:
    """Fail closed: any unparsed/missing/unattributable observation is a refusal."""
    reasons: list[str] = []
    if sig.get("errors"):
        return "ERROR", [f"probe failure: {e}" for e in sig["errors"]]
    if not sig.get("processes_readable", False):
        return "ERROR", ["process inventory unavailable"]

    if sig.get("kfd_pids"):
        reasons.append(f"KFD compute PIDs present: {sig['kfd_pids']}")
    if sig.get("workload_matches"):
        reasons.append(f"known GPU workload processes alive: {sig['workload_matches'][:3]}")
    if sig.get("listener") and str(sig.get("listener_pid", "")).isdigit():
        reasons.append(f"listener on port {cfg.port} (pid={sig['listener_pid']})")
    if sig.get("faults"):
        reasons.append(f"recent GPU fault lines: {len(sig['faults'])}")
    holders = [p for p in sig.get("lock_fd_holders", []) if p != sig.get("self_pid")]
    if holders:
        reasons.append(f"cooperative lock held by pid(s) {holders}")
    if reasons:
        return "BUSY", reasons

    if not sig.get("gpulock_agrees", False):
        return "LOCK_UNAVAILABLE", [
            "gpulock.sh contract drift: the repository copy and the runtime copy the launchers "
            "source are not byte-identical, so the mutex semantics cannot be trusted"]

    # Cross-check the views. Note the deliberate asymmetry with gpulock.sh: the lock's identity
    # is its FD, and gpu_unlock only appends a log line, so a leftover text record naming a dead
    # PID is normal and NOT evidence of ownership. A contradiction is.
    conflict = []
    if sig.get("kfd_pids") == [] and sig.get("kfd_device_holders"):
        conflict.append(f"rocm-smi idle but /dev/kfd held by {sig['kfd_device_holders']}")
    holders_all = sig.get("lock_fd_holders", [])
    ghost = [p for p in holders_all if not os.path.exists(f"/proc/{p}")]
    if ghost:
        conflict.append(f"lock fd holder(s) {ghost} could not be attributed to a live process")
    rec = (sig.get("lock_content") or "").split()
    if rec and rec[0].isdigit() and not holders_all:
        rec_pid = int(rec[0])
        if os.path.exists(f"/proc/{rec_pid}") and rec_pid not in holders_all:
            conflict.append(f"lock record names live pid {rec_pid} ({rec[1] if len(rec) > 1 else ''}) "
                            f"but no fd holder")
        else:
            sig.setdefault("warnings", []).append(
                f"stale lock record {sig.get('lock_content')!r}: pid {rec_pid} is gone and no fd "
                f"holder exists, which gpulock.sh's own semantics allow (release = fd close)")
    if sig.get("listener_pid") == "unattributed":
        conflict.append("listener exists but its PID could not be attributed")
    if conflict:
        return "CONFLICT", conflict
    if not sig.get("lock_exists") and not sig.get("lock_dir_writable", False):
        return "LOCK_UNAVAILABLE", [
            f"lock path {sig['lock_file']} absent and its directory is not writable, so the "
            f"shared mutex identity cannot be established"]

    free = (sig.get("vram_total_mib") or 0) - (sig.get("vram_used_mib") or 0)
    need = cfg.budget_mib + FLOOR_MIB
    if (sig.get("vram_used_mib") or 0) + need > (sig.get("vram_total_mib") or 0):
        return "BUSY", [f"vram budget refused: used={sig['vram_used_mib']} + {need} > total={sig['vram_total_mib']}"]
    if sig.get("mem_available_mib") is not None and sig["mem_available_mib"] < cfg.mem_floor_mib:
        return "BUSY", [f"host RAM gate: MemAvailable={sig['mem_available_mib']} MiB < {cfg.mem_floor_mib} MiB"]

    if not cfg.acquire:
        return "IDLE_OBSERVED_NO_EXCLUSIVITY", [
            "signals consistent-idle; no lock acquired, so no exclusivity claim"]
    return "EXCLUSIVE_COOPERATIVE_LOCK", [
        f"eligible to take the cooperative mutex on {sig['lock_file']}; cooperative scope only "
        f"(uncoordinated programs can still submit work); free VRAM {free} MiB, MemAvailable "
        f"{sig.get('mem_available_mib')} MiB"]


def try_acquire(lock_file: str) -> tuple[int | None, str]:
    """Open + non-blocking flock; returns the fd to keep (inherited by the child tree)."""
    try:
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        return None, f"open failed: {exc}"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, "EWOULDBLOCK: another participant holds the mutex"
    except OSError as exc:
        os.close(fd)
        return None, f"flock failed: {exc}"
    return fd, "acquired"


def verify_held(fd: int, lock_file: str, self_pid: int) -> tuple[bool, str]:
    """Prove ownership by fd identity, not by the file's text record."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # re-asserting our own lock: no-op
    except OSError as exc:                               # returns None on success, raises otherwise
        return False, f"re-flock on our own fd failed: {exc}"
    holders = _pids_holding(lock_file)
    if self_pid not in holders:
        return False, f"no /proc fd entry for pid {self_pid} pointing at {lock_file}"
    return True, f"verified holder pid={self_pid}"


def history(tag: str, event: str, log_path: str, extra: str = "") -> None:
    """Append in gpulock.sh's own field order: pid|epoch|event|tag."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}|{int(time.time())}|{event}|{tag}{(' ' + extra) if extra else ''}\n")
    except OSError:
        pass  # auditing must never be the reason a run proceeds or fails loudly


def audit(exp_id: str, record: dict) -> None:
    try:
        os.makedirs(AUDIT_DIR, exist_ok=True)
        with open(os.path.join(AUDIT_DIR, f"{exp_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1, sort_keys=True)
    except OSError as exc:
        print(f"audit write failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock-file", default=LIVE_LOCK_FILE)
    ap.add_argument("--history-log", default=LIVE_HISTORY_LOG)
    ap.add_argument("--tag", default="rdna2")
    ap.add_argument("--budget-mib", type=int, default=DEFAULT_BUDGET_MIB)
    ap.add_argument("--mem-floor-mib", type=int, default=DEFAULT_MEM_FLOOR_MIB)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--fault-window-min", type=int, default=DEFAULT_FAULT_WINDOW_MIN)
    ap.add_argument("--acquire", action="store_true", help="take the cooperative mutex")
    ap.add_argument("--signals", help="JSON fixture instead of live probes (CPU-only tests)")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("preflight")
    run = sub.add_parser("run")
    run.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    mode = ns.mode or "preflight"
    cmd = [c for c in getattr(ns, "cmd", []) if c != "--"]

    if ns.signals:
        with contextlib.suppress(OSError):
            sig = json.load(open(ns.signals, encoding="utf-8"))
            sig.setdefault("self_pid", os.getpid())
    else:
        sig = collect(ns)
    state, reasons = classify(sig, ns)
    exp_id = f"{int(time.time())}-{os.getpid()}-{ns.tag}"
    report = {"experiment_id": exp_id, "state": state, "reasons": reasons,
              "warnings": sig.get("warnings", []),
              "admitted": state in ADMITTING, "policy": {"budget_mib": ns.budget_mib,
              "mem_floor_mib": ns.mem_floor_mib, "acquire": ns.acquire},
              "signals": sig}

    if mode == "preflight":
        _emit(report, ns.json)
        audit(exp_id, report)
        return 0 if state in ADMITTING or state == "IDLE_OBSERVED_NO_EXCLUSIVITY" else 9

    # Run mode is admission-controlled, not observation-controlled. Listing only the
    # explicitly-bad states used to let `run` launch the target while the classification was
    # IDLE_OBSERVED_NO_EXCLUSIVITY (no mutex held), which is exactly the state the contract
    # says must refuse.
    if state not in ADMITTING:
        _emit(report, ns.json)
        audit(exp_id, report)
        history(ns.tag, "refuse", ns.history_log, state)
        return 9

    fd = None
    if ns.acquire:
        fd, why = try_acquire(ns.lock_file)
        if fd is None:
            report.update({"state": "BUSY", "reasons": [f"acquisition: {why}"], "admitted": False})
            _emit(report, ns.json)
            audit(exp_id, report)
            history(ns.tag, "refuse-busy", ns.history_log)
            return 9
        ok, why = verify_held(fd, ns.lock_file, os.getpid())
        if not ok:
            os.close(fd)
            report.update({"state": "ERROR", "reasons": [f"ownership unverified: {why}"],
                           "admitted": False})
            _emit(report, ns.json)
            audit(exp_id, report)
            return 9
        try:
            with os.fdopen(os.dup(fd), "w") as fh:
                fh.write(f"{os.getpid()} {ns.tag} {ns.budget_mib}\n")
        except OSError:
            pass
        history(ns.tag, "acquire", ns.history_log, f"rdna2-admission {why}")
        # Re-observe after acquisition: the contract is that the lock orders participants,
        # not that it proves the card is otherwise empty. Fixture mode re-uses the fixture so
        # classification tests stay deterministic and never touch live device probes.
        if ns.signals:
            fresh = json.loads(json.dumps(sig))
            fresh["self_pid"] = os.getpid()
        else:
            fresh = collect(ns)
        fresh["lock_fd_holders"] = [os.getpid()]
        state2, reasons2 = classify(fresh, ns)
        if state2 not in ADMITTING:
            report["post_acquire"] = {"state": state2, "reasons": reasons2}
            if state2 != "IDLE_OBSERVED_NO_EXCLUSIVITY":
                history(ns.tag, "release", ns.history_log, "post-acquire conflict")
                os.close(fd)
                report.update({"state": "CONFLICT", "reasons": reasons2, "admitted": False})
                _emit(report, ns.json)
                audit(exp_id, report)
                return 9
    if not cmd:
        report["target"] = None
        _emit(report, ns.json)
        audit(exp_id, report)
        if fd is not None:
            history(ns.tag, "release", ns.history_log, "no target command")
            os.close(fd)
        return 0

    report["target"] = cmd
    started = time.time()
    try:
        # fd stays open across the child tree, so the mutex lives for the whole workload.
        rc = subprocess.call(cmd)
    except OSError as exc:
        rc = 127
        report["target_error"] = str(exc)
    finally:
        report["target_rc"] = rc
        report["duration_s"] = round(time.time() - started, 3)
        survivors = [p for p in _pids_holding(ns.lock_file) if p != os.getpid()]
        if fd is not None:
            history(ns.tag, "release", ns.history_log, f"target_rc={rc}")
            os.close(fd)
        if survivors:
            report["inherited_fd_survivors"] = survivors
        audit(exp_id, report)
        _emit(report, ns.json)
    if report.get("inherited_fd_survivors"):
        print(f"WARNING: pid(s) {report['inherited_fd_survivors']} still hold the lock fd "
              f"(inherited across exec); not killing anything", file=sys.stderr)
    return rc


def _emit(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print(f"state={report['state']} admitted={report['admitted']} "
              f"experiment={report['experiment_id']}")
        for r in report["reasons"]:
            print(f"  - {r}")
        for w in report.get("warnings", []):
            print(f"  ! {w}")


if __name__ == "__main__":
    sys.exit(main())
