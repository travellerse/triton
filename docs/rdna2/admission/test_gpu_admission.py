#!/usr/bin/env python3
"""CPU-only tests for docs/rdna2/admission/gpu_admission.py (M1L §4).

No torch, no triton, no device: every device-facing observation is a fixture, and every
lock used is a temporary file. The invariant under test is that the target command is
never executed unless the protocol positively reached an admitting state.

Run:  python3 -m unittest discover -s docs/rdna2/admission -p 'test_*.py' -v
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "gpu_admission.py")


def clean_signals(**over):
    """A signal set that collect() would produce when everything is quiet and free."""
    s = {
        "collected_at": time.time(),
        "errors": [],
        "warnings": [],
        "vram_total_mib": 12272,
        "vram_used_mib": 866,
        "kfd_pids": [],
        "lock_file": "/tmp/does-not-matter",
        "lock_exists": True,
        "lock_dir_writable": True,
        "lock_content": "",
        "lock_fd_holders": [],
        "kfd_device_holders": [],
        "processes_readable": True,
        "workload_matches": [],
        "listener": False,
        "listener_pid": "",
        "faults": [],
        "mem_available_mib": 18000,
        "gpulock_sha_replica": "a" * 64,
        "gpulock_sha_runtime": "a" * 64,
        "gpulock_agrees": True,
        "self_pid": os.getpid(),
    }
    s.update(over)
    return s


class Policy:
    """Mirrors the argparse namespace classify() expects."""
    def __init__(self, acquire=False, budget_mib=2048, mem_floor_mib=3500, port=8188,
                 lock_file="/tmp/x"):
        self.acquire, self.budget_mib = acquire, budget_mib
        self.mem_floor_mib, self.port = mem_floor_mib, port
        self.lock_file = lock_file
        self.fault_window_min = 60


sys.path.insert(0, HERE)
import gpu_admission as ga  # noqa: E402


class ClassifyTable(unittest.TestCase):
    def assert_state(self, sig, expect, acquire=False, **kw):
        state, reasons = ga.classify(sig, Policy(acquire=acquire, **kw))
        self.assertEqual(state, expect, f"{expect} expected, reasons={reasons}")
        self.assertTrue(reasons, "every decision must carry a reason")
        return reasons

    def test_01_clean_without_acquire_is_observational_only(self):
        self.assertIn("no exclusivity claim",
                      self.assert_state(clean_signals(), "IDLE_OBSERVED_NO_EXCLUSIVITY")[0])

    def test_02_clean_with_acquire_admits(self):
        self.assert_state(clean_signals(), "EXCLUSIVE_COOPERATIVE_LOCK", acquire=True)

    def test_03_dead_owner_record_is_informational_not_conflict(self):
        # gpulock.sh releases by closing the fd and never clears the record, so leftover text
        # from a dead PID is the normal state and must not block forever.
        state, reasons = ga.classify(clean_signals(lock_content="999999 r47c 9000",
                                                   lock_fd_holders=[]), Policy())
        self.assertEqual(state, "IDLE_OBSERVED_NO_EXCLUSIVITY", reasons)

    def test_03b_record_naming_a_live_pid_without_holder_conflicts(self):
        live = os.getppid()
        sig = clean_signals(lock_content=f"{live} r47c 9000", lock_fd_holders=[])
        self.assertIn("live pid", self.assert_state(sig, "CONFLICT")[0])

    def test_03c_unattributable_fd_holder_conflicts(self):
        self.assert_state(clean_signals(lock_fd_holders=[999999]), "BUSY")  # holder => BUSY
        sig = clean_signals(lock_fd_holders=[999998], lock_content="")
        state, reasons = ga.classify(sig, Policy(acquire=True))
        self.assertIn(state, ("BUSY", "CONFLICT"), reasons)

    def test_04_lock_held_by_other_pid_is_busy(self):
        self.assert_state(clean_signals(lock_fd_holders=[999]), "BUSY")

    def test_05_kfd_pids_present_is_busy(self):
        self.assert_state(clean_signals(kfd_pids=[4242]), "BUSY")

    def test_06_known_workload_alive_is_busy(self):
        self.assert_state(clean_signals(workload_matches=["python main.py --port 8188"]), "BUSY")

    def test_07_listener_without_pid_attribution_conflicts(self):
        self.assert_state(clean_signals(listener=True, listener_pid="unattributed"), "CONFLICT")

    def test_08_unparsable_showpids_is_error(self):
        self.assert_state(clean_signals(kfd_pids=None, errors=["showpids: no idle marker"]), "ERROR")

    def test_09_unparsable_vram_is_error(self):
        self.assert_state(clean_signals(vram_total_mib=None, vram_used_mib=None,
                                       errors=["vram: no card0 line"]), "ERROR")

    def test_10_process_inventory_unavailable_is_error(self):
        self.assert_state(clean_signals(processes_readable=False), "ERROR")

    def test_11_missing_probe_command_is_error(self):
        self.assert_state(clean_signals(errors=["missing tool: rocm-smi"]), "ERROR")

    def test_12_mem_below_floor_blocks(self):
        self.assertIn("MemAvailable", self.assert_state(
            clean_signals(mem_available_mib=1200), "BUSY")[0])

    def test_13_vram_budget_blocks(self):
        # 9500 used + (2048 budget + 1200 floor) > 12272 total -> gpulock.sh's own rule
        self.assertIn("budget refused", self.assert_state(
            clean_signals(vram_used_mib=9500, vram_total_mib=12272), "BUSY",
            budget_mib=2048)[0])
        self.assertIn("budget refused", self.assert_state(
            clean_signals(vram_used_mib=12265, vram_total_mib=12272), "BUSY",
            budget_mib=0)[0])

    def test_14_unwritable_lock_path_is_unavailable(self):
        self.assert_state(clean_signals(lock_exists=False, lock_dir_writable=False),
                          "LOCK_UNAVAILABLE", acquire=True)

    def test_14b_missing_lock_in_writable_dir_is_still_acquirable(self):
        # gpulock.sh creates the file with `exec 9>`; flock identity is the fd, not the inode
        self.assert_state(clean_signals(lock_exists=False, lock_dir_writable=True),
                          "EXCLUSIVE_COOPERATIVE_LOCK", acquire=True)

    def test_15_contract_drift_blocks_acquisition(self):
        self.assert_state(clean_signals(gpulock_agrees=False, gpulock_sha_runtime=None),
                          "LOCK_UNAVAILABLE", acquire=True)

    def test_16_gpu_fault_lines_block(self):
        self.assert_state(clean_signals(
            faults=["amdgpu [gfxhub] page fault pid 123"]), "BUSY")

    def test_17_kfd_idle_but_device_held_conflicts(self):
        self.assert_state(clean_signals(kfd_pids=[], kfd_device_holders=[555]), "CONFLICT")

    def test_18_only_one_state_admits(self):
        self.assertEqual(ga.ADMITTING, {"EXCLUSIVE_COOPERATIVE_LOCK"})
        for st in ga.STATES:
            self.assertEqual(st in ga.ADMITTING, st == "EXCLUSIVE_COOPERATIVE_LOCK")


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rdna2-admission-test-")
        self.d = self.tmp.name
        self.lock = os.path.join(self.d, ".gpu_lock")
        self.history = os.path.join(self.d, "gpu_lock_history.log")
        self.marker = os.path.join(self.d, "marker")
        self.audit_dir = os.path.join(self.d, "audit")
        open(self.history, "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, sig, name="f"):
        p = os.path.join(self.d, f"{name}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(sig, fh)
        self.audit_dir = os.path.join(self.d, "audit")
        return p

    def tool(self, *args, env=None):
        e = dict(os.environ)
        e.update(env or {})
        e["PYTHONPATH"] = HERE
        e["TRITON_ADMISSION_AUDIT_DIR"] = self.audit_dir
        return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True, env=e)

    def base(self, sig, extra=()):
        return ("--lock-file", self.lock, "--history-log", self.history, "--tag", "tst",
                "--signals", self.fixture(sig), "--json", "--acquire", *extra)

    def read_history(self):
        return pathlib.Path(self.history).read_text(encoding="utf-8").splitlines()

    def test_19_admitted_run_executes_target_once(self):
        r = self.tool(*self.base(clean_signals()), "run", "--",
                      sys.executable, "-c", f"open({self.marker!r},'w').write('ran')")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(self.marker), "target must run when admitted")
        rec = json.loads(r.stdout)
        self.assertEqual(rec["state"], "EXCLUSIVE_COOPERATIVE_LOCK")
        self.assertTrue(rec["admitted"])
        self.assertEqual(rec["target_rc"], 0)
        ev = [l.split("|")[2] for l in self.read_history()]
        self.assertEqual(ev, ["acquire", "release"])
        self.assertFalse([p for p in ga._pids_holding(self.lock) if p != os.getpid()],
                        "lock must be gone after the run")

    def test_20_child_failure_propagates_and_releases(self):
        r = self.tool(*self.base(clean_signals()), "run", "--",
                      sys.executable, "-c", "import sys; sys.exit(3)")
        self.assertEqual(r.returncode, 3)
        ev = self.read_history()
        self.assertTrue(any("release" in l and "target_rc=3" in l for l in ev), ev)
        self.assertEqual([p for p in ga._pids_holding(self.lock) if p != os.getpid()], [])

    def test_21_contention_refuses_and_never_launches(self):
        holder = open(self.lock, "a+")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            r = self.tool(*self.base(clean_signals()), "run", "--",
                          sys.executable, "-c", f"open({self.marker!r},'w')")
            self.assertEqual(r.returncode, 9)
            self.assertFalse(os.path.exists(self.marker), "blocked state must not launch")
            rec = json.loads(r.stdout)
            self.assertEqual(rec["state"], "BUSY")
            self.assertIn("EWOULDBLOCK", rec["reasons"][0])
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    def test_22_blocked_classification_never_launches(self):
        for name, sig, expect in (
                ("kfd", clean_signals(kfd_pids=[7]), "BUSY"),
                ("workload", clean_signals(workload_matches=["python main.py --port 8188"]), "BUSY"),
                ("live-record-no-holder",
                 clean_signals(lock_content=f"{os.getppid()} x y", lock_fd_holders=[]),
                 "CONFLICT"),
                ("drift", clean_signals(gpulock_agrees=False), "LOCK_UNAVAILABLE"),
                ("mem", clean_signals(mem_available_mib=500), "BUSY"),
                ("probe", clean_signals(errors=["missing tool: rocm-smi"]), "ERROR")):
            with self.subTest(name):
                marker = f"{self.marker}-{name}"
                r = self.tool(*self.base(sig), "run", "--",
                              sys.executable, "-c", f"open({marker!r},'w')")
                self.assertEqual(r.returncode, 9, name)
                self.assertFalse(os.path.exists(marker), f"{name} launched the target")
                self.assertEqual(json.loads(r.stdout)["state"], expect)

    def test_23_missing_tool_without_fixture_is_error(self):
        empty_bin = os.path.join(self.d, "bin")
        os.makedirs(empty_bin, exist_ok=True)
        for name in ("ps", "ss", "python3"):
            os.symlink(f"/usr/bin/{name}", os.path.join(empty_bin, name))
        env = {"PATH": empty_bin}
        r = self.tool("--lock-file", self.lock, "--history-log", self.history, "--json",
                      "--acquire", "run", "--", sys.executable, "-c", f"open({self.marker!r},'w')",
                      env=env)
        self.assertEqual(r.returncode, 9)
        self.assertFalse(os.path.exists(self.marker))
        self.assertEqual(json.loads(r.stdout)["state"], "ERROR")

    def test_24_refusal_preserves_existing_lock_record_and_history(self):
        with open(self.lock, "w", encoding="utf-8") as fh:
            fh.write("1234 other 9000\n")
        prior = "999|1|acquire|preexisting\n"
        with open(self.history, "w", encoding="utf-8") as fh:
            fh.write(prior)
        r = self.tool(*self.base(clean_signals(kfd_pids=[1234], lock_content="1234 other 9000")),
                      "run", "--", sys.executable, "-c", f"open({self.marker!r},'w')")
        self.assertEqual(r.returncode, 9)
        self.assertEqual(pathlib.Path(self.lock).read_text(encoding="utf-8"),
                         "1234 other 9000\n",
                         "tool must not clear another owner's record")
        self.assertTrue(self.read_history()[0] == prior.strip(), "history must be append-only")
        self.assertFalse(os.path.exists(self.marker))

    def test_25_ownership_dies_with_the_holder_process(self):
        """The lock identity is an fd, so a killed holder releases it: prove that here with
        our own throwaway child (never the live GPU lock, never a peer's process)."""
        child = subprocess.Popen(["flock", "-n", self.lock, "-c", "sleep 30"],
                                 start_new_session=True)
        time.sleep(0.4)
        holders = ga._pids_holding(self.lock)
        self.assertIn(child.pid, holders, f"child should be a visible fd holder, got {holders}")
        self.assertTrue(all(p != os.getpid() for p in holders))
        r = self.tool(*self.base(clean_signals(lock_fd_holders=[child.pid])), "run", "--",
                      sys.executable, "-c", f"open({self.marker!r},'w')")
        self.assertEqual(r.returncode, 9)
        self.assertFalse(os.path.exists(self.marker))
        # our own throwaway tree only: the inherited fd must die with the holder group
        os.killpg(child.pid, 9)
        child.wait()
        self.assertEqual(ga._pids_holding(self.lock), [])
        r2 = self.tool(*self.base(clean_signals()), "run", "--", "/bin/true")
        self.assertEqual(r2.returncode, 0, "acquisition must succeed once the holder is gone")

    def test_26_preflight_reports_reasons_and_audits_every_decision(self):
        r = self.tool("--lock-file", self.lock, "--history-log", self.history, "--tag", "tst",
                      "--signals", self.fixture(clean_signals(kfd_pids=[3])), "--json",
                      "preflight")
        self.assertEqual(r.returncode, 9)
        rec = json.loads(r.stdout)
        self.assertTrue(rec["reasons"] and not rec["admitted"])
        files = os.listdir(self.audit_dir)
        self.assertTrue(any(rec["experiment_id"] in f for f in files),
                        "each decision must be recorded")

    def test_27_no_target_means_report_only(self):
        r = self.tool(*self.base(clean_signals()), "run")
        self.assertEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(self.marker))
        rec = json.loads(r.stdout)
        self.assertIsNone(rec["target"])

    def test_28_run_without_acquire_must_not_launch_on_observed_idle(self):
        # M1M found the fail-open hole: `run` used to refuse only the explicitly-bad states,
        # so an observation-only classification (no mutex taken) still exec'd the target.
        r = self.tool("--lock-file", self.lock, "--history-log", self.history, "--tag", "tst",
                      "--signals", self.fixture(clean_signals()), "--json",
                      "run", "--", sys.executable, "-c", f"open({self.marker!r},'w')")
        self.assertEqual(r.returncode, 9, r.stdout + r.stderr)
        self.assertFalse(os.path.exists(self.marker),
                         "unadmitted state must never reach the target")
        rec = json.loads(r.stdout)
        self.assertEqual(rec["state"], "IDLE_OBSERVED_NO_EXCLUSIVITY")
        self.assertFalse(rec["admitted"])
        self.assertFalse(rec["policy"]["acquire"])
        self.assertEqual([l.split("|")[2] for l in self.read_history()], ["refuse"])
        self.assertEqual([p for p in ga._pids_holding(self.lock) if p != os.getpid()], [])

    def test_29_preflight_stays_informational_on_idle(self):
        # The run-mode guard must not turn preflight's "quiet, but we hold nothing" into a
        # refusal; a later phase uses rc=0 here to decide it is worth attempting acquisition.
        r = self.tool("--lock-file", self.lock, "--history-log", self.history, "--tag", "tst",
                      "--signals", self.fixture(clean_signals()), "--json", "preflight")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = json.loads(r.stdout)
        self.assertEqual(rec["state"], "IDLE_OBSERVED_NO_EXCLUSIVITY")
        self.assertFalse(rec["admitted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
