# RDNA2 GPU runtime admission contract (M1L)

Operational protocol for running any Triton RDNA2 **device** test on this host. Written so a
later phase can follow it without interpreting anything. Tooling:
`docs/rdna2/admission/gpu_admission.py` (+ CPU-only tests in the same directory).

## 1. Starting an eligible runtime test

```
python3 docs/rdna2/admission/gpu_admission.py \
    --tag <experiment-id> --budget-mib <peak estimate> --mem-floor-mib <floor> --acquire \
    run -- <the single command that runs the test>
```

`--acquire` is **required for `run`**, and was not in the original form recorded in §23.8 of the
implementation report. M1M found the wrapper's `run` path used to refuse only `BUSY`, `CONFLICT`,
`ERROR` and `LOCK_UNAVAILABLE`, so `run` without `--acquire` classified the device as
`IDLE_OBSERVED_NO_EXCLUSIVITY` and then **executed the target with no mutex held** — a fail-open
hole against §1.1. `run` now refuses every state that is not `EXCLUSIVE_COOPERATIVE_LOCK`
(`test_28`), while `preflight` stays informational and still exits `0` on an idle observation
(`test_29`).

Nothing else is authorized. Do not run a GPU test directly, and do not import `torch` or
`triton` before this wrapper has admitted the run: the wrapper is what performs the read-only
preflight first. Exit code `0` means the target ran (its own status is reported as `target_rc`
in the JSON record); `9` means refusal, always with printed reasons.

## 2. Which lock is acquired

The existing cross-project cooperative mutex, not a new one:

| item | value |
|---|---|
| lock file (identity) | `/tmp/qwen21-vram/.gpu_lock` |
| primitive | `flock(2)` exclusive, non-blocking (`flock -n 9` / `fcntl.flock(LOCK_EX\|LOCK_NB)`) |
| contract source | `/home/trave/code/rdna2-attention/evidence/harness/gpulock.sh` |
| copy the launchers source | `/tmp/qwen21-vram/gpulock.sh` |
| ownership | **the open file descriptor**, not the file's text |
| budget rule | `used + (budget + 1200 MiB floor) ≤ total` (same as `gpulock.sh`) |
| history log | `/tmp/qwen21-vram/gpu_lock_history.log`, append-only, `pid\|epoch\|event\|tag` |

The wrapper verifies the two `gpulock.sh` copies are byte-identical before trusting the
protocol; drift is `LOCK_UNAVAILABLE` rather than a guess.

## 3. What proves the lock is held

Acquisition is followed by an explicit verification step: re-`flock` on our own fd must succeed
**and** `/proc/<our pid>/fd/*` must contain a link to the lock file. A lock file that merely
exists, or that contains a record, proves nothing — `gpu_unlock` only appends a log line, so a
text record from a dead PID is normal residue (observed live: `2205481 r47c 9000` with zero fd
holders). A record naming a **live** PID with no fd holder is a contradiction → `CONFLICT`.

## 4. What proves launchers participate

Verified participation (read-only audit, this phase):
`/home/trave/code/rdna2-attention/evidence/harness/{run.sh,run_x.sh,te_vram/*.sh,chunk_sweep/run_sweep.sh,fused_attn/run_fused.sh,default_path/run.sh}`
and the ComfyUI instance launcher `/tmp/q21-r47/start_instance.sh` all source `gpulock.sh` and
call `gpu_lock <tag> <budget>` before starting the workload; the lock fd is inherited across
`exec`, so it is released exactly when the server dies. History shows `acquire`/`release` pairs
for tags `r46ord, r46dbg, r47a, r47b, r47c`.

Not participating: `/tmp/q21-harness/run_one.py` (the A/B driver) implements only a host-RAM
floor (`--mem-floor-mb`) and takes no mutex — it relies on the instance launcher that it drives.
Uncoordinated consumers (desktop compositor, Electron/WebKit on `/dev/dri/renderD128`, any user
command) never take the lock at all.

## 5. Signals checked, before and after acquisition

Before: `rocm-smi --showmeminfo vram --csv`; `rocm-smi --showpids`; `ps -eo pid,etime,args`
matched against known workload patterns; `ss -ltnp` for the ComfyUI port; `/proc/*/fd` scan for
the lock file and for `/dev/kfd`; `journalctl -k --since "<window>"` for `gfxhub`/`amdgpu` page
fault, ring reset, GPU reset, KFD fault; `/proc/meminfo` MemAvailable; lock fd holders.
After acquisition the whole set is sampled again, and any new disagreement aborts before the
target command runs.

## 6. Disagreements that block

* `rocm-smi` reports no KFD PID while some process still holds `/dev/kfd` → `CONFLICT`.
* Lock record names a live PID with no fd holder → `CONFLICT`.
* A lock fd holder that cannot be mapped to a live process → `CONFLICT`.
* A listener on the ComfyUI port whose PID cannot be attributed → `CONFLICT`.
* `--showpids` with neither the idle marker nor a parsable PID row, unparsable VRAM CSV,
  unreadable `meminfo`, or an unavailable `journalctl -k` → `ERROR` (gpulock.sh's own `dmesg`
  probe is inert here because `kernel.dmesg_restrict=1`; this wrapper refuses instead of
  silently passing).

`BUSY` (refuse): any KFD compute PID, any known workload process, an attributed listener on the
port, a live foreign lock holder, contention on acquisition, VRAM budget rule failure, host RAM
below `--mem-floor-mib`, or GPU fault lines inside the window.

## 7. Where frameworks may first be imported

Only inside the target command the wrapper execs, after `EXCLUSIVE_COOPERATIVE_LOCK` was
verified. No probe imports `torch`/`triton`; classification is pure stdlib.

## 8. Memory thresholds

* VRAM: `--budget-mib` must be the experiment's own measured/estimated device peak; `gpulock.sh`'s
  `+1200 MiB` floor applies. The peer instance launcher documents that the real ComfyUI peak
  (10.8–11.6 GiB) **cannot** satisfy `used + budget + 1200 ≤ 12272` on this card, so budget is a
  pre-flight check, never a cap — a Triton micro-kernel experiment must state a small honest
  budget (default 2048 MiB) rather than inherit 9000.
* Host RAM: `--mem-floor-mib` (default 3500, the value the peer's own runs used) checked against
  `MemAvailable`.

## 9. Lock lifetime

Held from acquisition until the wrapper exits; the fd stays open across the child, so the lock
covers the *whole* runtime test process tree, not just wrapper start-up. On exit the wrapper
closes the fd (that is the real release) and appends `release`. It then re-scans for processes
still holding the fd and warns loudly about survivors (the documented `FACTS.md` failure mode:
a SIGKILLed runner leaves a child holding the lock fd). It never kills anything.

## 10. What is recorded per experiment

`/tmp/rdna2-admission/audit/<epoch>-<pid>-<tag>.json` with experiment id, state, reasons,
warnings, policy (budget/floor/acquire), the full signal snapshot, target command, `target_rc`,
duration, and any fd survivors; plus the shared `acquire`/`release` lines in the protocol's own
history log.

## 11. Behaviour when a probe fails

Fail closed: `ERROR` → refuse (`9`). Missing tools, unparsable output, unreadable kernel log,
unwritable lock path, unverifiable ownership, or post-acquisition disagreement all refuse. There
is no retry loop and no waiting for someone else's process.

## 12. Limits of the guarantee

`EXCLUSIVE_COOPERATIVE_LOCK` means: no *other participant of this protocol* can hold the mutex
while we do. It does **not** mean the GPU is ours:
* any program that does not call `gpu_lock` can still submit work;
* the wrapper's own two-run race (probe → acquire) is closed only for participants;
* `rocm-smi` VRAM/KFD readings are point samples and can be stale (observed: "No KFD PIDs"
  while a peer's ComfyUI was mid-run);
* kernel fault visibility depends on `journalctl` permissions, which `flock` cannot fix.

## Category separation

| category | what passing it proves | what it does not prove |
|---|---|---|
| ownership coordination | we hold the shared mutex and participants are excluded | no other process uses the GPU |
| workload observation | the sampled signals currently look idle or busy | anything about the future, or about non-participants |
| memory admission | the declared budget fits with the 1200 MiB floor and host RAM floor | that the experiment will not OOM |
| GPU health | no fault lines in the window | the device is trustworthy for correctness |
| runtime-test correctness | (the test's own result) | nothing about the four rows above |

Passing one row never licenses another. A runtime result is admissible evidence only if the
record shows `state = EXCLUSIVE_COOPERATIVE_LOCK`, verified ownership, a clean post-acquisition
re-check, and the executed target's assembly captured from the same process.

## What must still happen for real exclusivity

Nothing in software on this host can exclude an uncoordinated consumer. To raise the guarantee
from "cooperative" to "exclusive" one of these is needed, each a decision by the project owner,
not by an agent: (a) a GPU-scoped cgroup/systemd slice the user runs all interactive GPU work
in, (b) a driver-level mechanism (e.g. devd/KFD arbitration) that the desktop sessions respect,
or (c) a dedicated second device for Triton experiments. Until then, treat this contract as
cooperative-only and keep any result that depends on system-wide silence out of evidence.
