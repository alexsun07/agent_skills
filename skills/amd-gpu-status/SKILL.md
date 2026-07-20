---
name: amd-gpu-status
description: >
  Check AMD GPU status on a ROCm machine: which GPUs are busy vs. free, and which
  processes (PIDs) — and which user or container — are occupying the busy ones. Use
  this skill whenever the user asks about AMD/Instinct GPU status, load, or
  availability — e.g. "is GPU 3 free", "who's using the GPUs on this box", "what's
  running on the MI300X", "which PID is holding VRAM", "GPU 还有空吗", "机器上谁在跑" —
  even if they don't name a specific tool. Works over SSH to a target machine or
  locally.
---

# AMD GPU Status

Two questions, nothing else:

1. **Which GPUs are free** vs. busy.
2. **Who's on the busy ones** — PID → user / command / container.

## Where the commands run

The GPU tools live on the **target AMD machine**. If the user names or implies a
remote host (e.g. `smci355-...`, "the MI300X box"), run over SSH: `ssh <host> '...'`.
If you're already on the target, drop the `ssh` wrapper.

## Step 1 — Which GPUs are busy, and their PIDs

```
ssh <host> 'rocm-smi --showpids'
```

This lists the compute processes and the GPU(s) each occupies. A GPU with no PID is
free; a GPU with one or more PIDs is busy. Collect the distinct PIDs.

If you also want utilization / VRAM at a glance, `rocm-smi` (no args) prints the
per-GPU dashboard, but `--showpids` alone answers "busy or free" — a running PID
means occupied even if utilization reads 0% between steps, so trust the PID over a
momentary util number.

## Step 2 — Blame the PIDs on busy GPUs

For each PID, run the bundled `scripts/pid_blame` helper:

```
ssh <host> 'bash -s <pid>' < scripts/pid_blame     # remote
bash scripts/pid_blame <pid>                         # local
```

It prints the owning user, how long the process has run, and its full command. Its
key job: **when a PID runs as `root`** (the usual case for containerized training),
it reads `/proc/<pid>/cgroup`, extracts the docker container ID, and matches it
against `docker ps -a` — so an anonymous root process becomes a specific container +
image. That's what lets you say *who* to ask. For non-root PIDs it stops at the
username, since that's already the answer.

Run it once per distinct PID. If a PID holds a GPU but `pid_blame` can't resolve it
(no cgroup match, permissions), say so rather than guessing — reading other users'
`/proc` or `docker ps` may need elevated permissions.

## Reporting

Lead with what they asked. "Is GPU 3 free?" → answer in one line first. Otherwise a
compact table:

```
GPU  Status   Process (PID / owner)
  0  free     —
  3  busy     train.py (PID 12345 / container: alice-sglang)

Free right now: GPU 0, 1, 2 (3 of 8).
```

Keep it tight.
