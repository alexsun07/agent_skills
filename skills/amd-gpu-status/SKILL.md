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

It prints the owning user, how long the process has run (e.g. `3d 5h 12m 8s`), and
its full command. **Always carry that runtime into the report** — a job that's been
up for days reads very differently from one started five minutes ago. Its
key job: **when a PID runs as `root`** (the usual case for containerized training),
it reads `/proc/<pid>/cgroup`, extracts the docker container ID, and matches it
against `docker ps -a` — so an anonymous root process becomes a specific container +
image. That's what lets you say *who* to ask. For non-root PIDs it stops at the
username, since that's already the answer.

Run it once per distinct PID. If a PID holds a GPU but `pid_blame` can't resolve it
(no cgroup match, permissions), say so rather than guessing — reading other users'
`/proc` or `docker ps` may need elevated permissions.

## Optional — guessing who launched a container

`pid_blame` is enough for almost every question. Stop there by default.

Sometimes the container name and image say nothing about who owns it
(`mini-swe-agent-eval`, `m3-repro`, or a docker-assigned name like
`priceless_keller`). There's a second script, `scripts/container_owner <name-or-id>`,
that tries to guess the human. **Do not run it as part of the normal flow.** Instead,
after reporting, offer it in one line — e.g. "容器名看不出归属，我可以从挂载路径等
角度猜一下是谁起的，要吗？" — and run it only if they say yes.

Be honest about what it is: everything but one signal is circumstantial. Docker
records the calling user nowhere a non-root reader can reach — the daemon journal
needs the `adm` group, auditd and `/var/lib/docker` need root. So the script ranks:

- `[strong]` a live `docker run/exec` client still attached — the process owner
  really did type the command. Only survives foreground runs; `-d` leaves nothing.
- `[strong]` a bind mount landing in `/home/<user>/...` — best coverage in practice.
- `[weak]` container name / image namespace — **corroboration only**. The namespace
  says who *built* the image. A `sabreshao/vllm` image running out of
  `/home/yinfeliu/` is a real case; going by the image name would blame the wrong
  person.
- `[guess]` who was logged in at `.Created` — usually several people, since tmux
  sessions stay open for days. A shortlist, never an answer.

When it prints `UNKNOWN`, report UNKNOWN. Don't promote the login shortlist into a
name.

## Reporting

Lead with what they asked. "Is GPU 3 free?" → answer in one line first. Otherwise a
compact table:

```
GPU  Status   Process (PID / owner)                        Uptime
  0  free     —                                            —
  3  busy     train.py (PID 12345 / container: alice-sglang)  3d 5h

Free right now: GPU 0, 1, 2 (3 of 8).
```

Round the uptime to two units (`3d 5h`, `12m 8s`) — the seconds only matter for
short-lived processes.

Keep it tight.
