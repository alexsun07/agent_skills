---
name: weekly-report
description: >
  Write the user's personal weekly work report in English from the current session
  and project state. Use this skill whenever the user asks you to write, generate,
  draft, or update a weekly report / weekly summary / 周报 / weekly update — e.g.
  "帮我写个周报", "write my weekly report", "总结一下这周做的事", "generate this week's
  update", "把这个 session 的工作整理成周报" — even if they don't name a date or a
  file. The report is grounded in facts: things actually completed and verified this
  week, drawn from the conversation context, git history, code, and profiling/benchmark
  data. It works in two steps: first propose a numbered list of the key points and revise
  it until the user approves, then write the full report — a numbered summary led by the
  week's overall result, then a detailed section per point with methods, techniques, and
  the concrete git commits / PRs. It stays disciplined: no future plans, no agent
  speculation, and only brief, hedged coverage of still-exploratory work.
---

# Weekly Report

Produce a personal weekly work report that a manager or teammate could read and
immediately understand **what got done and what it accomplished** — nothing more.

The reader trusts this report because everything in it is real. That trust is the
whole point: a report padded with plans, guesses, or half-finished experiments makes
the reader second-guess the parts that *are* solid. So the discipline below isn't
bureaucratic — it's what makes the report worth reading.

## The two-step workflow

Never jump straight to the full report. Work in two steps, with a checkpoint in between.

### Step 1 — propose the key points, then stop

Gather the evidence first (session context, git, code, benchmark data — see below), then
show the user a **numbered list of the key points** the report will cover.

The block below is **only an illustration of the format** — invented content, made-up
hashes, an arbitrary number of items. Never copy its wording, its topics, or its item
count into a real report. Your points come entirely from this session's actual work; if
the week has two real points, list two.

```
1. Overall: E2E serving throughput on MI300X 18K → 25K tok/s/GPU across this week's changes
2. Paged-attention block-size tuning — TTFT 240ms → 197ms (~18%), `a1b2c3d`
3. (exploration, brief) FP8 KV cache — one measurement, ~20% memory saved
```

Three lines here because that's enough to show the three roles a point can play — headline,
finished work item, hedged exploration. A real week has however many points it has.

Each line is one sentence: what was done and the outcome, with the commit/PR that backs
it. Enough for the user to judge whether the point belongs, is worded right, and is in
the right order — not a paragraph.

**Point 1 is the headline — the week as a whole, not the first work item.** A reader
who sees only line 1 should know how the week went overall. The strongest version is a
single end-to-end number the individual items add up to ("E2E throughput 18K → 25K
tok/s/GPU", "P99 latency down 31% on the production config"), because that's the thing a
manager actually wants and it's the one number the per-item points can't convey on their
own. Take it from a real end-to-end measurement — if nobody measured the whole system,
don't manufacture the number by adding up per-change wins. In that case make point 1 a
factual one-line roll-up of the week instead ("three changes landed on the MI300X serving
path: one measured perf win, one correctness fix, one new benchmark"), and say plainly
that no end-to-end measurement was taken. Points 2 onward are the individual work items
that make up the headline.

Order the rest the way the report will read: finished, verified, high-impact work first;
brief exploratory items last, marked as such.

Then **stop and ask the user to confirm** — do they want points added, dropped, merged,
reordered, or reframed? Do not write the report file in this step.

### The confirmation loop

If the user comes back with changes, apply them and **show the revised numbered list
again**, then ask again. Keep looping until they approve. Re-showing costs a few seconds
and is what makes the checkpoint real: the whole point is that the user signs off on the
final wording, and a revision they never saw is a revision they never agreed to. Don't
treat "change point 3 to X" as approval of the rest — they're reacting to one line, not
ratifying the list.

Only an explicit go-ahead ("looks good", "ok write it", "确认") moves you to step 2. If
their reply is ambiguous — a comment that might be a tweak or might be a question — ask
rather than assume; assuming costs the user a report they have to correct afterward.

### Step 2 — write the full report

Once the user has approved the list, write the complete report to a file. Its **Summary**
is the approved numbered list — the same points, same order, same numbering, and the
wording they signed off on. Each numbered point then gets its own detailed section, in
that same order and numbering, so the user can match summary item 3 to section 3 at a
glance.

Point 1 is the headline. If it's a roll-up that says nothing beyond what sections 2..N
already document, it doesn't need a section of its own — start the sections at 2. Give it
a section when there's real substance to show: how the end-to-end number was measured,
the config and hardware it was measured on, a before/after table.

If the user's first message clearly asks to skip the checkpoint ("just write it", "don't
ask me, generate the report"), honor that and go straight to step 2.

## Where the content comes from

You are almost always invoked inside a session that has already been running for a
while. The material for the report is right there in the conversation and the repo:

- **Session context** — what you and the user actually did and confirmed this session.
  This is usually the richest source; mine it first.
- **Git history** — `git log`, `git show`, `git diff` in the current project. Use this
  to recover exact commit hashes, PR references, file changes, and to *verify* that
  something claimed in conversation actually landed.
- **Code** — read the files that changed to describe what a change really does, not
  what someone hoped it would do.
- **Profiling / benchmark data** — trace files, benchmark outputs, before/after numbers
  produced during the session. These are the strongest evidence of a performance claim;
  quote the concrete numbers.

If the session is thin and the git log is the main signal, that's fine — just report
what the commits and code actually show.

## File output

In step 2 (after the user confirms the points), write the report to the **current working
directory** as a Markdown file named:

```
{YYYYMMDD}-{username}.md
```

- `{username}` — the Linux login name. Get it with `whoami` (do not guess).
- `{YYYYMMDD}` — the report date. Default to **today in UTC+8** (get it with
  `TZ='Asia/Shanghai' date +%Y%m%d`). If the user names a specific date, use that instead.

Write the report in **English**.

## What to include — and what to leave out

These four rules are the heart of the skill. When you're unsure whether something
belongs, come back to them.

### 1. Report finished, verified work — not work in flight

Lead with things that are **done and confirmed correct**: a bug that's fixed and the
fix validated, a feature that's complete and working, a performance win you actually
measured. If you can't point to evidence that something is finished and correct (a
passing test, a merged PR, a benchmark number, a confirmation in the session), it
doesn't belong in the main body. This is what makes the report trustworthy.

### 2. No future plans

Leave out "next steps", "TODO", "planning to", "will investigate". A weekly report is a
record of what happened, not a roadmap. If the user wants a plan, that's a separate
document. Ending a section on a plan dilutes the accomplishment it should be closing on.

### 3. Only facts — no agent speculation

Do not write what you *think* or *guess* happened. If you didn't see it land, verify it
in git or the code before claiming it. Never dress up an assumption as a result. When in
doubt, check — `git log`, `git show`, read the file — or leave it out.

### 4. Keep exploratory work brief and hedged

Work that is still being explored — an experiment underway, an approach being trialed —
can appear, but keep it **short and restrained**. Report only what was concretely
observed (e.g. "measured X under config Y"), not where it might lead. Minimize
speculation, give it less space than finished work, and never let it read like a
conclusion. If it's purely a guess with nothing observed yet, drop it.

## Report structure

Use this shape. Everything in `[brackets]` is a placeholder to be replaced with the real
work, and the four numbered items are just to show the pattern — the report has exactly
as many points as the user confirmed. Adapt the section titles to the actual work; this
is a guide, not a rigid form.

```markdown
# Weekly Report — {username} — {YYYY-MM-DD}

## Summary

1. [Headline — the week overall, ideally one end-to-end number.]
2. [Point 2 — a work item: what was done and what it achieved, quantified.]
3. [Point 3 — ...]

## 1. [Headline — only if there's substance beyond the sections below]

How the end-to-end result was measured: config, hardware, before/after.

## 2. [Point 2 — a short descriptive title]

What was done and what it achieved. Then the substance:
- **Method / approach**: how it was done, what techniques or tools were used.
- **Evidence**: the concrete result — benchmark numbers, the bug's symptom now gone,
  the test that passes.
- **Changes**: commit hashes and/or PR references. E.g. `abc1234`, PR #123.

## 3. [Point 3 — ...]

...

## N. Exploration (optional, keep brief)

Short, hedged notes on work still in progress — only what was actually observed.
```

### On the summary

The summary is the part most people read, and it is a **numbered list** — the same points
the user approved in step 1, in the same order and wording. One line each, no sub-bullets.
Make it stand on its own: someone should be able to read only the summary and know
everything that mattered this week. Quantify where you can — "reduced TTFT by 18% on
MI300X" beats "improved latency".

Point 1 answers "how did the week go?" in one line, and points 2 onward answer "what
did you do?". Many readers stop after line 1, which is why the overall result goes
there rather than being left for them to infer from a list of individual changes.

### On the sections

There is one section per summary point (except a pure roll-up point 1), numbered to
match, in the same order. Each
section should close on a verified outcome, not trail off into intentions. Include the
git commits and PRs because they make the work checkable — the reader (or future you)
can go look. Pull exact hashes from `git log`; don't approximate them.

## Before you finish

Re-read the draft against the four rules. The most common failure is a sentence that
slipped in a plan ("next I'll…"), a guess stated as fact, or an exploratory result
written up as if it were a win. Cut those. A shorter report of solid facts beats a
longer one padded with maybes.

Then check the shape: point 1 states the week's overall result (and any end-to-end number
in it came from a real measurement, not from adding up the per-item wins), the summary is
a numbered list, every summary point has a matching numbered section, and the numbering,
order, and wording match what the user approved. If you added or dropped a point while
writing, say so when you report back.
