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
  data. It leads with an outcome summary, then breaks work into sections with methods,
  techniques, and the concrete git commits / PRs. It stays disciplined: no future
  plans, no agent speculation, and only brief, hedged coverage of still-exploratory work.
---

# Weekly Report

Produce a personal weekly work report that a manager or teammate could read and
immediately understand **what got done and what it accomplished** — nothing more.

The reader trusts this report because everything in it is real. That trust is the
whole point: a report padded with plans, guesses, or half-finished experiments makes
the reader second-guess the parts that *are* solid. So the discipline below isn't
bureaucratic — it's what makes the report worth reading.

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

Write the report to the **current working directory** as a Markdown file named:

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

Use this shape. Adapt the section titles to the actual work — these are a guide, not a
rigid form.

```markdown
# Weekly Report — {username} — {YYYY-MM-DD}

## Summary

A few sentences up top on the outcomes: what was accomplished this week. Lead with
impact — performance improved by N%, bug X fixed, feature Y shipped. Concrete and
quantified wherever the numbers exist.

## [Work item 1 — a short descriptive title]

What was done and what it achieved. Then the substance:
- **Method / approach**: how it was done, what techniques or tools were used.
- **Evidence**: the concrete result — benchmark numbers, the bug's symptom now gone,
  the test that passes.
- **Changes**: commit hashes and/or PR references. E.g. `abc1234`, PR #123.

## [Work item 2 — ...]

...

## Exploration (optional, keep brief)

Short, hedged notes on work still in progress — only what was actually observed.
```

### On the summary

The summary is the part most people read. Make it stand on its own: someone should be
able to read only the summary and know the three or four things that mattered this week.
Quantify where you can — "reduced TTFT by 18% on MI300X" beats "improved latency".

### On the sections

Each section should close on a verified outcome, not trail off into intentions. Include
the git commits and PRs because they make the work checkable — the reader (or future
you) can go look. Pull exact hashes from `git log`; don't approximate them.

## Before you finish

Re-read the draft against the four rules. The most common failure is a sentence that
slipped in a plan ("next I'll…"), a guess stated as fact, or an exploratory result
written up as if it were a win. Cut those. A shorter report of solid facts beats a
longer one padded with maybes.
