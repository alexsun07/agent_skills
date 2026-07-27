---
name: conductor-system
description: >
  Query and reserve machines in AMD Conductor using the `conductor` CLI. Use this
  skill whenever the user wants to check Conductor reservations, find free/available
  time on a machine, book or reserve a Conductor system, extend an existing
  reservation, or inspect current and future booking status — even if they don't say
  the word "Conductor" but clearly mean an AMD lab machine (e.g. "smc300x-...",
  "MI300X box", "GPU 机器还有空吗", "帮我约一下那台机器"). Also handles first-time setup
  (CLI install + auth) when the machine isn't ready yet.
---

# Conductor System

Query and reserve AMD lab machines through the `conductor` CLI. A typical run is:
verify setup → identify the machine → query reservations → compute free time →
create reservations in policy-safe chunks → verify. Communicate what you find and
what you plan to do at each step; reservations are shared state, so never create one
without checking for overlaps and confirming intent first.

In the command templates below, `<...>` marks a value you must substitute before
running — an entity ID, a timestamp, a title. Never send a command with a literal
placeholder still in it; Conductor will try to parse `<now-utc>` as a real date and
fail with a database error.

## Configuration

Reservation policy and defaults live in `conductor-machines.yml` in this skill's
directory, not in the instructions, because limits and defaults differ per machine
and pool and the user edits them over time. Read it before any query or create:

```text
<skill-dir>/conductor-machines.yml
```

Use it for: machine names → entity IDs, `max_reservation_hours`,
`fallback_chunk_hours`, `furthest_future_booking` notes, the reservation `--user`
email, and reservation defaults (title, project, milestone, team, description,
timezone).
Never hard-code duration or future-booking limits in this skill — they belong in the
YAML. If a requested machine isn't listed, query Conductor for it, ask the user
whether to add it, and use conservative chunks only after confirming policy.

## Step 0: Verify Setup

Confirm the CLI works and you're authenticated before doing anything else:

```bash
conductor whoami
```

If it prints `email`, `short_id`, and `teams`, you're good — go to Step 1. This is
the common case; setup is one-time per machine.

If it fails (command not found, a keyring/D-Bus traceback, a `401`, or a DNS error),
read `references/setup.md` and apply the matching fix, then re-run `whoami`.

## Step 1: Identify the Machine

Prefer the `entity_id` already recorded in the YAML. Only query when the machine is
missing or you need to confirm it:

```bash
conductor --output json list system -n 5 \
  --filter 'system_datas.name=<system-name>'
```

Use the top-level `id` of the returned system item as the reservation `--entity`.

## Step 2: Query Reservations

Filter on `dates.*` fields — that is the reliable query path. Fetch future
reservations for the entity, sorted by start. Compute the current UTC time into a
variable and let the shell substitute it — the `date_end` filter needs a real
ISO-8601 timestamp, not a placeholder, or Conductor rejects the query with a
database parse error:

```bash
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
conductor --output json list reservation -n 100 \
  --filter 'dates.entity_id=<entity-id>' \
  --filter "dates.date_end={\"comparator\":\"gt\",\"value\":\"$NOW\"}" \
  --sort dates.date_start --sort-direction ascending
```

Note the `date_end` filter is in double quotes so `$NOW` expands; the other filter
stays single-quoted. A literal `<now-utc>` (or any non-date string) fails with
`There was an error processing your query` / a SQLAlchemy error — that means the
timestamp wasn't substituted.

Filtering notes learned from the platform:

- Do **not** filter with `target_info.name` / `target_info.id`; they appear in
  output but aren't valid reservation filter paths. `dates.target_info.name` can
  work, but `dates.entity_id` is more reliable.
- To find reservations for a specific person, filter `users.user_id=<conductor-user-id>`.
  `users.email` is **not** a supported filter.

## Step 3: Compute Free Time

Don't do the timezone conversion and interval math by hand — it's easy to get DST,
midnight-crossing, and gap-merging wrong, and every run would redo it. Pipe the
query straight through the bundled scripts, which are deterministic. Script paths
below are relative to this skill's directory, so `cd` there first or use absolute
paths:

```bash
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
conductor --output json list reservation -n 100 \
  --filter 'dates.entity_id=<entity-id>' \
  --filter "dates.date_end={\"comparator\":\"gt\",\"value\":\"$NOW\"}" \
  --sort dates.date_start --sort-direction ascending \
| python3 scripts/parse_reservations.py --now "$NOW" \
    --tz '<timezone-from-yaml>' --skip-gap-minutes <skip-gaps-from-yaml> \
    --me "$AMD_EMAIL" \
| python3 scripts/render_report.py --machine '<system-name>'
```

- `parse_reservations.py` reads the CLI JSON on stdin and emits structured JSON:
  merged busy intervals, computed free gaps, and a `warnings` array. Always pass
  `--me "$AMD_EMAIL"` — that env var is already set in the user's shell rc, so the
  identity is read at run time and never stored in the repo. It checks each
  reservation's `users` array (distinct from the creator) and flags the ones the
  user is on, which the report shows as a "我" column. This answers "which of these
  are mine?" and matters because someone else may create a reservation that includes
  you. If the API schema drifts (missing `items`/`users`, unparseable dates), the
  parser warns on stderr and in the JSON rather than silently reporting wrong data —
  surface those warnings to the user.
- `render_report.py` turns that JSON into the report format below. Use it directly
  when the user just wants to see availability.
- When you need the intervals for planning (e.g. to split into chunks), read the
  parser's JSON yourself instead of piping to the renderer.

Then plan the reservation. **Always book the largest policy-legal window first —
never start from a fallback chunk size.** For each contiguous free interval you were
asked to book, compute the end as the *earliest* of these ceilings:

- `start + max_reservation_hours`
- the machine's `furthest_future_booking` horizon (measured from now, e.g. `~48h`)
- the reservation `milestone` from the YAML — `date_end` must be **≤** milestone, or
  the create is rejected before it ever reaches Conductor
- the end of the free interval itself (if the user asked for a bounded window)

Book that single largest chunk first. `fallback_chunk_hours` is **only** for retrying
*after* a real create fails (Step 4) — it is never the starting size. If one chunk
can't cover the whole requested window (e.g. `max_reservation_hours` is smaller than
the window), tile forward with additional max-size chunks. Show the user the busy/free
report plus the planned chunk(s) so they can confirm before anything is created.

## Step 4: Create Reservations

Reservations are shared and hard to undo, so guard each chunk:

1. Re-query overlaps for that exact window (`dates.entity_id`). If anything
   overlaps, skip that chunk and report it.
2. Run `--dry-run`. Treat it as a syntax/eligibility check, not a guarantee.
3. If dry-run passes, run the real create **at the largest-first size computed in
   Step 3**.
4. Only if the real create fails on a duration/future-booking limit, retry the same
   start with the machine's `fallback_chunk_hours` in order (largest first), stopping
   at the first that succeeds. Never begin with a fallback size.
5. Verify by querying reservations again — partial success is possible.

Both create commands share the same flags; add or remove `--dry-run`:

```bash
conductor reservation --dry-run create \
  --title '<title>' --project '<project>' --milestone '<milestone>' \
  --allocation-team '<team>' --entity '<entity-id>' --user "$AMD_EMAIL" \
  --start-date '<start-local-iso8601>' --end-date '<end-local-iso8601>' \
  --description '<description>'
```

To put more than one person on the reservation, repeat `--user` — the flag's help
says singular, but it accepts multiple occurrences and adds each user. Accept an
email, ntid, or short_id. Always include the requester (`$AMD_EMAIL`) plus whoever
they name:

```bash
  ... --user "$AMD_EMAIL" --user '<colleague-email-or-short_id>' ...
```

Everyone listed shows up in the reservation's `users` array, so `--me "$AMD_EMAIL"`
in Step 3 will correctly flag it as yours even when a colleague created it.

If a batch creates multiple chunks, stop at the first failed create and report which
chunks succeeded (`stop_on_first_create_failure` behavior).

## Platform Limits and Gotchas

- **Real create is the source of truth.** `--dry-run` can pass while the create
  fails with `Reservation exceeds furthest future reservation limit.` or
  `Reservation exceeds reservation duration limit.` (sometimes both). This is why
  dry-run alone is never enough.
- On a duration/future-booking failure, retry the chunk with the machine's
  `fallback_chunk_hours` (in order), then stop and report if all fail. Don't invent
  smaller sizes beyond the configured list.
- **`date_end` must be ≤ the `milestone`.** This is validated client-side (a pydantic
  `milestone should be greater than or equal to date_end` error) *before* the request
  is sent, so it isn't a future-booking failure and `fallback_chunk_hours` won't help
  — it just makes the chunk end earlier. Always cap `date_end` at the milestone when
  computing the largest-first window (Step 3). If the user needs time past the
  milestone, ask them to bump the `milestone` in the YAML.
- Always verify after create — never assume success.
- DNS errors mentioning `conductor.amd.com` mean VPN/connectivity is down; stop and
  ask the user to fix it.

## Response Format

Keep reports concise. Times in the config timezone (Beijing by default):

```markdown
当前北京时间：YYYY-MM-DD HH:MM

当前占用：
- HH:MM -> HH:MM | title | creator | 我：是/否

空闲时间：
- YYYY-MM-DD HH:MM -> YYYY-MM-DD HH:MM

已预约：
- YYYY-MM-DD HH:MM -> YYYY-MM-DD HH:MM | Reservation ID: ...

未能预约：
- YYYY-MM-DD HH:MM -> YYYY-MM-DD HH:MM | reason
```
