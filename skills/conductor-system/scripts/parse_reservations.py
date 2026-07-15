#!/usr/bin/env python3
"""Parse `conductor --output json list reservation` output into structured intervals.

Reads the raw CLI JSON on stdin and emits a compact JSON object on stdout with
busy intervals (merged) and computed free gaps, all converted to a target
timezone. This exists so the read-side of the skill is deterministic: timezone
conversion, interval merging, and the "skip short gaps" rule are error-prone to do
by hand and should be identical on every run.

The parser is deliberately defensive. Conductor's schema could drift, so instead
of trusting field paths blindly it collects warnings for anything unexpected
(missing items array, unparseable dates, absent fields) and still returns what it
can. Warnings go to stderr AND into the JSON under "warnings" so a caller notices
when the API shape changed rather than silently reporting wrong free time.

Usage:
    conductor --output json list reservation ... | \
        parse_reservations.py [--tz Asia/Shanghai] [--skip-gap-minutes 60] \
                              [--now <iso-utc>]

--now defaults to the current time; pass it explicitly for reproducible output.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None


def _warn(warnings, msg):
    warnings.append(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def parse_utc(value):
    """Parse an ISO-8601 timestamp to an aware UTC datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    s = value.strip()
    # Conductor emits both "...Z" and "...+00:00"; normalize the Z form.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Conductor reservation dates are UTC even when unsuffixed.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def merge_intervals(intervals):
    """Merge overlapping or adjacent (start,end) pairs. Input may be unsorted."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda p: p[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:  # overlap or exactly adjacent
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tz", default="Asia/Shanghai", help="Display timezone (IANA name)")
    ap.add_argument("--skip-gap-minutes", type=int, default=60,
                    help="Ignore free gaps shorter than this many minutes")
    ap.add_argument("--now", default=None,
                    help="Reference 'now' as ISO-8601 UTC; defaults to current time")
    ap.add_argument("--me", default=None,
                    help="Your short_id or email; flags reservations you're a user of")
    args = ap.parse_args()

    me = (args.me or "").strip().lower()

    warnings = []

    if ZoneInfo is None:
        print(json.dumps({"error": "zoneinfo unavailable (needs Python 3.9+)"}))
        return 1
    try:
        tz = ZoneInfo(args.tz)
    except Exception as e:
        print(json.dumps({"error": f"bad timezone {args.tz!r}: {e}"}))
        return 1

    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"error": "no input on stdin"}))
        return 1
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"stdin is not valid JSON: {e}"}))
        return 1

    now = parse_utc(args.now) if args.now else datetime.now(timezone.utc)
    if now is None:
        _warn(warnings, f"could not parse --now {args.now!r}; using current time")
        now = datetime.now(timezone.utc)

    # The CLI wraps results as {"ok": true, "items": [...], "count": N}. Tolerate
    # a bare list too, in case a future version changes the envelope.
    if isinstance(data, dict):
        if data.get("ok") is False:
            _warn(warnings, f"CLI reported ok=false: {data.get('error') or data}")
        items = data.get("items")
        if items is None:
            _warn(warnings, "no 'items' key in CLI output; schema may have changed")
            items = []
    elif isinstance(data, list):
        items = data
    else:
        print(json.dumps({"error": "unexpected top-level JSON type",
                          "warnings": warnings}))
        return 1

    busy = []
    reservations = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            _warn(warnings, f"item {idx} is not an object; skipped")
            continue
        start = parse_utc(item.get("date_start"))
        end = parse_utc(item.get("date_end"))
        if start is None or end is None:
            _warn(warnings, f"item {idx} (id={item.get('id')}) has unparseable "
                            f"date_start/date_end; skipped")
            continue
        if end <= start:
            _warn(warnings, f"item {idx} (id={item.get('id')}) has end <= start; skipped")
            continue

        creator = item.get("creator") or {}
        if not isinstance(creator, dict):
            creator = {}
        who = creator.get("short_id") or creator.get("email") or "unknown"

        # A reservation has its own users array, distinct from the creator; the
        # user we care about may be a listed user without being the creator.
        users_raw = item.get("users")
        if users_raw is None:
            users_raw = []
            _warn(warnings, f"item {idx} (id={item.get('id')}) has no 'users' key")
        users = []
        for u in users_raw:
            if isinstance(u, dict):
                users.append(u.get("short_id") or u.get("email") or "unknown")
        me_in = None
        if me:
            me_in = any(
                me in (str(x).lower() for x in (u.get("short_id"), u.get("email")) if x)
                for u in users_raw if isinstance(u, dict)
            )

        reservations.append({
            "id": item.get("id"),
            "title": item.get("title") or "",
            "creator": who,
            "users": users,
            "me_in": me_in,
            "start": start.astimezone(tz).isoformat(),
            "end": end.astimezone(tz).isoformat(),
        })
        busy.append((start, end))

    merged = merge_intervals(busy)

    # Free gaps between merged busy intervals, from `now` onward.
    threshold = timedelta(minutes=args.skip_gap_minutes)
    free = []
    skipped_short = 0
    cursor = now
    for start, end in merged:
        if end <= now:
            continue
        gap_start = max(cursor, now)
        if start > gap_start:
            if start - gap_start >= threshold:
                free.append((gap_start, start))
            else:
                skipped_short += 1
        cursor = max(cursor, end)
    # Trailing open-ended free time after the last busy interval.
    open_from = max(cursor, now)

    def to_local(pairs):
        return [{"start": s.astimezone(tz).isoformat(),
                 "end": e.astimezone(tz).isoformat()} for s, e in pairs]

    result = {
        "timezone": args.tz,
        "now": now.astimezone(tz).isoformat(),
        "reservation_count": len(reservations),
        "reservations": reservations,
        "busy": to_local(merged),
        "free": to_local(free),
        "free_open_ended_from": open_from.astimezone(tz).isoformat(),
        "skipped_short_gaps": skipped_short,
        "skip_gap_minutes": args.skip_gap_minutes,
        "warnings": warnings,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
