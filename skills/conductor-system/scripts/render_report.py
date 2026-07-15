#!/usr/bin/env python3
"""Render parse_reservations.py output into the skill's Chinese status report.

Reads the parser's JSON on stdin and prints the 当前占用 / 空闲时间 report the
skill shows the user. Kept separate from the parser so the parsing logic stays
reusable and the presentation can change without touching the interval math.

Usage:
    ... | parse_reservations.py | render_report.py [--machine <name>]
"""

import argparse
import json
import sys
from datetime import datetime


def fmt_dt(iso, with_date=True):
    """Format an ISO local timestamp as 'YYYY-MM-DD HH:MM' or 'HH:MM'."""
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return iso
    return dt.strftime("%Y-%m-%d %H:%M" if with_date else "%H:%M")


def fmt_end(start_iso, end_iso):
    """Show the end as bare HH:MM, but include its date if it differs from start."""
    try:
        s = datetime.fromisoformat(start_iso)
        e = datetime.fromisoformat(end_iso)
    except (ValueError, TypeError):
        return end_iso
    return e.strftime("%H:%M" if s.date() == e.date() else "%Y-%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine", default=None, help="Machine name to show in the header")
    args = ap.parse_args()

    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"无法解析输入 JSON: {e}", file=sys.stderr)
        return 1

    if "error" in data:
        print(f"解析出错: {data['error']}", file=sys.stderr)
        return 1

    warnings = data.get("warnings") or []
    lines = []

    header = "当前北京时间" if data.get("timezone") == "Asia/Shanghai" else "当前时间"
    lines.append(f"{header}：{fmt_dt(data.get('now', ''))}")
    if args.machine:
        lines.append(f"机器：{args.machine}")
    lines.append("")

    lines.append("当前占用：")
    busy = data.get("busy", [])
    reservations = data.get("reservations", [])
    if reservations:
        for r in reservations:
            row = (
                f"- {fmt_dt(r['start'])} -> {fmt_end(r['start'], r['end'])} "
                f"| {r.get('title') or '(no title)'} | {r.get('creator', 'unknown')}"
            )
            me_in = r.get("me_in")
            if me_in is True:
                row += " | 我：是"
            elif me_in is False:
                row += " | 我：否"
            lines.append(row)
    elif busy:
        for b in busy:
            lines.append(f"- {fmt_dt(b['start'])} -> {fmt_end(b['start'], b['end'])}")
    else:
        lines.append("- （无）")
    lines.append("")

    lines.append("空闲时间：")
    free = data.get("free", [])
    for f in free:
        lines.append(f"- {fmt_dt(f['start'])} -> {fmt_dt(f['end'])}")
    open_from = data.get("free_open_ended_from")
    if open_from:
        lines.append(f"- {fmt_dt(open_from)} 起，之后无预约")
    if not free and not open_from:
        lines.append("- （无）")

    skipped = data.get("skipped_short_gaps", 0)
    if skipped:
        lines.append("")
        lines.append(f"（已忽略 {skipped} 个短于 {data.get('skip_gap_minutes')} 分钟的碎片空档）")

    print("\n".join(lines))

    if warnings:
        print("\n⚠ 解析警告（API 格式可能有变，请核对）：", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
