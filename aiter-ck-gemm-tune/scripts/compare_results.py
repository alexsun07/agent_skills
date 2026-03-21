#!/usr/bin/env python3
"""Compare before/after benchmark results from aiter GEMM test logs.

Parses two log files (bench_before and bench_after) and produces a
comparison table showing TFLOPS changes and speedup percentages.

Usage:
  python3 compare_results.py <before_log> <after_log>

The logs should be output from aiter's op_tests/test_gemm_*.py scripts,
which produce markdown-formatted tables with columns like:
  dtype | m | n | k | ... | ck us | ck TFLOPS | ...
"""

import argparse
import re
import sys
from collections import defaultdict


def parse_log(log_file):
    """Parse a benchmark log and return {(m, n, k): {col: value}} dict.

    Handles the markdown table format from aiter test scripts.
    """
    results = {}
    headers = None

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue

            cells = [c.strip() for c in line.split("|")[1:-1]]

            if not cells:
                continue

            # Detect header row
            if "dtype" in cells[0].lower() or "m" in [c.lower() for c in cells]:
                headers = [c.strip() for c in cells]
                continue

            # Skip separator rows
            if headers is None or all(
                set(c.strip()) <= set("-: ") for c in cells
            ):
                continue

            if len(cells) != len(headers):
                continue

            row = {}
            for h, v in zip(headers, cells):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = v

            # Extract key (m, n, k) — handle different column name cases
            m = row.get("m")
            n = row.get("n")
            k = row.get("k")

            if m is None or n is None or k is None:
                continue

            key = (int(m), int(n), int(k))

            # Build a hashable sub-key for preshuffle variants
            preshuffle = row.get("ck_preshuffle", "")
            full_key = (key, str(preshuffle))

            results[full_key] = row

    return headers, results


def main():
    parser = argparse.ArgumentParser(
        description="Compare before/after benchmark results"
    )
    parser.add_argument("before_log", help="Path to bench_before log file")
    parser.add_argument("after_log", help="Path to bench_after log file")
    parser.add_argument(
        "--metric",
        default="ck TFLOPS",
        help='Metric column to compare (default: "ck TFLOPS")',
    )
    args = parser.parse_args()

    _, before = parse_log(args.before_log)
    _, after = parse_log(args.after_log)

    if not before:
        print(f"ERROR: No results parsed from {args.before_log}", file=sys.stderr)
        sys.exit(1)
    if not after:
        print(f"ERROR: No results parsed from {args.after_log}", file=sys.stderr)
        sys.exit(1)

    metric = args.metric

    # Find common keys
    common = sorted(set(before.keys()) & set(after.keys()))

    if not common:
        print("ERROR: No common shapes found between the two logs.", file=sys.stderr)
        print(f"  Before log has {len(before)} shapes", file=sys.stderr)
        print(f"  After log has {len(after)} shapes", file=sys.stderr)
        sys.exit(1)

    # Print comparison table
    has_preshuffle = any(k[1] != "" for k in common)

    if has_preshuffle:
        print(
            f"| {'M':>6} | {'N':>6} | {'K':>5} | {'preshuffle':>10} | "
            f"{'Before':>10} | {'After':>10} | {'Speedup':>8} |"
        )
        print(
            f"|{'-'*8}|{'-'*8}|{'-'*7}|{'-'*12}|"
            f"{'-'*12}|{'-'*12}|{'-'*10}|"
        )
    else:
        print(
            f"| {'M':>6} | {'N':>6} | {'K':>5} | "
            f"{'Before':>10} | {'After':>10} | {'Speedup':>8} |"
        )
        print(
            f"|{'-'*8}|{'-'*8}|{'-'*7}|"
            f"{'-'*12}|{'-'*12}|{'-'*10}|"
        )

    M_CATEGORIES = [
        ("small M (1-63, decode)", lambda m: m <= 63),
        ("medium M (64-512)", lambda m: 64 <= m <= 512),
        ("large M (>512, prefill)", lambda m: m > 512),
    ]

    speedups = []
    by_nk = defaultdict(list)
    by_nk_mcat = defaultdict(lambda: defaultdict(list))

    for key in common:
        (m, n, k), preshuffle = key
        bval = before[key].get(metric)
        aval = after[key].get(metric)

        if bval is None or aval is None:
            continue
        if not isinstance(bval, (int, float)) or not isinstance(aval, (int, float)):
            continue
        if bval == 0:
            continue

        speedup = (aval - bval) / bval * 100
        speedups.append(speedup)
        by_nk[(n, k)].append(speedup)

        for cat_name, cat_fn in M_CATEGORIES:
            if cat_fn(m):
                by_nk_mcat[(n, k)][cat_name].append(speedup)
                break

        sign = "+" if speedup >= 0 else ""

        if has_preshuffle:
            print(
                f"| {m:>6} | {n:>6} | {k:>5} | {preshuffle:>10} | "
                f"{bval:>10.1f} | {aval:>10.1f} | {sign}{speedup:>6.1f}% |"
            )
        else:
            print(
                f"| {m:>6} | {n:>6} | {k:>5} | "
                f"{bval:>10.1f} | {aval:>10.1f} | {sign}{speedup:>6.1f}% |"
            )

    if not speedups:
        print(f"\nNo comparable {metric} values found.", file=sys.stderr)
        sys.exit(1)

    # Summary
    avg = sum(speedups) / len(speedups)
    improved = sum(1 for s in speedups if s > 1)
    regressed = sum(1 for s in speedups if s < -1)

    print(f"\n--- Summary ({metric}) ---")
    print(f"Shapes compared: {len(speedups)}")
    print(f"Average speedup: {'+' if avg >= 0 else ''}{avg:.1f}%")
    print(f"Min speedup:     {'+' if min(speedups) >= 0 else ''}{min(speedups):.1f}%")
    print(f"Max speedup:     {'+' if max(speedups) >= 0 else ''}{max(speedups):.1f}%")
    print(f"Improved (>1%):  {improved}/{len(speedups)}")
    print(f"Regressed (<-1%): {regressed}/{len(speedups)}")

    print(f"\n--- Per (N, K) breakdown ---")
    for (n, k), sps in sorted(by_nk.items()):
        avg_nk = sum(sps) / len(sps)
        print(
            f"  N={n}, K={k}: avg {'+' if avg_nk >= 0 else ''}{avg_nk:.1f}%, "
            f"min {'+' if min(sps) >= 0 else ''}{min(sps):.1f}%, "
            f"max {'+' if max(sps) >= 0 else ''}{max(sps):.1f}%"
        )

    print(f"\n--- Per (N, K) by M category ---")
    for cat_name, _ in M_CATEGORIES:
        print(f"\n  [{cat_name}]")
        for (n, k) in sorted(by_nk_mcat.keys()):
            sps = by_nk_mcat[(n, k)].get(cat_name)
            if not sps:
                continue
            avg_c = sum(sps) / len(sps)
            print(
                f"    N={n}, K={k}: avg {'+' if avg_c >= 0 else ''}{avg_c:.1f}%, "
                f"min {'+' if min(sps) >= 0 else ''}{min(sps):.1f}%, "
                f"max {'+' if max(sps) >= 0 else ''}{max(sps):.1f}%"
            )


if __name__ == "__main__":
    main()
