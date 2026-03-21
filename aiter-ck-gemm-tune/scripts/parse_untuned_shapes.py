#!/usr/bin/env python3
"""Parse aiter logs to extract unique untuned GEMM shapes (N, K) per kernel variant.

Scans log files for lines matching:
  shape is M:<val>, N:<val>, K:<val>, not found tuned config in <path>_tuned_gemm.csv, will use default config!

Outputs deduplicated (N, K) pairs grouped by kernel variant.

Usage:
  python3 parse_untuned_shapes.py <log_file>
  python3 parse_untuned_shapes.py <log_file> --variant a8w8_blockscale --csv output.csv --m-sweep
  python3 parse_untuned_shapes.py <log_file> --variant all --csv output.csv --m-sweep

Options:
  --variant <name>  Filter to a specific variant, or "all" for all variants.
                    If omitted, prints summary only (no CSV).
  --csv <file>      Write results to CSV file (M,N,K format for tuning input).
                    Requires --variant.
  --m-sweep         Generate M sweep (1,2,4,...,32768) for each (N,K) pair in CSV output.
"""

import argparse
import re
import sys
from collections import defaultdict


LOG_PATTERN = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+), not found tuned config in .*/(\w+)_tuned_gemm"
)

M_SWEEP = [2**i for i in range(16)]  # 1, 2, 4, ..., 32768


def parse_log(log_file):
    """Parse a log file and return {variant: set of (N, K)} pairs."""
    variants = defaultdict(set)
    with open(log_file, "r") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                n, k = int(match.group(2)), int(match.group(3))
                variant = match.group(4)
                variants[variant].add((n, k))
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Parse aiter logs for untuned GEMM shapes"
    )
    parser.add_argument("log_file", help="Path to the log file to parse")
    parser.add_argument(
        "--variant",
        help='Filter to a specific kernel variant, or "all" for all variants',
    )
    parser.add_argument(
        "--csv", help="Output CSV file path (requires --variant)"
    )
    parser.add_argument(
        "--m-sweep",
        action="store_true",
        help="Generate M sweep (1,2,4,...,32768) for each (N,K) in CSV",
    )
    args = parser.parse_args()

    variants = parse_log(args.log_file)

    if not variants:
        print("No untuned shapes found in log file.", file=sys.stderr)
        sys.exit(1)

    # Print summary to stdout (always)
    print(f"Found {len(variants)} kernel variant(s) with untuned shapes:\n")
    for variant, shapes in sorted(variants.items()):
        print(f"  Variant: {variant}")
        print(f"    Unique (N, K) pairs: {len(shapes)}")
        for n, k in sorted(shapes):
            print(f"      N={n}, K={k}")
        print()

    if len(variants) > 1:
        print(
            "Multiple variants found. Use --variant <name> to select one, "
            "or --variant all to include all.",
        )

    # Write CSV if requested
    if args.csv:
        if not args.variant:
            print(
                "ERROR: --csv requires --variant to specify which variant(s) to output.",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.variant == "all":
            selected = variants
        elif args.variant in variants:
            selected = {args.variant: variants[args.variant]}
        else:
            print(
                f"ERROR: Variant '{args.variant}' not found in log. "
                f"Available: {', '.join(sorted(variants.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(args.csv, "w") as f:
            f.write("M,N,K\n")
            for variant, shapes in sorted(selected.items()):
                for n, k in sorted(shapes):
                    if args.m_sweep:
                        for m in M_SWEEP:
                            f.write(f"{m},{n},{k}\n")
                    else:
                        f.write(f"0,{n},{k}\n")

        total_nk = sum(len(s) for s in selected.values())
        total_rows = total_nk * (len(M_SWEEP) if args.m_sweep else 1)
        print(f"CSV written to: {args.csv}")
        print(
            f"  Variant(s): {', '.join(sorted(selected.keys()))}"
        )
        print(f"  Unique (N,K) pairs: {total_nk}")
        print(f"  Total rows: {total_rows}")


if __name__ == "__main__":
    main()
