#!/usr/bin/env python3
"""
Generate Markdown benchmark report from CSV results.

Reads one or more CSV files produced by bench_sglang.py, merges them,
and generates a comprehensive Markdown report with comparison tables,
best-result highlights, and optimization suggestions.

Usage:
    python3 generate_report.py \
        --csv results_TP8_DP1.csv results_TP4_DP2.csv \
        --model "DeepSeek-R1-0528" \
        --num-gpus 8 \
        --output benchmark_report.md
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime


def load_results(*csv_paths):
    """Load and merge results from one or more CSV files."""
    all_results = []
    for path in csv_paths:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("error"):
                    continue  # skip failed runs
                all_results.append(row)
    return all_results


def safe_float(val, default=None):
    """Safely convert to float, returning default on failure."""
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def fmt(val, precision=2):
    """Format a numeric value for display."""
    f = safe_float(val)
    if f is None:
        return "N/A"
    if f >= 10000:
        return f"{f:,.0f}"
    if f >= 100:
        return f"{f:,.1f}"
    return f"{f:.{precision}f}"


def generate_report(results, model_name, num_gpus, mtp_enabled=False, gpu_model=""):
    """Generate the full Markdown report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# SGLang Benchmark Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Model**: {model_name}")
    if gpu_model:
        lines.append(f"- **GPU**: {gpu_model}")
    lines.append(f"- **Number of GPUs**: {num_gpus}")
    lines.append(f"- **MTP**: {'Enabled' if mtp_enabled else 'Disabled'}")
    lines.append(f"- **Mode**: Mix (non-disaggregated)")
    lines.append(f"- **Generated**: {now}")
    lines.append("")

    # Group by config
    configs = defaultdict(list)
    for r in results:
        configs[r.get("config", "default")].append(r)

    # List configs tested
    lines.append("## Parallel Configurations Tested")
    lines.append("")
    for config_name in configs:
        count = len(configs[config_name])
        lines.append(f"- **{config_name}**: {count} benchmark runs")
    lines.append("")

    # Per-config detailed tables
    for config_name, rows in configs.items():
        lines.append(f"## Results: {config_name}")
        lines.append("")

        # Sort rows by ISL, then OSL, then concurrency
        rows.sort(key=lambda r: (
            safe_float(r.get("isl"), 0),
            safe_float(r.get("osl"), 0),
            safe_float(r.get("concurrency"), 0),
        ))

        lines.append(
            "| ISL | OSL | CON | TTFT(ms) | TPOT(ms) | "
            "Input(tok/s) | Output(tok/s) | Total(tok/s) | "
            "Per-GPU Total(tok/s) | Per-GPU Output(tok/s) |"
        )
        lines.append(
            "|----:|----:|----:|---------:|---------:|"
            "-------------:|--------------:|-------------:|"
            "--------------------:|---------------------:|"
        )

        for r in rows:
            lines.append(
                f"| {r.get('isl', '')} "
                f"| {r.get('osl', '')} "
                f"| {r.get('concurrency', '')} "
                f"| {fmt(r.get('mean_ttft_ms'))} "
                f"| {fmt(r.get('mean_tpot_ms'))} "
                f"| {fmt(r.get('input_throughput'))} "
                f"| {fmt(r.get('output_throughput'))} "
                f"| {fmt(r.get('total_throughput'))} "
                f"| {fmt(r.get('per_gpu_total_throughput'))} "
                f"| {fmt(r.get('per_gpu_output_throughput'))} |"
            )

        lines.append("")

        # P99 latency table
        lines.append(f"### Latency Percentiles: {config_name}")
        lines.append("")
        lines.append(
            "| ISL | OSL | CON | "
            "TTFT p50(ms) | TTFT p99(ms) | "
            "TPOT p50(ms) | TPOT p99(ms) |"
        )
        lines.append(
            "|----:|----:|----:|"
            "-------------:|-------------:|"
            "-------------:|-------------:|"
        )
        for r in rows:
            lines.append(
                f"| {r.get('isl', '')} "
                f"| {r.get('osl', '')} "
                f"| {r.get('concurrency', '')} "
                f"| {fmt(r.get('median_ttft_ms'))} "
                f"| {fmt(r.get('p99_ttft_ms'))} "
                f"| {fmt(r.get('median_tpot_ms'))} "
                f"| {fmt(r.get('p99_tpot_ms'))} |"
            )
        lines.append("")

    # Cross-config comparison (if multiple configs)
    if len(configs) > 1:
        lines.append("## Cross-Config Comparison")
        lines.append("")

        # Average metrics per config
        lines.append("### Average Metrics by Config")
        lines.append("")
        lines.append(
            "| Config | Avg TTFT(ms) | Avg TPOT(ms) | "
            "Avg Total(tok/s) | Avg Per-GPU(tok/s) |"
        )
        lines.append(
            "|--------|-------------:|-------------:|"
            "-----------------:|-------------------:|"
        )

        config_avgs = {}
        for config_name, rows in configs.items():
            ttfts = [safe_float(r.get("mean_ttft_ms")) for r in rows
                     if safe_float(r.get("mean_ttft_ms")) is not None]
            tpots = [safe_float(r.get("mean_tpot_ms")) for r in rows
                     if safe_float(r.get("mean_tpot_ms")) is not None]
            totals = [safe_float(r.get("total_throughput")) for r in rows
                      if safe_float(r.get("total_throughput")) is not None]
            per_gpus = [safe_float(r.get("per_gpu_total_throughput")) for r in rows
                        if safe_float(r.get("per_gpu_total_throughput")) is not None]

            avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
            avg_tpot = sum(tpots) / len(tpots) if tpots else None
            avg_total = sum(totals) / len(totals) if totals else None
            avg_pgpu = sum(per_gpus) / len(per_gpus) if per_gpus else None

            config_avgs[config_name] = {
                "avg_ttft": avg_ttft,
                "avg_tpot": avg_tpot,
                "avg_total": avg_total,
                "avg_per_gpu": avg_pgpu,
            }

            lines.append(
                f"| {config_name} "
                f"| {fmt(avg_ttft)} "
                f"| {fmt(avg_tpot)} "
                f"| {fmt(avg_total)} "
                f"| {fmt(avg_pgpu)} |"
            )
        lines.append("")

        # Per ISL/OSL comparison across configs
        lines.append("### Head-to-Head: Total Throughput by Workload")
        lines.append("")
        config_names = list(configs.keys())
        header = "| ISL | OSL | CON | " + " | ".join(
            f"{c} (tok/s)" for c in config_names
        ) + " | Best |"
        separator = "|----:|----:|----:|" + "|".join(
            "-" * (len(c) + 9) + ":" for c in config_names
        ) + "|------|"
        lines.append(header)
        lines.append(separator)

        # Group all results by (ISL, OSL, CON) key
        by_key = defaultdict(dict)
        for config_name, rows in configs.items():
            for r in rows:
                key = (r.get("isl", ""), r.get("osl", ""), r.get("concurrency", ""))
                by_key[key][config_name] = r

        for key in sorted(by_key.keys(), key=lambda k: (
            safe_float(k[0], 0), safe_float(k[1], 0), safe_float(k[2], 0)
        )):
            isl, osl, con = key
            vals = {}
            for c in config_names:
                r = by_key[key].get(c)
                vals[c] = safe_float(r.get("total_throughput")) if r else None

            best = ""
            valid_vals = {c: v for c, v in vals.items() if v is not None}
            if valid_vals:
                best = max(valid_vals, key=valid_vals.get)

            row_parts = [f"| {isl} | {osl} | {con} "]
            for c in config_names:
                v = vals[c]
                marker = " **" if c == best and len(valid_vals) > 1 else " "
                end_marker = "** " if c == best and len(valid_vals) > 1 else " "
                row_parts.append(f"|{marker}{fmt(v)}{end_marker}")
            row_parts.append(f"| {best} |")
            lines.append("".join(row_parts))

        lines.append("")

    # Best results summary
    lines.append("## Best Results")
    lines.append("")

    valid_results = [r for r in results if safe_float(r.get("total_throughput"))]

    if valid_results:
        # Best total throughput
        best_tp = max(valid_results, key=lambda r: safe_float(r["total_throughput"]))
        lines.append(
            f"- **Highest Total Throughput**: {fmt(best_tp['total_throughput'])} tok/s "
            f"({best_tp.get('config', '')}, ISL={best_tp.get('isl')}, "
            f"OSL={best_tp.get('osl')}, CON={best_tp.get('concurrency')})"
        )

        # Best per-GPU throughput
        best_pgpu = [r for r in results
                     if safe_float(r.get("per_gpu_total_throughput"))]
        if best_pgpu:
            best_pg = max(best_pgpu,
                          key=lambda r: safe_float(r["per_gpu_total_throughput"]))
            lines.append(
                f"- **Highest Per-GPU Throughput**: "
                f"{fmt(best_pg['per_gpu_total_throughput'])} tok/s/GPU "
                f"({best_pg.get('config', '')}, ISL={best_pg.get('isl')}, "
                f"OSL={best_pg.get('osl')}, CON={best_pg.get('concurrency')})"
            )

        # Best output throughput
        best_out = [r for r in results if safe_float(r.get("output_throughput"))]
        if best_out:
            best_o = max(best_out,
                         key=lambda r: safe_float(r["output_throughput"]))
            lines.append(
                f"- **Highest Output Throughput**: "
                f"{fmt(best_o['output_throughput'])} tok/s "
                f"({best_o.get('config', '')}, ISL={best_o.get('isl')}, "
                f"OSL={best_o.get('osl')}, CON={best_o.get('concurrency')})"
            )

    valid_ttft = [r for r in results if safe_float(r.get("mean_ttft_ms"))]
    if valid_ttft:
        best_ttft = min(valid_ttft, key=lambda r: safe_float(r["mean_ttft_ms"]))
        lines.append(
            f"- **Lowest TTFT**: {fmt(best_ttft['mean_ttft_ms'])} ms "
            f"({best_ttft.get('config', '')}, ISL={best_ttft.get('isl')}, "
            f"OSL={best_ttft.get('osl')}, CON={best_ttft.get('concurrency')})"
        )

    valid_tpot = [r for r in results if safe_float(r.get("mean_tpot_ms"))]
    if valid_tpot:
        best_tpot = min(valid_tpot, key=lambda r: safe_float(r["mean_tpot_ms"]))
        lines.append(
            f"- **Lowest TPOT**: {fmt(best_tpot['mean_tpot_ms'])} ms "
            f"({best_tpot.get('config', '')}, ISL={best_tpot.get('isl')}, "
            f"OSL={best_tpot.get('osl')}, CON={best_tpot.get('concurrency')})"
        )

    lines.append("")

    # Optimization suggestions
    lines.append("## Optimization Suggestions")
    lines.append("")
    suggestions = analyze_and_suggest(results, num_gpus, configs, mtp_enabled)
    for s in suggestions:
        lines.append(f"- {s}")
    lines.append("")

    return "\n".join(lines)


def analyze_and_suggest(results, num_gpus, configs, mtp_enabled):
    """Analyze benchmark results and generate optimization suggestions."""
    suggestions = []

    # 1. TP vs DP comparison
    if len(configs) > 1:
        config_totals = {}
        for config_name, rows in configs.items():
            vals = [safe_float(r.get("total_throughput")) for r in rows
                    if safe_float(r.get("total_throughput")) is not None]
            if vals:
                config_totals[config_name] = sum(vals) / len(vals)

        if len(config_totals) >= 2:
            best_cfg = max(config_totals, key=config_totals.get)
            worst_cfg = min(config_totals, key=config_totals.get)
            if config_totals[worst_cfg] > 0:
                improvement = (
                    (config_totals[best_cfg] - config_totals[worst_cfg])
                    / config_totals[worst_cfg] * 100
                )
                suggestions.append(
                    f"**{best_cfg}** outperforms **{worst_cfg}** by "
                    f"{improvement:.1f}% on average total throughput. "
                    f"Consider using **{best_cfg}** as the production config."
                )

    # 2. Concurrency saturation detection
    for config_name, rows in configs.items():
        # Group by (ISL, OSL)
        by_shape = defaultdict(list)
        for r in rows:
            key = (r.get("isl"), r.get("osl"))
            by_shape[key].append(r)

        for (isl, osl), shape_rows in by_shape.items():
            sorted_rows = sorted(
                shape_rows,
                key=lambda r: safe_float(r.get("concurrency"), 0)
            )
            if len(sorted_rows) >= 3:
                tps = [
                    (safe_float(r.get("concurrency")),
                     safe_float(r.get("total_throughput")))
                    for r in sorted_rows
                    if safe_float(r.get("total_throughput")) is not None
                ]
                if len(tps) >= 3:
                    # Check if throughput plateaus
                    peak_idx = max(range(len(tps)), key=lambda i: tps[i][1])
                    if peak_idx < len(tps) - 1:
                        peak_con = tps[peak_idx][0]
                        peak_tp = tps[peak_idx][1]
                        last_tp = tps[-1][1]
                        if last_tp < peak_tp * 0.95:
                            suggestions.append(
                                f"Throughput saturates around CON={int(peak_con)} "
                                f"for ISL={isl}, OSL={osl} ({config_name}). "
                                f"Higher concurrency degrades performance. "
                                f"Consider capping concurrency at ~{int(peak_con)}."
                            )
                            break  # one per config

    # 3. High TTFT warning
    high_ttft_found = False
    for r in results:
        ttft = safe_float(r.get("mean_ttft_ms"))
        if ttft and ttft > 3000:
            suggestions.append(
                f"High TTFT detected ({fmt(ttft)}ms at ISL={r.get('isl')}, "
                f"CON={r.get('concurrency')}). Consider enabling chunked prefill "
                f"(`--chunked-prefill-size 8192`) or reducing concurrency."
            )
            high_ttft_found = True
            break
    if not high_ttft_found:
        for r in results:
            ttft = safe_float(r.get("mean_ttft_ms"))
            if ttft and ttft > 1500:
                suggestions.append(
                    f"Moderate TTFT ({fmt(ttft)}ms at ISL={r.get('isl')}, "
                    f"CON={r.get('concurrency')}). Chunked prefill "
                    f"(`--chunked-prefill-size`) may help reduce first-token latency."
                )
                break

    # 4. High TPOT warning
    for r in results:
        tpot = safe_float(r.get("mean_tpot_ms"))
        if tpot and tpot > 100:
            suggestions.append(
                f"High TPOT detected ({fmt(tpot)}ms at CON={r.get('concurrency')}, "
                f"{r.get('config', '')}). This suggests decode is memory-bandwidth "
                f"bound. Consider: increasing TP for better per-token latency, "
                f"reducing batch size, or enabling MTP if supported."
            )
            break

    # 5. MTP suggestion
    if not mtp_enabled:
        suggestions.append(
            "MTP was disabled for this baseline. If the model supports MTP "
            "(DeepSeek-R1/V3, Qwen3), re-run with `--enable-mtp` to compare "
            "decode throughput improvement."
        )

    # 6. General AMD-specific suggestions
    suggestions.append(
        "Verify AITER CK GEMM kernels are tuned for this model's shapes "
        "(use `aiter-ck-gemm-tune` skill). Untuned kernels can cost 10-30% "
        "throughput."
    )
    suggestions.append(
        "For production deployments, evaluate PD-disaggregation to separate "
        "prefill and decode workloads for better latency isolation under load."
    )

    return suggestions


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown benchmark report from CSV results"
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="Input CSV file(s) from bench_sglang.py"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--gpu-model", default="", help="GPU model name (e.g., MI355X)")
    parser.add_argument("--mtp", action="store_true", help="MTP was enabled")
    parser.add_argument(
        "--output", default="benchmark_report.md",
        help="Output Markdown file path"
    )
    args = parser.parse_args()

    results = load_results(*args.csv)
    if not results:
        print("Error: no valid results found in CSV files")
        sys.exit(1)

    report = generate_report(
        results, args.model, args.num_gpus,
        mtp_enabled=args.mtp, gpu_model=args.gpu_model
    )

    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report saved to {args.output}")
    print()
    print(report)


if __name__ == "__main__":
    main()
