#!/usr/bin/env python3
"""把采到的 trace 拆开看一眼，判定它到底是不是"稳态 prefill 的真 trace"。

采集脚本跑完必须调这个，**不看文件存不存在就下结论**。我们踩过的坑是
1.4KB / 57 个事件 / 唯一 kernel 是 hipDeviceSynchronize 的 trace —— 文件在，
大小非零，`ls` 看着一切正常，其实引擎全程空转。

判定分三层，逐层收紧：

  1. 文件层   每个 rank 的 .pt.trace.json.gz 至少 MIN_BYTES（默认 64KB）。
  2. 事件层   cat=="kernel" 的事件数 >= MIN_KERNELS，且**去重后的 kernel 名字**
              >= MIN_DISTINCT。只有一两个名字基本就是空转 + 同步原语。
  3. 语义层   a) kernel 名字里要认得出 gemm / moe / attention 这三类里的至少两类，
                 否则采到的不是 transformer 前向。
              b) vLLM 每个引擎迭代都会套一个 user_annotation：
                     execute_context_<N>(<T>)_generation_<M>(<G>)
                 N/T = 这一步有几个 context(prefill) 请求、多少 prefill token。
                 这是**从 trace 内部**证明"采到的是 prefill 稳态"的直接证据 ——
                 不用去信外部的时间戳或者 /metrics 快照。

退出码非 0 = 这份 trace 不可用，别往账本里收。
"""

import argparse
import collections
import gzip
import io
import json
import os
import re
import sys

ANNOT_RE = re.compile(
    r"execute_(?:(?P<tot>\d+)_)?context_(?P<nctx>\d+)\((?:sq)?(?P<ctxtok>\d+)"
)

# kernel 名字里的类别关键字。宽一点，不同后端命名差别很大。
FAMILY_PATTERNS = {
    "gemm": r"gemm|matmul|hgemm|ck_tile|_mm_|wvSplitK|cijk|mfma|dot_",
    "moe": r"moe|expert|topk|routing|silu_and_mul|fused_mul",
    "attention": r"attn|attention|flash|paged|mla|fmha|softmax",
    "norm": r"norm|rms",
    "comm": r"all_?reduce|nccl|rccl|all_?gather|reduce_scatter|quick_reduce",
    "quant": r"quant|fp8|mxfp4|scaled|e8m0|shuffle",
}

# 纯同步/空转原语。如果 kernel 事件几乎全是这些，就是采空了。
IDLE_ONLY = re.compile(r"DeviceSynchronize|StreamSynchronize|EventRecord|^Memset$", re.I)


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as fh:
        return json.load(io.TextIOWrapper(fh, encoding="utf-8"))


def analyse(path, args):
    size = os.path.getsize(path)
    doc = load(path)
    events = doc.get("traceEvents", [])

    by_cat = collections.Counter(e.get("cat", "<none>") for e in events)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    kdur = collections.Counter()
    kcnt = collections.Counter()
    for e in kernels:
        kdur[e.get("name", "?")] += e.get("dur", 0)
        kcnt[e.get("name", "?")] += 1

    families = {
        fam: sum(c for n, c in kcnt.items() if re.search(pat, n, re.I))
        for fam, pat in FAMILY_PATTERNS.items()
    }

    iters = []
    for e in events:
        if e.get("cat") != "user_annotation":
            continue
        m = ANNOT_RE.search(e.get("name", ""))
        if m:
            iters.append(
                {
                    "nctx": int(m.group("nctx")),
                    "ctxtok": int(m.group("ctxtok")),
                    "dur_ms": e.get("dur", 0) / 1000.0,
                }
            )

    gpu_us = sum(e.get("dur", 0) for e in kernels)
    span_us = 0
    if kernels:
        t0 = min(e["ts"] for e in kernels)
        t1 = max(e["ts"] + e.get("dur", 0) for e in kernels)
        span_us = t1 - t0

    res = {
        "path": path,
        "size": size,
        "n_events": len(events),
        "by_cat": by_cat,
        "n_kernels": len(kernels),
        "distinct_kernels": len(kcnt),
        "kdur": kdur,
        "kcnt": kcnt,
        "families": families,
        "iters": iters,
        "live_iters": [],
        "gpu_us": gpu_us,
        "span_us": span_us,
        "fail": [],
        "warn": [],
    }

    if size < args.min_bytes:
        res["fail"].append(f"文件只有 {size}B < {args.min_bytes}B —— 基本就是采空了")
    if size > args.max_bytes:
        res["fail"].append(
            f"文件 {size/1e6:.0f}MB > {args.max_bytes/1e6:.0f}MB —— 调小 PROFILE_MAX_ITERS"
        )
    if len(kernels) < args.min_kernels:
        res["fail"].append(f"GPU kernel 事件只有 {len(kernels)} < {args.min_kernels}")
    if len(kcnt) < args.min_distinct:
        res["fail"].append(
            f"去重后只有 {len(kcnt)} 种 kernel < {args.min_distinct} —— 像是空转"
        )
    real_kernels = sum(c for n, c in kcnt.items() if not IDLE_ONLY.search(n))
    if real_kernels == 0:
        res["fail"].append("kernel 里没有一个是真算子（全是同步原语）")
    hit = [f for f in ("gemm", "moe", "attention") if families.get(f, 0) > 0]
    if len(hit) < 2:
        res["fail"].append(
            f"gemm/moe/attention 只认出 {hit} —— 不像是 transformer 前向"
        )
    if args.expect_prefill:
        if not iters:
            res["fail"].append(
                "没有 execute_context_* 标注 —— 无法从 trace 内部确认是 prefill 稳态"
            )
        else:
            live_iters = [i for i in iters if i["nctx"] > 0]
            res["live_iters"] = live_iters
            # 判据是"**真带 prefill 的迭代够不够多**"，不是"空迭代占比低不低"。
            # 空迭代 = 调度器这一步没排到活（前端还在 tokenize 75000 token 的 prompt），
            # 它在 trace 里是个 0.2ms 的壳子，一个 kernel 都不产生，
            # 既不污染 kernel 统计也不占体积 —— 拿它当失败条件是抓错了东西。
            # 真要防的"采空了"由上面的 kernel 层判据兜着。
            if len(live_iters) < args.min_prefill_iters:
                res["fail"].append(
                    f"只有 {len(live_iters)} 个迭代真带了 prefill（< {args.min_prefill_iters}）"
                    " —— 窗口太窄，调大 PROFILE_MAX_ITERS"
                )
            if sum(i["ctxtok"] for i in live_iters) == 0:
                res["fail"].append("窗口内 prefill token 总数为 0")
            # 空迭代占比只是提示：它说明 max_iterations 的预算被空转吃掉了，
            # 想要 N 个真 prefill 迭代就得把 max_iterations 开大到 N/(1-占比)。
            idle_frac = 1 - len(live_iters) / len(iters)
            if idle_frac > args.warn_idle_frac:
                res["warn"].append(
                    f"{len(iters)-len(live_iters)}/{len(iters)} 个迭代是空的（{idle_frac:.0%}）——"
                    f" max_iterations 有将近一半预算花在空转上，想要 N 个真迭代"
                    f" 就设 max_iterations ≈ {args.min_prefill_iters}/(1-{idle_frac:.2f})"
                )
    return res


def report(res, args):
    print(f"\n=== {os.path.basename(res['path'])} ===")
    print(f"  大小            {res['size']/1e6:.2f} MB ({res['size']} B)")
    print(f"  traceEvents     {res['n_events']}")
    print(
        "  按 cat          "
        + ", ".join(f"{k}={v}" for k, v in res["by_cat"].most_common(8))
    )
    print(f"  GPU kernel      {res['n_kernels']} 个事件 / {res['distinct_kernels']} 种")
    print(
        f"  GPU busy        {res['gpu_us']/1000:.1f} ms / 窗口 {res['span_us']/1000:.1f} ms"
        + (
            f"  ({100*res['gpu_us']/res['span_us']:.1f}%)"
            if res["span_us"]
            else ""
        )
    )
    print(
        "  kernel 类别     "
        + ", ".join(f"{k}={v}" for k, v in res["families"].items() if v)
    )
    print(f"  top kernel（按 GPU 时间）:")
    for n, d in res["kdur"].most_common(args.top):
        print(f"    {d/1000:9.1f} ms  x{res['kcnt'][n]:<6d} {n[:96]}")
    if res["iters"]:
        ctx = [i["nctx"] for i in res["iters"]]
        tok = [i["ctxtok"] for i in res["iters"]]
        dur = [i["dur_ms"] for i in res["iters"]]
        print(f"  引擎迭代        {len(res['iters'])} 个（来自 execute_* 标注）")
        print(
            f"    context 请求数/迭代  min={min(ctx)} max={max(ctx)} "
            f"均值={sum(ctx)/len(ctx):.1f}"
        )
        print(
            f"    prefill token/迭代   min={min(tok)} max={max(tok)} "
            f"均值={sum(tok)/len(tok):.0f}"
        )
        print(
            f"    迭代耗时 ms          min={min(dur):.1f} max={max(dur):.1f} "
            f"均值={sum(dur)/len(dur):.1f}"
        )
    live = res.get("live_iters") or []
    if live:
        tok = sorted(i["ctxtok"] for i in live)
        print(f"    其中真带 prefill 的  {len(live)} 个，"
              f"token/迭代 中位数={tok[len(tok)//2]} 合计={sum(tok)}")
    for w in res.get("warn", []):
        print(f"  ~  {w}")
    for f in res["fail"]:
        print(f"  !! {f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="trace 文件或目录")
    p.add_argument("--min-bytes", type=int, default=64 * 1024)
    p.add_argument("--max-bytes", type=int, default=200 * 1024 * 1024)
    p.add_argument("--min-kernels", type=int, default=500)
    p.add_argument("--min-distinct", type=int, default=5)
    p.add_argument("--min-prefill-iters", type=int, default=5,
                   help="窗口里至少要有几个**真带 prefill**的引擎迭代")
    p.add_argument("--warn-idle-frac", type=float, default=0.4,
                   help="空迭代占比超过它就提示调大 max_iterations（只警告不判失败）")
    p.add_argument("--expect-prefill", action="store_true")
    p.add_argument("--expect-ranks", type=int, default=0)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [
                    os.path.join(root, n)
                    for n in names
                    if n.endswith((".pt.trace.json", ".pt.trace.json.gz"))
                ]
        else:
            files.append(path)
    files.sort()

    if not files:
        print("没找到任何 trace 文件", file=sys.stderr)
        return 2

    bad = 0
    for f in files:
        try:
            res = analyse(f, args)
        except Exception as exc:  # 解不开也是一种失败，不要静默
            print(f"\n=== {f} ===\n  !! 解析失败: {exc!r}")
            bad += 1
            continue
        report(res, args)
        bad += bool(res["fail"])

    print(f"\n=== 汇总 ===\n  {len(files)} 个 trace，{bad} 个不合格")
    if args.expect_ranks and len(files) != args.expect_ranks:
        print(f"  !! 期望 {args.expect_ranks} 个 rank 的 trace，实际 {len(files)} 个")
        bad += 1
    if bad:
        print("  判定：不可用")
        return 1
    print("  判定：可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
