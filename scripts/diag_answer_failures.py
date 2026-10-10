"""Diagnose DR/FPR failures in a completed codex run.

Reproduces the per-case DR/FPR decision from ``AnswerMetric`` exactly, then
buckets each failure so we can see *why* DR (0.658) and FPR (0.148) are low,
rather than guessing.

The key question the buckets answer:
  * did the agent actually call sandbox tools for this case?
  * if it called them, did the tool return usable data (status 0 + real payload)?
  * if it had data, did it still emit a failure phrase / empty answer?

Usage:
    .venv/bin/python scripts/diag_answer_failures.py <run_dir> <dataset_csv>
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_bench.evaluation.metrics.answer import (  # noqa: E402
    AnswerMetric,
    FAILURE_PATTERN,
    extract_content_from_answer,
    extract_route_index_from_answer,
)

FAIL_HEAD = re.compile(
    r"(无法提供|未能成功|失败|未找到|未缓存|不支持|无可用|无法计算|无法直接计算|"
    r"查询结果异常|工具限制|技术限制|需进一步计算|需要进一步计算|没有可用|"
    r"无法获取|暂未获取|未获得|无法查询到|暂不支持|系统当前仅支持|由于)",
)


def tool_payloads(item: dict) -> list[dict]:
    """Flatten every tool result payload for a case into dicts."""
    out = []
    for w in item.get("worker_output") or []:
        for _tid, res in (w.get("tool_results") or {}).items():
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except json.JSONDecodeError:
                    continue
            # MCP wraps the payload: {"content":[{"type":"text","text":"{...}"}]}
            texts = []
            if isinstance(res, dict) and isinstance(res.get("content"), list):
                for c in res["content"]:
                    if isinstance(c, dict) and c.get("text"):
                        texts.append(c["text"])
            elif isinstance(res, dict) and "text" in res:
                texts.append(res["text"])
            if not texts and isinstance(res, str):
                texts.append(res)
            if isinstance(res, dict) and "status" in res:
                out.append(res)
                continue
            for t in texts:
                try:
                    out.append(json.loads(t))
                except (json.JSONDecodeError, TypeError):
                    continue
    return out


def summarize_payload(p: dict) -> str:
    """Human label for a tool payload's data availability."""
    if not isinstance(p, dict):
        return "not-dict"
    if p.get("status") not in (0, "0", None):
        return f"status={p.get('status')} err={(p.get('error') or '')[:40]}"
    data = p.get("data")
    if data in (None, {}, [], ""):
        return "empty-data"
    if isinstance(data, dict):
        if data.get("count") in ("0", 0):
            return "count=0"
        if data.get("status") not in (None, "1") and "route" in data:
            return f"route-status={data.get('status')}"
        for k in ("route", "pois", "location", "name"):
            if data.get(k):
                return f"has-{k}"
        return "dict-keys:" + ",".join(list(data)[:4])
    return type(data).__name__


def main(run_dir: str, dataset_csv: str) -> None:
    run_dir = Path(run_dir)
    rows = list(csv.DictReader(open(dataset_csv, newline="", encoding="utf-8")))
    meta = {f"case_{i}": r for i, r in enumerate(rows, start=1)}

    raw = json.loads(
        (run_dir / "qwen3-4b" / "raw_results.json").read_text(encoding="utf-8")
    )

    metric = AnswerMetric()

    overall = Counter()
    by_family: dict[str, Counter] = defaultdict(Counter)
    fail_samples: dict[str, list[dict]] = defaultdict(list)
    tool_name_counts = Counter()

    for item in raw:
        cid = item["case_id"]
        m = meta.get(cid, {})
        scenario = m.get("task_scenario", "")
        family = m.get("intent_family", "")

        # --- reproduce AnswerMetric.compute ---
        answer = item.get("reporter_output") or ""
        content = extract_content_from_answer(answer)
        if len(content) == 0:
            content = answer
        eval_type = metric.TASK_SCENARIO_TO_EVAL_TYPE.get(scenario, "general")
        delivery = metric._check_delivery_success(content, answer, eval_type)

        pred = {"answer": answer}
        gt = {"task_scenario": scenario, "intent_family": family}
        res = metric.compute(cid, pred, gt)
        dr = res.details["DR"]
        fpr = res.details["FPR"]

        # --- tool availability ---
        calls = []
        for w in item.get("worker_output") or []:
            for tc in w.get("tool_calls") or []:
                calls.append(tc.get("name"))
        payloads = tool_payloads(item)
        summaries = [summarize_payload(p) for p in payloads]
        usable = any(s.startswith(("has-", "dict-keys")) for s in summaries)
        for c in calls:
            tool_name_counts[c] += 1

        answer_empty = not content.strip() or content.lower() == "nan"
        fail_hit = bool(FAIL_HEAD.search(content))

        # --- bucket ---
        if dr == 0:
            if answer_empty:
                bucket = "A_DR0_empty_answer"
            elif fail_hit:
                bucket = "B_DR0_failure_phrase"
            else:
                bucket = "C_DR0_no_route_keywords"
        else:
            if fpr == 1.0:
                bucket = "OK"
            else:
                bucket = "D_DR1_FPR0_wrong_value"
        overall[bucket] += 1
        by_family[family][bucket] += 1

        if bucket in ("B_DR0_failure_phrase", "D_DR1_FPR0_wrong_value") and len(fail_samples[bucket]) < 6:
            fail_samples[bucket].append({
                "case": cid,
                "scenario": scenario,
                "family": family,
                "n_calls": len(calls),
                "tools": calls[:6],
                "payloads": summaries[:4],
                "usable_data": usable,
                "meta": {k: m.get(k) for k in ("matched_transit_index", "ans_loc") if m.get(k)},
                "content_head": content[:180].replace("\n", " "),
            })

    n = len(raw)
    print(f"run={run_dir.name}  cases={n}")
    print("\n== overall buckets ==")
    for k, v in overall.most_common():
        print(f"  {k:34s} {v:4d}  {v/n:6.1%}")

    print("\n== per intent_family ==")
    for fam in sorted(by_family):
        tot = sum(by_family[fam].values())
        line = "  ".join(
            f"{k.split('_')[0]}={by_family[fam][k]}" for k in
            ("OK", "A_DR0_empty_answer", "B_DR0_failure_phrase", "C_DR0_no_route_keywords", "D_DR1_FPR0_wrong_value")
        )
        print(f"  {fam[:38]:38s} n={tot:3d}  {line}")

    print("\n== tool call frequency ==")
    for t, c in tool_name_counts.most_common():
        print(f"  {t:24s} {c}")

    for bucket in ("B_DR0_failure_phrase", "D_DR1_FPR0_wrong_value"):
        print(f"\n== samples: {bucket} ==")
        for s in fail_samples[bucket]:
            print(f"  [{s['case']}] {s['scenario']} ({s['family'][:22]}) "
                  f"calls={s['n_calls']} usable={s['usable_data']} meta={s['meta']}")
            print(f"      tools={s['tools']}")
            print(f"      payloads={s['payloads']}")
            print(f"      content={s['content_head']!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
