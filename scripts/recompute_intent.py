"""Recompute planner_output.intent in an existing run's raw_results.json.

Used when the intent-classification rules in ``codex.py`` change but the (long)
codex run does not need to be re-executed: eval only reads raw_results.json, so
rewriting the intent field in place and re-running ``mbench eval`` refreshes the
instruction metric without paying for the run again.

Usage:
    .venv/bin/python scripts/recompute_intent.py <run_dir> <dataset_csv>
"""

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobility_bench.agent.frameworks.codex import CodexFramework  # noqa: E402


def main(run_dir: str, dataset_csv: str) -> None:
    run_dir = Path(run_dir)
    rows = list(csv.DictReader(open(dataset_csv, newline="", encoding="utf-8")))
    query_by_case = {f"case_{i}": r["query"] for i, r in enumerate(rows, start=1)}

    for model_dir in [p for p in run_dir.iterdir() if p.is_dir()]:
        raw_file = model_dir / "raw_results.json"
        if not raw_file.exists():
            continue
        raw = json.loads(raw_file.read_text(encoding="utf-8"))
        backup = raw_file.with_suffix(".json.bak")
        if not backup.exists():
            shutil.copy2(raw_file, backup)

        changed = 0
        for item in raw:
            query = query_by_case.get(item.get("case_id", ""), "")
            if not query:
                continue
            planner = item.get("planner_output") or {}
            new_intent = CodexFramework._infer_intent(query, planner.get("thinking", ""))
            if planner.get("intent") != new_intent:
                changed += 1
                planner["intent"] = new_intent
                item["planner_output"] = planner

        raw_file.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{raw_file}: updated {changed}/{len(raw)} intents")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
