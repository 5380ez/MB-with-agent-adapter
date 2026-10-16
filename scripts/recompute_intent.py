"""Recompute planner_output.intent in an existing run's raw_results.json.

Intent is no longer inferred from the tool trajectory: the prompt asks the agent
to classify it itself and put the label in its final JSON answer. This script
therefore re-reads the intent straight out of ``reporter_output``, so a stored
run can be refreshed without re-running the agent. Rewrite the intent field in
place, then re-run ``mbench eval`` to refresh the instruction metric for free.

Usage:
    .venv/bin/python scripts/recompute_intent.py <run_dir>
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobility_bench.agent.frameworks._cli_common import (  # noqa: E402
    extract_intent_from_answer,
)


def main(run_dir: str) -> None:
    run_dir = Path(run_dir)

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
            planner = item.get("planner_output") or {}
            new_intent = extract_intent_from_answer(item.get("reporter_output", ""))
            if planner.get("intent") != new_intent:
                changed += 1
                planner["intent"] = new_intent
                item["planner_output"] = planner

        raw_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{raw_file}: updated {changed}/{len(raw)} intents")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
