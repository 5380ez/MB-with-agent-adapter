"""Merge evaluation_summary.csv of multiple frameworks into one comparison table.

Produces:
  - a wide table (rows = intent_family / metric / sub_dimension,
    columns = framework)
  - a long table (one row per framework/row combo)

Usage:
    python scripts/merge_eval_summaries.py \
        --run run_20261006_132102_0ded0a6f:codex \
        --run run_20261009_095601_0047c053:cc \
        --run run_20261009_101951_f2a6234f:dsh \
        --run run_20261009_103259_e0dd679d:dsh_mini \
        --run run_20261009_104445_f721a769:mini_swe \
        --output data/results/merged_eval_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRIC_ORDER = ["tool", "instruction", "planning", "answer", "efficiency"]

# Sub-dimension order within each metric (for a readable row order).
SUB_DIM_ORDER = {
    "tool": ["TS_P", "TS_R", "SC"],
    "instruction": ["ID", "IE"],
    "planning": ["DEC_P", "DEC_R"],
    "answer": ["DR", "FPR"],
    "efficiency": ["IT", "OT"],
}


def parse_runs(run_specs: list[str]) -> list[tuple[str, str]]:
    """Parse 'run_id:label' specs into (run_id, label) pairs."""
    runs = []
    for spec in run_specs:
        run_id, _, label = spec.partition(":")
        runs.append((run_id.strip(), label.strip() or run_id.strip()))
    return runs


def load_summary(results_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    summary_file = results_dir / run_id / "evaluation" / "evaluation_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary for {label} ({run_id}): {summary_file}")

    df = pd.read_csv(summary_file)
    df["framework"] = label
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", dest="runs", action="append", required=True,
                        help="'run_id:label' pair, repeatable")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output", default="data/results/merged_eval_summary")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    results_dir = Path(args.results_dir)

    frames = [load_summary(results_dir, run_id, label) for run_id, label in runs]
    long_df = pd.concat(frames, ignore_index=True)

    # Empty intent_family means the overall aggregate — label it explicitly so
    # pandas does not drop those rows during the pivot (NaN groups are dropped).
    overall_label = "(Overall)"
    long_df["intent_family"] = long_df["intent_family"].fillna(overall_label)

    def _keys(df: pd.DataFrame) -> pd.DataFrame:
        df["_metric_key"] = df["metric"].map(
            {m: i for i, m in enumerate(METRIC_ORDER)}
        ).fillna(len(METRIC_ORDER)).astype(int)
        df["_subdim_key"] = df.apply(
            lambda r: SUB_DIM_ORDER.get(r["metric"], []).index(r["sub_dimension"])
            if r["sub_dimension"] in SUB_DIM_ORDER.get(r["metric"], [])
            else 999,
            axis=1,
        )
        # Overall first, then the rest.
        df["_overall_first"] = (df["intent_family"] != overall_label).astype(int)
        return df

    long_df = _keys(long_df)

    wide_df = long_df.pivot_table(
        index=["intent_family", "metric", "sub_dimension"],
        columns="framework",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Put overall first, then metric / sub-dimension order.
    wide_df = _keys(wide_df).sort_values(
        by=["_overall_first", "_metric_key", "_subdim_key", "intent_family"],
    ).drop(columns=["_overall_first", "_metric_key", "_subdim_key"])

    column_order = [label for _, label in runs]
    wide_df = wide_df[
        ["intent_family", "metric", "sub_dimension"] + column_order
    ]

    # Drop sorting helper columns from the long table too.
    long_df = long_df.drop(columns=["_overall_first", "_metric_key", "_subdim_key"])
    long_df = long_df[["intent_family", "metric", "sub_dimension", "model", "value", "framework"]]

    output_base = Path(args.output)
    wide_csv = output_base.with_suffix(".csv")
    wide_xlsx = output_base.with_suffix(".xlsx")
    long_csv = output_base.with_name(output_base.name + "_long.csv")

    wide_df.to_csv(wide_csv, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(wide_xlsx, engine="openpyxl") as writer:
            wide_df.to_excel(writer, sheet_name="comparison", index=False)
            long_df.to_excel(writer, sheet_name="long", index=False)
    except Exception as exc:  # openpyxl may be missing; CSV still written.
        print(f"[warn] Excel output skipped: {exc}")

    print(f"Wide table -> {wide_csv}")
    print(f"Long table -> {long_csv}")
    if wide_xlsx.exists():
        print(f"Excel      -> {wide_xlsx}")

    print("\n--- Wide table preview ---")
    print(wide_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())