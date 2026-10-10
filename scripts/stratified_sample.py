"""Stratified random sample of the 100k MobilityBench pool.

Reads the two 50k release files, merges them, normalises the two different
spellings the release uses for the preference-constrained family, draws 500 rows
proportional to the four official intent families, and writes the result using
exactly sample_10.csv's column list and order (the full release carries 46
columns; the sample is a 33-column projection of it).

Usage:
    python scripts/stratified_sample.py <out_csv> [n]
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "datasets"
SOURCES = [
    DATA / "all_data_benchmark_50000.csv",
    DATA / "all_data_benchmark_50000_2.csv",
]
SCHEMA = DATA / "sample_10.csv"

# The release labels the same family two ways; fold them together so the
# stratification matches the README's four-family split.
FAMILY_ALIASES = {"Preference-constrained Planning": "Preference-Constrained Route Planning"}


def largest_remainder(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {k: v * n / total for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    short = n - sum(alloc.values())
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:short]:
        alloc[k] += 1
    return alloc


def main(out_path: str, n: int = 500) -> None:
    columns = next(csv.reader(open(SCHEMA, newline="", encoding="utf-8-sig")))

    rows: list[dict] = []
    main_cols: list[str] = []
    for src in SOURCES:
        with open(src, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not main_cols:
                main_cols = list(reader.fieldnames or [])
            for r in reader:
                r["intent_family"] = FAMILY_ALIASES.get(r["intent_family"], r["intent_family"])
                rows.append(r)

    counts = Counter(r["intent_family"] for r in rows)
    alloc = largest_remainder(dict(counts), n)

    rng = random.Random(20261006)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["intent_family"]].append(r)

    picked: list[dict] = []
    for family, k in sorted(alloc.items()):
        picked.extend(rng.sample(by_family[family], k))
    rng.shuffle(picked)

    out = Path(out_path)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(picked)

    print(f"pool: {len(rows)} rows from {len(SOURCES)} files")
    print(f"wrote {out} rows={len(picked)} cols={len(columns)}")
    print("stratum allocation (target vs drawn):")
    drawn = Counter(r["intent_family"] for r in picked)
    for family in counts:
        print(f"  {family}: {counts[family]} pool -> {alloc[family]} target -> {drawn[family]} drawn")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 500)
