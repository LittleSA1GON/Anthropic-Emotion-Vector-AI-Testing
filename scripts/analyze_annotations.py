"""Analyze completed 1-5 human ratings after unlocking a blinded intervention study."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True, help="Completed blinded_annotations.csv")
    p.add_argument("--key", required=True, help="blinding_key_PRIVATE.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    key = json.loads(Path(args.key).read_text())
    rows = list(csv.DictReader(Path(args.ratings).open(newline="", encoding="utf-8")))
    by = defaultdict(lambda: {"emotion_alignment_1to5": [], "task_quality_1to5": []})
    for row in rows:
        if row["trial_id"] not in key:
            raise ValueError("Unknown trial ID in rating file")
        k = key[row["trial_id"]]
        coefficient = k["coefficient"] if isinstance(k, dict) else k
        group = (row["emotion"], float(coefficient))
        for metric in ("emotion_alignment_1to5", "task_quality_1to5"):
            v = float(row[metric])
            if not 1 <= v <= 5:
                raise ValueError(f"Out of range rating: {metric}")
            by[group][metric].append(v)
    result = [{"emotion": emotion, "coefficient": coefficient, "n": len(vals["task_quality_1to5"]),
               "mean_emotion_alignment_1to5": sum(vals["emotion_alignment_1to5"]) / len(vals["emotion_alignment_1to5"]),
               "mean_task_quality_1to5": sum(vals["task_quality_1to5"]) / len(vals["task_quality_1to5"])}
              for (emotion, coefficient), vals in sorted(by.items())]
    Path(args.out).write_text(json.dumps({"rating_summary": result, "method": "descriptive means only; no reliability or significance inferred"}, indent=2))
    print(f"Analyzed {len(rows)} complete human annotations to {args.out}")


if __name__ == "__main__":
    main()
