"""Generate an annotator-facing blinded CSV and a PRIVATE condition key.

Usage: python scripts/blind_hf_generations.py --run runs/fp16_full --out runs/fp16_blind
Do not distribute blinding_key_PRIVATE.json until all annotator ratings are frozen.
"""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    src, dst = Path(args.run), Path(args.out)
    m = json.loads((src / "manifest.json").read_text())
    if m["status"] != "complete" or m["origin"] != "real_model_inference":
        raise SystemExit("Only a completed real model run can be blinded")
    rows = []
    for f in sorted((src / "generations").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit("No real generations found; run with --generate-samples")
    dst.mkdir(parents=True, exist_ok=True)
    shuffled = list(enumerate(rows))
    random.Random(args.seed).shuffle(shuffled)
    key = {}
    with (dst / "blinded_annotations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "pair_id", "emotion", "response", "emotion_alignment_1to5", "task_quality_1to5", "notes"])
        w.writeheader()
        for tid, (idx, row) in enumerate(shuffled):
            w.writerow({"trial_id": tid, "pair_id": row["pair_id"], "emotion": row["emotion"],
                        "response": row["text"], "emotion_alignment_1to5": "", "task_quality_1to5": "", "notes": ""})
            key[str(tid)] = {"coefficient": row["coefficient"], "original_index": idx, "pair_id": row["pair_id"]}
    (dst / "blinding_key_PRIVATE.json").write_text(json.dumps(key, indent=2))
    print(f"Prepared {len(shuffled)} blinded output ratings; keep private key private")


if __name__ == "__main__":
    main()
