"""Versioned matched-pair emotion prompts and anti-leakage validation."""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

EMOTIONS = ("joy", "calm", "anger", "anxiety")
SPLITS = ("train", "validation", "test")


def load_pairs(path: str | Path) -> list[dict]:
    data = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_pairs(data)
    return data


def validate_pairs(data: list[dict]) -> dict:
    if not data:
        raise ValueError("Dataset is empty")
    ids = set()
    split_by_scenario = defaultdict(set)
    counts = Counter()
    prompts_by_split = defaultdict(set)
    for r in data:
        for key in ("id", "scenario_id", "split", "emotion", "neutral_prompt", "emotion_prompt", "source"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f"Missing/invalid {key}: {r}")
        if r["id"] in ids:
            raise ValueError(f"Duplicate ID {r['id']}")
        ids.add(r["id"])
        if r["split"] not in SPLITS or r["emotion"] not in EMOTIONS:
            raise ValueError(f"Unknown split/emotion: {r['id']}")
        if r["neutral_prompt"] == r["emotion_prompt"]:
            raise ValueError(f"Pair prompts are identical: {r['id']}")
        split_by_scenario[r["scenario_id"]].add(r["split"])
        prompts_by_split[r["split"]].add(r["neutral_prompt"])
        prompts_by_split[r["split"]].add(r["emotion_prompt"])
        counts[(r["split"], r["emotion"])] += 1
    leakage = [s for s, x in split_by_scenario.items() if len(x) > 1]
    if leakage:
        raise ValueError(f"Scenario leakage across splits: {leakage}")
    for a in SPLITS:
        for b in SPLITS:
            if a < b and prompts_by_split[a] & prompts_by_split[b]:
                raise ValueError(f"Prompt leakage across {a}/{b}")
    for s in SPLITS:
        for e in EMOTIONS:
            if counts[(s, e)] < 2:
                raise ValueError(f"At least 2 pairs required for {s}/{e}")
    return {f"{s}/{e}": counts[(s, e)] for s in SPLITS for e in EMOTIONS}


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_pairs(data: list[dict], split: str, emotion: str, limit: int = 0) -> list[dict]:
    records = [r for r in data if r["split"] == split and r["emotion"] == emotion]
    if limit:
        records = records[:limit]
    if len(records) < 2:
        raise ValueError(f"Need 2+ {split}/{emotion} pairs; found {len(records)}")
    return records
