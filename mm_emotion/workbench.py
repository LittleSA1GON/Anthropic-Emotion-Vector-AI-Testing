"""Safe, auditable primitives for the interactive emotion-vector workbench.

Does not load models on import. Interactive quick-fits are exploratory and are
never represented as held-out causal evidence.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

from .vectors import normalized_difference, pairwise_pca_direction, layer_metrics

_ALLOWED = ("jsonl", "json", "csv")
_REQUIRED = ("neutral_prompt", "emotion_prompt")


def safe_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("._-")[:80]
    if not name or name in {".", ".."}:
        raise ValueError("Name must contain a letter or number")
    return name


def parse_pairs(content: str | bytes, fmt: str, *, max_rows: int = 2000) -> list[dict]:
    """Parse uploaded matched-pair data, requiring nonempty contrastive prompts.

    CSV columns may be just neutral_prompt,emotion_prompt (other fields optional).
    No split leakage claims are made for partially specified user datasets.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    if fmt not in _ALLOWED:
        raise ValueError(f"format must be one of {_ALLOWED}")
    if fmt == "csv":
        rows = list(csv.DictReader(io.StringIO(content)))
    elif fmt == "jsonl":
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        rows = json.loads(content)
        if isinstance(rows, dict) and "pairs" in rows:
            rows = rows["pairs"]
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ValueError(f"Expected a list of at most {max_rows} records")
    cleaned = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx+1} is not an object")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in _REQUIRED):
            raise ValueError(f"Row {idx+1} must have nonempty neutral_prompt and emotion_prompt")
        if row["neutral_prompt"].strip() == row["emotion_prompt"].strip():
            raise ValueError(f"Row {idx+1} has identical prompts")
        clean = dict(row)
        for k in _REQUIRED:
            clean[k] = clean[k].strip()
        cleaned.append(clean)
    return cleaned


def pair_to_jsonl(rows: list[dict]) -> bytes:
    parse_pairs(json.dumps(rows), "json")
    return ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n").encode("utf-8")


def make_quick_vector(backend: Any, pairs: list[dict], layers: list[int],
                      method: str = "mean_diff", *, label: str = "custom") -> tuple[np.ndarray, dict]:
    """Fit exploratory layer directions on >=2 pairs, recording *training-only* scores."""
    if len(pairs) < 2:
        raise ValueError("At least two independent contrast pairs required")
    parse_pairs(json.dumps(pairs), "json")
    if not layers or len(set(layers)) != len(layers) or any(x < 0 or x >= len(backend.layers) for x in layers):
        raise ValueError("Invalid or repeated 0-based layer indices")
    positive, neutral = [], []
    for pair in pairs:
        positive.append(backend.capture(pair["emotion_prompt"], layers))
        neutral.append(backend.capture(pair["neutral_prompt"], layers))
    pos, neu = np.stack(positive), np.stack(neutral)
    if method == "mean_diff":
        directions = normalized_difference(pos, neu)
    elif method == "pairwise_pca":
        directions = pairwise_pca_direction(pos, neu)
    else:
        raise ValueError("method must be mean_diff or pairwise_pca")
    scores = layer_metrics(pos, neu, directions)
    for i, score in enumerate(scores):
        score["layer_0based"] = layers[i]
    return directions, {
        "schema": "mm-emotion.quick-vector.v1",
        "origin": "real_model_inference",
        "evaluation": "training_only_not_held_out",
        "label": label,
        "model_id": backend.model_id,
        "resolved_revision": getattr(getattr(getattr(backend,'model',None),'config',None),'_commit_hash',None),
        "precision": backend.precision,
        "backend": "hf",
        "capture_position": "last_input_token_decoder_block_output",
        "method": method,
        "n_pairs": len(pairs),
        "pairs_sha256": hashlib.sha256(pair_to_jsonl(pairs)).hexdigest(),
        "layers_0based": layers,
        "layer_scores": scores,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_quick_vector(directory: str | Path, name: str, directions: np.ndarray, meta: dict) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (safe_name(name) + ".npz")
    if target.exists():
        raise FileExistsError(f"Vector already exists; choose a new name: {target}")
    layers = meta["layers_0based"]
    if directions.ndim != 2 or len(layers) != len(directions) or not np.isfinite(directions).all():
        raise ValueError("Invalid direction shape or values")
    # Avoid creating a partial final vector if metadata fails to serialize.
    metadata_text = json.dumps(meta, sort_keys=True, indent=2, allow_nan=False)
    np.savez_compressed(target, directions=directions.astype(np.float32), layers=np.asarray(layers, dtype=np.int32),
                        selected=np.asarray(layers[0], dtype=np.int32))
    meta = {**meta, "vector_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    metadata_text = json.dumps(meta, sort_keys=True, indent=2, allow_nan=False)
    target.with_suffix(".json").write_text(metadata_text, encoding="utf-8")
    return target


def load_vector(path: str | Path, *, model_id: str | None = None,
                precision: str | None = None, num_layers: int | None = None,
                hidden_size: int | None = None) -> tuple[np.ndarray, list[int], dict]:
    path = Path(path)
    if not path.is_file() or path.suffix != ".npz":
        raise ValueError("Choose an existing .npz vector")
    meta_file = path.with_suffix(".json")
    if not meta_file.exists():
        raise ValueError("Missing matching .json vector provenance; refuse to steer")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as npz:
        dirs = npz["directions"].astype(np.float32)
        layers = npz["layers"].astype(int).tolist()
    if dirs.ndim != 2 or len(layers) != len(dirs) or len(set(layers)) != len(layers):
        raise ValueError("Invalid vector shape/layers")
    if not np.isfinite(dirs).all() or np.linalg.norm(dirs, axis=1).min() < 1e-8:
        raise ValueError("Invalid vector values")
    if meta.get("origin") != "real_model_inference":
        raise ValueError("Only genuine-model vectors are accepted for inference")
    if meta.get("layers_0based") is not None and meta["layers_0based"] != layers:
        raise ValueError("Vector metadata layer IDs differ from saved arrays")
    if meta.get("vector_sha256") is not None:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != meta["vector_sha256"]:
            raise ValueError("Vector checksum mismatch: saved evidence was altered")
    if model_id is not None and meta.get("model_id") != model_id:
        raise ValueError("Vector model ID differs from loaded model; explicit alignment experiment needed")
    if precision is not None and meta.get("precision") != precision:
        raise ValueError("Vector precision differs from loaded model; use an explicitly documented transfer run")
    if num_layers is not None and (min(layers) < 0 or max(layers) >= num_layers):
        raise ValueError("Vector references unavailable decoder layers")
    if hidden_size is not None and dirs.shape[1] != hidden_size:
        raise ValueError("Vector and model hidden dimensions differ")
    return dirs, layers, meta


def sweep_grid(coefficients: str, *, max_points: int = 9) -> list[float]:
    try:
        values = [float(t.strip()) for t in coefficients.split(",") if t.strip()]
    except ValueError as exc:
        raise ValueError("Use comma-separated numeric coefficients, e.g. -6,-3,0,3,6") from exc
    if not values or len(values) > max_points or any(not np.isfinite(v) or abs(v) > 30 for v in values):
        raise ValueError(f"Provide 1-{max_points} finite coefficients in [-30,30]")
    # Mandatory 0 control; duplicates removed in the original order.
    return list(dict.fromkeys([0.0] + values))


def validate_test_prompts(text: str, *, max_prompts: int = 20) -> list[str]:
    rows = [x.strip() for x in text.splitlines() if x.strip()]
    if not rows or len(rows) > max_prompts:
        raise ValueError(f"Provide 1-{max_prompts} nonempty prompts, one per line")
    if any(len(x) > 12000 for x in rows):
        raise ValueError("Prompt over 12,000 characters")
    return rows


def serialize_results(path: str | Path, rows: list[dict], settings: dict) -> Path:
    """Never overwrite prior evidence. Results are genuine only if caller actually ran inference."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"Experiment record already exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {"schema": "mm-emotion.steering-session.v1", "origin": "real_model_generation",
           "created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings, "results": rows,
           "interpretation": "Research generations, not a validated emotion or causal-effect metric"}
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return p
