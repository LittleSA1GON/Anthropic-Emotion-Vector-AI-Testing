"""Structured activation storage, provenance, and external evidence export."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_capture(path, ids, layers, positive, neutral, *, metadata):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(positive, dtype=np.float32)
    b = np.asarray(neutral, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != len(ids) or a.shape[1] != len(layers):
        raise ValueError("Capture arrays must have shape [len(ids), len(layers), hidden]")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid capture values")
    np.savez_compressed(p, positive=a, neutral=b, ids=np.array(ids), layers=np.array(layers, dtype=np.int32))
    write_json(p.with_suffix(".json"), metadata | {"schema": "mm-emotion.capture.v1", "sha256": sha256(p), "shape": list(a.shape)})


def load_capture(path):
    p = Path(path)
    with np.load(p, allow_pickle=False) as d:
        a, b = d["positive"].copy(), d["neutral"].copy()
        layers, ids = d["layers"].astype(int).tolist(), d["ids"].tolist()
    metadata = json.loads(p.with_suffix(".json").read_text())
    if sha256(p) != metadata["sha256"]:
        raise ValueError("Capture file checksum mismatch")
    return ids, layers, a, b, metadata


def runtime_metadata():
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
            "platform": platform.platform(), "git_commit": os.environ.get("MODEL_MOUNTAIN_GIT_COMMIT", "unknown")}
    try:
        import torch
        info.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                     "gpu_cc": [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]})
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except ImportError:
        pass
    return info
