"""Dataset -> actual activations -> train vectors -> validation-only layer selection -> holdout evaluation."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import numpy as np
from .artifacts import load_capture, runtime_metadata, save_capture, write_json, write_jsonl, sha256
from .dataset import EMOTIONS, SPLITS, load_pairs, dataset_sha256, select_pairs
from .vectors import normalized_difference, pairwise_pca_direction, layer_metrics, select_layer, cosine


def build_backend(kind, *, model, precision, max_length, layers, mia_config=None, aperture_mib=512):
    if kind == "hf":
        from .hf_backend import HFBackend
        return HFBackend(model, precision=precision, max_length=max_length)
    if kind == "mia":
        from .mia_backend import MIABackend
        if mia_config is None:
            raise ValueError("MIA requires --mia-config")
        cfg = json.loads(Path(mia_config).read_text())
        selected = [i - 1 for i in cfg["hidden_states"]["layers"]]
        if selected != layers:
            raise ValueError(f"MIA config 1-based layer IDs mismatch. Expected {[i+1 for i in layers]}, got {cfg['hidden_states']['layers']}")
        return MIABackend(model, precision, max_length, config_file=str(Path(mia_config).resolve()),
                          aperture_mib=aperture_mib)
    raise ValueError(f"Unknown backend: {kind}")


def run_experiment(*, backend_kind, model, precision, data_path, output, layers,
                   limit_per_split=0, max_length=512, mia_config=None, aperture_mib=512,
                   generate_samples=0, coefficients=(0.0, 3.0), vector_method="mean_diff"):
    dataset = load_pairs(data_path)
    root = Path(output)
    if root.exists() and list(root.iterdir()):
        raise FileExistsError(f"Output path must be empty to avoid overwriting evidence: {root}")
    root.mkdir(parents=True, exist_ok=True)
    coefficients = list(dict.fromkeys([0.] + list(coefficients)))
    if any(not np.isfinite(c) or abs(c)>30 for c in coefficients):
        raise ValueError('Coefficients must be finite and within [-30,30]')
    backend = None
    manifest = {"schema": "mm-emotion.run.v1", "status": "in_progress", "origin": "real_model_inference",
                "backend": backend_kind, "precision": precision, "model_id": model,
                "source_dataset": str(data_path), "dataset_sha256": dataset_sha256(data_path),
                "layers_0based": layers, "max_length": max_length, "limit_per_split": limit_per_split,
                "generation_count": generate_samples, "vector_method": vector_method, "runtime": runtime_metadata(), "stage": "capture"}
    write_json(root / "manifest.json", manifest)
    summary = []
    try:
        backend = build_backend(backend_kind, model=model, precision=precision, max_length=max_length,
                                layers=layers, mia_config=mia_config, aperture_mib=aperture_mib)
        manifest['resolved_revision'] = getattr(getattr(getattr(backend,'model',None),'config',None),'_commit_hash',None)
        for emotion in EMOTIONS:
            capfiles = {}
            for split in SPLITS:
                pairs = select_pairs(dataset, split, emotion, limit_per_split)
                pos = [backend.capture(p["emotion_prompt"], layers) for p in pairs]
                neutral = [backend.capture(p["neutral_prompt"], layers) for p in pairs]
                path = root / "captures" / f"{emotion}_{split}.npz"
                save_capture(path, [p["id"] for p in pairs], layers, pos, neutral,
                             metadata={"emotion": emotion, "split": split, "model_id": model,
                                       "precision": precision, "backend": backend_kind,
                                       "dataset_sha256": manifest["dataset_sha256"],
                                       "origin": "real_model_inference"})
                capfiles[split] = path
            _, train_layers, train_p, train_n, train_meta = load_capture(capfiles["train"])
            _, val_layers, val_p, val_n, val_meta = load_capture(capfiles["validation"])
            _, test_layers, test_p, test_n, test_meta = load_capture(capfiles["test"])
            if train_layers != val_layers or train_layers != test_layers:
                raise RuntimeError("Inconsistent layer ordering")
            directions = (normalized_difference(train_p, train_n) if vector_method == "mean_diff"
                          else pairwise_pca_direction(train_p, train_n) if vector_method == "pairwise_pca"
                          else None)
            if directions is None:
                raise ValueError(f"Unknown vector method: {vector_method}")
            val_scores = layer_metrics(val_p, val_n, directions)
            chosen = select_layer(val_scores, layers)
            test_scores = layer_metrics(test_p, test_n, directions)
            selected_position = layers.index(chosen)
            vector_dir = root / "vectors"
            vector_dir.mkdir(exist_ok=True)
            vfile = vector_dir / f"{emotion}.npz"
            np.savez_compressed(vfile, directions=directions, layers=np.asarray(layers, dtype=np.int32),
                                selected=np.asarray(chosen, dtype=np.int32))
            # MIA uses one selected 0-based block and expects a torch-save dict.
            try:
                import torch
                torch.save({"dir": torch.tensor(directions[selected_position].copy(), dtype=torch.float32)},
                           vector_dir / f"{emotion}_mia.pt")
            except ImportError:
                pass
            spec = {"emotion": emotion, "trained_on": "train", "vector_method": vector_method, "layer_selection_on": "validation",
                    "test_evaluation_on": "test", "layer_0based": chosen,
                    "vector_l2_norm": float(np.linalg.norm(directions[selected_position])),
                    "validation_layers": dict(zip(map(str, layers), val_scores)),
                    "test_layers": dict(zip(map(str, layers), test_scores)),
                    "vector_sha256": sha256(vfile), "dataset_sha256": manifest["dataset_sha256"],
                    "model_id": model, "precision": precision, "backend": backend_kind,
                    "origin": "real_model_inference", "note": "Exploratory; synthetic dataset, not validated psychology"}
            write_json(vector_dir / f"{emotion}.json", spec)
            sc = test_scores[selected_position]
            summary.append({"emotion": emotion, "selected_layer_0based": chosen,
                            "test_n": sc["n_pairs"], "test_positive_fraction": sc["positive_fraction"],
                            "test_mean_margin": sc["mean_margin"], "val_positive_fraction": val_scores[selected_position]["positive_fraction"]})
            if generate_samples:
                if backend_kind != "hf":
                    # MIA requires a distinct 'steer' worker; use standalone script after capture.
                    continue
                rows = []
                for row in [r for r in dataset if r["split"] == "test" and r["emotion"] == emotion][:generate_samples]:
                    for coeff in coefficients:
                        result = backend.generate(row["neutral_prompt"], chosen, directions[selected_position], coeff)
                        rows.append({"pair_id": row["id"], "emotion": emotion,
                                     "coefficient": coeff, "layer_0based": chosen, **result,
                                     "model_id":model,"precision":precision,"backend":backend_kind,
                                     "vector_sha256":spec['vector_sha256'],"exception":None,
                                     "prompt": row["neutral_prompt"], "origin": "real_model_generation"})
                write_jsonl(root / "generations" / f"{emotion}.jsonl", rows)
        write_json(root / "summary.json", {"metrics": summary, "note": "Test results are on a small synthetic matched-pair dataset and not externally validated"})
        with (root / "summary.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        manifest["status"], manifest["stage"] = "complete", "finished"
        write_json(root / "manifest.json", manifest)
        return summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        write_json(root / "manifest.json", manifest)
        raise
    finally:
        if hasattr(backend, "close"):
            backend.close()


def compare_runs(a, b, out):
    a, b = Path(a), Path(b)
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    for field in ("model_id", "backend", "dataset_sha256", "layers_0based", "vector_method"):
        if ma[field] != mb[field]:
            raise ValueError(f"Run mismatch for {field}; not a controlled precision comparison")
    if ma["status"] != "complete" or mb["status"] != "complete":
        raise ValueError("Cannot compare incomplete runs")
    results = []
    for emotion in EMOTIONS:
        aa = np.load(a / "vectors" / f"{emotion}.npz", allow_pickle=False)
        bb = np.load(b / "vectors" / f"{emotion}.npz", allow_pickle=False)
        layers = aa["layers"].tolist()
        for i, layer in enumerate(layers):
            results.append({"emotion": emotion, "layer_0based": layer,
                            "direction_cosine": cosine(aa["directions"][i], bb["directions"][i]),
                            "precision_a": ma["precision"], "precision_b": mb["precision"]})
    write_json(out, {"schema": "mm-emotion.comparison.v1", "run_a": str(a), "run_b": str(b),
                     "comparable": True, "cosine_by_emotion_layer": results,
                     "caution": "Cosine similarity alone does not demonstrate causal or behavioral preservation"})
    return results
