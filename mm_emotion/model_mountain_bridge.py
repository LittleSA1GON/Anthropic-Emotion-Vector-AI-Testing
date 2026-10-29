"""Read-only research evidence bridge. NEVER modifies Model Mountain admission policy."""
import json
from pathlib import Path
from .artifacts import sha256, write_json


def export_evidence(run_dir, output_path):
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if manifest["status"] != "complete" or manifest["origin"] != "real_model_inference":
        raise ValueError("Only complete genuine model runs can be exported")
    record = {"schema": "model-mountain.research.evidence.v1", "research_area": "emotion_representation",
              "security_policy_effect": "none", "not_a_security_verdict": True,
              "model_id": manifest["model_id"], "precision": manifest["precision"],
              "backend": manifest["backend"], "dataset_sha256": manifest["dataset_sha256"],
              "run_manifest_sha256": sha256(root / "manifest.json"),
              "metrics": summary["metrics"], "source": str(root),
              "limits": ["Author-created synthetic prompts", "Not evidence of human or machine emotion",
                         "Does not calibrate MIA security gates", "Does not authorize agent actions"]}
    write_json(output_path, record)
    return record
