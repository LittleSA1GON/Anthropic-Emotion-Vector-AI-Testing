"""Build a markdown research report from a genuine complete run.

Usage: python scripts/render_report.py --run runs/fp16_full
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    root = Path(args.run)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if manifest["status"] != "complete" or manifest["origin"] != "real_model_inference":
        raise SystemExit("Report refuses incomplete or synthetic results")
    rows = summary["metrics"]
    lines = ["# Emotion Vector Research — Run Report", "", "**Source:** real model inference on a small author-created synthetic dataset. NOT evidence of machine emotion.", "",
             f"- Model: `{manifest['model_id']}`", f"- Weight precision: `{manifest['precision']}`", f"- Backend: `{manifest['backend']}`",
             f"- Dataset SHA256: `{manifest['dataset_sha256']}`", f"- Vector method: `{manifest.get('vector_method', 'mean_diff')}`", "",
             "| Emotion | Layer (0-based) | Held-out pairs | Positive fraction | Mean margin |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['emotion']} | {r['selected_layer_0based']} | {r['test_n']} | {r['test_positive_fraction']:.3f} | {r['test_mean_margin']:.4f} |")
    lines += ["", "Layer selected using validation, not test. The held-out set has very few scenarios; no statistical generalization or emotion inference is established.",
              "No randomized control, blinded rating or multiple-seed uncertainty estimate is included automatically.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print(root / "REPORT.md")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([x["emotion"] for x in rows], [x["test_positive_fraction"] for x in rows])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Held-out positive fraction (synthetic pairs)")
        ax.set_title(f"{manifest['model_id']} | {manifest['precision']} | {manifest['backend']}")
        fig.tight_layout()
        fig.savefig(root / "test_positive_fraction.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
