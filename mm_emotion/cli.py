"""Command-line entry point; `python -m mm_emotion.cli --help`."""
import argparse
import json
from pathlib import Path
import numpy as np
from .artifacts import runtime_metadata, write_json, write_jsonl
from .dataset import load_pairs, dataset_sha256, EMOTIONS, SPLITS
from .pipeline import run_experiment, compare_runs
from .vectors import normalized_difference, layer_metrics, select_layer


def main():
    p = argparse.ArgumentParser(prog="mm-emotion", description="Model Mountain Emotion Lab")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check environment, GPU capability and runtime packages")
    d = sub.add_parser("validate", help="Validate dataset and leakage prevention")
    d.add_argument("--data", default="data/emotion_pairs_v1.jsonl")
    s = sub.add_parser("smoke", help="CPU-only synthetic pipeline plumbing test; NOT empirical model data")
    s.add_argument("--out", default="runs/synthetic_smoke.json")
    r = sub.add_parser("run", help="Run the actual 3-split, 4-emotion capture and evaluation")
    r.add_argument("--backend", choices=["hf", "mia"], default="hf")
    r.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    r.add_argument("--precision", choices=["fp16", "fp32", "nf4", "awq", "gptq"], default="fp16")
    r.add_argument("--data", default="data/emotion_pairs_v1.jsonl")
    r.add_argument("--out", required=True)
    r.add_argument("--layers", default="5,11,17,23", help="0-based decoder block IDs")
    r.add_argument("--limit-per-split", type=int, default=0, help="0=all; minimum 2")
    r.add_argument("--max-length", type=int, default=512)
    r.add_argument("--mia-config", default="configs/mia_capture_qwen2_1p5b.json")
    r.add_argument("--aperture-mib", type=int, default=512)
    r.add_argument("--generate-samples", type=int, default=0, help="HF only, test subset")
    r.add_argument("--coefficients", default="0,3,6")
    r.add_argument("--vector-method", choices=["mean_diff", "pairwise_pca"], default="mean_diff")
    c = sub.add_parser("compare", help="Compare matched precision runs with identical model+data+layers")
    c.add_argument("--run-a", required=True)
    c.add_argument("--run-b", required=True)
    c.add_argument("--out", required=True)
    e = sub.add_parser("export-evidence", help="Create evidence record for Model Mountain v4 integration")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    m = sub.add_parser("mia-steer", help="Run independent MIA steering worker with captured vector")
    m.add_argument("--run", required=True)
    m.add_argument("--emotion", choices=EMOTIONS, required=True)
    m.add_argument("--data", default="data/emotion_pairs_v1.jsonl")
    m.add_argument("--out", required=True)
    m.add_argument("--coefficients", default="0,3,6")
    m.add_argument("--max-examples", type=int, default=3)
    m.add_argument("--aperture-mib", type=int, default=512)
    args = p.parse_args()
    if args.command == "doctor":
        info = runtime_metadata()
        info["mia_compute_capability_candidate"] = any(tuple(cc) >= (7,5) for cc in info.get("gpu_cc", []))
        info["nf4_compute_capability_candidate"] = any(tuple(cc) >= (6,0) for cc in info.get("gpu_cc", []))
        info["compatibility_note"] = "Compute capability alone does not verify package, driver or execution compatibility; run real experiments."
        print(json.dumps(info, indent=2))
    elif args.command == "validate":
        pairs = load_pairs(args.data)
        from .dataset import validate_pairs
        print(json.dumps({"pairs": len(pairs), "sha256": dataset_sha256(args.data), "counts": validate_pairs(pairs)}, indent=2))
    elif args.command == "smoke":
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 0.1, (8, 4, 32)).astype(np.float32)
        neu = rng.normal(0, 0.1, (8, 4, 32)).astype(np.float32)
        pos[:, :, :3] += 0.5
        vec = normalized_difference(pos[:4], neu[:4])
        val = layer_metrics(pos[4:6], neu[4:6], vec)
        layer = select_layer(val, [5,11,17,23])
        test = layer_metrics(pos[6:], neu[6:], vec)
        result = {"origin": "SYNTHETIC_NUMERICAL_SMOKE_ONLY", "contains_real_model_activations": False,
                  "selected_layer_0based": layer, "val_metrics": val, "test_metrics": test}
        write_json(args.out, result)
        print(json.dumps({"smoke_status": "pass", "evidence": "synthetic_only", "out": args.out}))
    elif args.command == "run":
        import multiprocessing as mp
        if args.backend == "mia":
            mp.set_start_method("spawn", force=True)
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        scores = run_experiment(backend_kind=args.backend, model=args.model, precision=args.precision,
                                data_path=args.data, output=args.out, layers=layers,
                                max_length=args.max_length, mia_config=args.mia_config,
                                aperture_mib=args.aperture_mib,
                                limit_per_split=args.limit_per_split,
                                generate_samples=args.generate_samples,
                                coefficients=tuple(float(x) for x in args.coefficients.split(",")),
                                vector_method=args.vector_method)
        print(json.dumps(scores, indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_runs(args.run_a, args.run_b, args.out), indent=2))
    elif args.command == "export-evidence":
        from .model_mountain_bridge import export_evidence
        print(json.dumps(export_evidence(args.run, args.out), indent=2))
    elif args.command == "mia-steer":
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        from .mia_steering import run_steering
        result = run_steering(args.run, args.emotion, args.data, args.out,
                              tuple(float(x) for x in args.coefficients.split(",")),
                              max_examples=args.max_examples, aperture_mib=args.aperture_mib)
        print(f"Recorded {len(result)} real MIA generations")


if __name__ == "__main__":
    main()
