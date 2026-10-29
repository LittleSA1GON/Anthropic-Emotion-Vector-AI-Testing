"""Run a separate genuine MIA steer worker (not simultaneously with capture worker)."""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
from .artifacts import runtime_metadata, write_json, write_jsonl
from .dataset import load_pairs


def run_steering(run_dir, emotion, dataset, output, coefficients=(0., 3., 6.), max_examples=3,
                 aperture_mib=512, max_model_len=512):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("MIA steering requires CUDA compute capability >=7.5")
    from mia import MiaLLM
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    run_dir, output = Path(run_dir), Path(output)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Capture experiment must be complete")
    metadata = json.loads((run_dir / "vectors" / f"{emotion}.json").read_text())
    vector_path = (run_dir / "vectors" / f"{emotion}_mia.pt").resolve()
    if not vector_path.exists():
        raise FileNotFoundError("Missing torch-exported MIA vector")
    model_id = manifest["model_id"]
    # Same weight precision is a necessary constraint; don't use a different model silently.
    settings = {"method": "add_vector", "optimal_layer": metadata["layer_0based"],
                "coefficient": 0.0, "vector_path": str(vector_path),
                "phase": "both", "positions": "last_token"}
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "mia_steering_config.json"
    write_json(config_file, {"steering": settings, "model_info": {"model": model_id}})
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["MIA_APERTURE_GPU_BYTES"] = str(aperture_mib * 1024 * 1024)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    def format_prompt(text):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) if tokenizer.chat_template else f"User: {text}\nAssistant:"
    engine_opts = {"model": model_id, "worker_name": "steer", "config_file": str(config_file.resolve()),
                   "max_model_len": max_model_len, "gpu_memory_utilization": 0.55,
                   "enforce_eager": True, "dtype": "float16", "tensor_parallel_size": 1}
    if manifest["precision"] in ("awq", "gptq"):
        engine_opts["quantization"] = manifest["precision"]
    elif manifest["precision"] != "fp16":
        raise ValueError("MIA steer only accepts fp16, awq, gptq precision")
    engine = MiaLLM(**engine_opts)
    records = []
    try:
        rows = [r for r in load_pairs(dataset) if r["emotion"] == emotion and r["split"] == "test"][:max_examples]
        for row in rows:
            text = format_prompt(row["neutral_prompt"])
            for coefficient in coefficients:
                params = SamplingParams(max_tokens=48, temperature=0.0,
                                        extra_args={"steer": {**settings, "coefficient": coefficient}})
                tic = time.perf_counter()
                if coefficient == 0:
                    result = engine.generate(text, SamplingParams(max_tokens=48, temperature=0.0), use_hook=False)
                else:
                    result = engine.generate(text, params)
                generated = result[0].outputs[0].text
                records.append({"pair_id": row["id"], "emotion": emotion, "prompt": row["neutral_prompt"],
                                "coefficient": coefficient, "text": generated,
                                "duration_seconds": time.perf_counter() - tic,
                                "origin": "real_mia_generation"})
    finally:
        try:
            engine.llm_engine.engine_core.shutdown()
        except Exception:
            pass
    write_jsonl(output / "steering_results.jsonl", records)
    shuffled = list(enumerate(records))
    random.Random(42).shuffle(shuffled)
    import csv
    with (output / "blinded_annotations.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "pair_id", "emotion", "response", "emotion_alignment_1to5", "task_quality_1to5", "notes"])
        writer.writeheader()
        for trial, (key, row) in enumerate(shuffled):
            writer.writerow({"trial_id": trial, "pair_id": row["pair_id"], "emotion": row["emotion"],
                             "response": row["text"], "emotion_alignment_1to5": "", "task_quality_1to5": "", "notes": ""})
    write_json(output / "blinding_key_PRIVATE.json", {str(i): records[original]["coefficient"] for i, (original, _) in enumerate(shuffled)})
    write_json(output / "provenance.json", {"schema": "mm-emotion.steering.v1", "origin": "real_mia_generation",
                                                  "source_capture": str(run_dir), "runtime": runtime_metadata(),
                                                  "caution": "Output is not evidence of emotion without blinded human annotation."})
    return records
