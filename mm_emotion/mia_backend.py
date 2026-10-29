"""Genuine IBM MIA capture adapter. Must be invoked from a standalone Python process.

Dependency source: https://github.com/IBM/vLLM-Hook/docs/configs.md
MIA 1-based capture layer IDs are mapped to these suite's 0-based block IDs.
"""
from __future__ import annotations
import os
import numpy as np


class MIABackend:
    def __init__(self, model_id: str, precision: str, max_length: int,
                 *, config_file: str, aperture_mib: int = 512, gpu_utilization: float = 0.55):
        if precision not in ("fp16", "awq", "gptq"):
            raise ValueError("MIA supports fp16 or prequantized AWQ/GPTQ paths in this suite")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ["MIA_APERTURE_GPU_BYTES"] = str(aperture_mib * 1024 * 1024)
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5):
            raise RuntimeError("MIA path requires CUDA compute capability >= 7.5")
        from mia import MiaLLM
        from transformers import AutoTokenizer
        self.model_id = model_id
        self.precision = precision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        opts = {"model": model_id, "worker_name": "capture_hs", "analyzer_name": "hidden_states",
                "config_file": config_file, "gpu_memory_utilization": gpu_utilization,
                "max_model_len": max_length, "enforce_eager": True,
                "tensor_parallel_size": 1, "dtype": "float16"}
        if precision != "fp16":
            opts["quantization"] = precision
        self.engine = MiaLLM(**opts)
        self.max_length = max_length

    def format_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "User: " + prompt + "\nAssistant:"

    def capture(self, prompt: str, layer_ids: list[int]) -> np.ndarray:
        from vllm import SamplingParams
        formatted = self.format_prompt(prompt)
        count = len(self.tokenizer(formatted, add_special_tokens=False)["input_ids"])
        if count + 1 > self.max_length:
            raise ValueError("Prompt exceeds MIA max_model_len")
        # Explicitly clear any prefix cache; without this repeated prefixes may produce empty probes.
        self.engine.llm_engine.reset_prefix_cache()
        out = self.engine.generate(formatted,
                                   SamplingParams(temperature=0.0, max_tokens=1,
                                                  extra_args={"hooks_on": "prefill", "hs_mode": "last_token"}),
                                   save_to_disk=False)
        if not out or getattr(out[0], "probes", None) is None:
            raise RuntimeError("MIA returned no genuine probe data")
        stats = self.engine.analyze(probes=out[0].probes, analyzer_spec={"reduce": "none"})
        hs = stats.get("hidden_states", {})
        captured = []
        for block_id in layer_ids:
            key = f"model.layers.{block_id}"
            if key not in hs or not hs[key]:
                raise RuntimeError(f"Missing genuine MIA capture for {key}; got keys {list(hs)}")
            tensor = hs[key][0]
            if hasattr(tensor, "detach"):
                arr = tensor.detach().float().cpu().numpy()
            else:
                arr = np.asarray(tensor, dtype=np.float32)
            if arr.ndim < 1 or arr.ndim > 3 or arr.shape[-1] < 32:
                raise RuntimeError(f"Unexpected MIA tensor shape: {arr.shape}")
            captured.append(arr.reshape(-1, arr.shape[-1])[-1].astype(np.float32).copy())
        output = np.stack(captured)
        if not np.isfinite(output).all():
            raise RuntimeError("Nonfinite MIA capture")
        return output

    def close(self):
        try:
            self.engine.llm_engine.engine_core.shutdown()
        except Exception:
            pass
