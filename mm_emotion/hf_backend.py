"""Capture actual decoder-block outputs and perform causal steering with Transformers.

Training prompt captures are prefill-only last-input-token activations. This intentionally
matches the target MIA `hooks_on=prefill, hs_mode=last_token` semantic, subject to backend checks.
"""
from __future__ import annotations
import time
from contextlib import contextmanager
import numpy as np


def decoder_layers(model):
    paths = (("model", "layers"), ("transformer", "h"), ("model", "decoder", "layers"),
             ("language_model", "model", "layers"))
    for path in paths:
        obj = model
        try:
            for name in path:
                obj = getattr(obj, name)
            if len(obj) and hasattr(obj[0], "register_forward_hook"):
                return obj
        except (AttributeError, TypeError, IndexError):
            pass
    raise RuntimeError("Unsupported decoder layer layout; do not substitute synthetic activations")


class HFBackend:
    def __init__(self, model_id: str, precision: str = "fp16", max_length: int = 512, *, token=None, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.model_id, self.precision, self.max_length = model_id, precision, max_length
        if precision not in ("fp16", "nf4", "fp32"):
            raise ValueError("precision must be fp16, nf4 or fp32")
        from .credentials import model_token
        auth = model_token(token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, token=auth,revision=revision)
        kwargs = {"trust_remote_code": False, "attn_implementation": "eager", "token": auth,"revision":revision}
        if precision == "nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("NF4 experiment requires CUDA in this suite")
            kwargs.update({"quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
                           "device_map": {"": 0}, "torch_dtype": torch.float16})
        else:
            kwargs.update({"torch_dtype": torch.float32 if precision == "fp32" else torch.float16,
                           "device_map": {"": 0} if torch.cuda.is_available() else "cpu"})
            if not torch.cuda.is_available() and precision == "fp16":
                raise RuntimeError("FP16 requires CUDA in this suite; use fp32 on CPU")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.layers = decoder_layers(self.model)
        self.device = next(self.model.parameters()).device

    def format_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "User: " + prompt + "\nAssistant:"

    def inputs(self, prompt):
        ids = self.tokenizer(self.format_prompt(prompt), return_tensors="pt", add_special_tokens=False)
        if ids["input_ids"].shape[-1] > self.max_length:
            raise ValueError("Prompt exceeds configured token limit; do not silently truncate")
        return {k: v.to(self.device) for k, v in ids.items()}

    def capture(self, prompt: str, layer_ids: list[int]) -> np.ndarray:
        """Actual per-block outputs [layers,hidden], output always float32 CPU."""
        if not layer_ids or len(set(layer_ids)) != len(layer_ids) or any(i < 0 or i >= len(self.layers) for i in layer_ids):
            raise ValueError("Invalid 0-based decoder block indices")
        cache = {}
        hooks = []
        for i in layer_ids:
            def recorder(_module, _inputs, output, idx=i):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("Unexpected decoder hidden-state shape")
                cache[idx] = hidden[:, -1, :].detach().float().cpu().numpy()[0].copy()
            hooks.append(self.layers[i].register_forward_hook(recorder))
        try:
            with self.torch.inference_mode():
                _ = self.model(**self.inputs(prompt), use_cache=False)
        finally:
            for handle in hooks:
                handle.remove()
        if set(cache) != set(layer_ids):
            raise RuntimeError("Missing decoder activations: capture invalid")
        out = np.stack([cache[i] for i in layer_ids])
        if not np.isfinite(out).all():
            raise RuntimeError("Captured activations contain nonfinite values; retry explicitly with fp32 and record it as a separate precision experiment")
        return out

    def generate(self, prompt: str, layer_id: int | None = None,
                 direction: np.ndarray | None = None, coefficient: float = 0.0,
                 max_new_tokens: int = 48, *, temperature: float = 0.0,
                 top_p: float = 1.0, seed: int = 42,
                 intervention_scope: str = "last_token", capture_layer: int | None = None) -> dict:
        """Intervene in decoder-block output, not in model weights.

        last_token: add only to the final token of each model forward pass.
        all_tokens: add to every token in each pass (stronger and less localized).
        prefill_only/decode_only: target the respective phases at last token.
        Baseline (coefficient=0) always removes the hook for clean A/B comparison.
        """
        if layer_id is not None and (layer_id < 0 or layer_id >= len(self.layers)):
            raise ValueError("Invalid 0-based decoder layer")
        if capture_layer is not None and capture_layer not in range(len(self.layers)):
            raise ValueError("Invalid capture layer")
        if intervention_scope not in ("last_token", "all_tokens", "prefill_only", "decode_only"):
            raise ValueError("Invalid intervention scope")
        if not np.isfinite(coefficient) or abs(coefficient) > 30:
            raise ValueError("Coefficient must be finite and within [-30,30]")
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be 1-512")
        if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
            raise ValueError("temperature/top_p out of range")
        handle = None
        if direction is not None and coefficient:
            if layer_id is None:
                raise ValueError("Steering requires layer_id")
            dim = self.model.config.hidden_size
            direction = np.asarray(direction, dtype=np.float32)
            if direction.shape != (dim,) or not np.isfinite(direction).all():
                raise ValueError("Direction dimension/finite check failed")
            delta = self.torch.as_tensor(direction.copy(), device=self.device, dtype=getattr(self.model, "dtype", next(self.model.parameters()).dtype))
            first_pass = True
            def modify(_module, _inputs, output):
                nonlocal first_pass
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.ndim != 3:
                    raise RuntimeError("Unexpected block output shape during steering")
                prefill = first_pass
                first_pass = False
                if (intervention_scope == "prefill_only" and not prefill) or (intervention_scope == "decode_only" and prefill):
                    return output
                updated = hidden.clone()
                if intervention_scope == "all_tokens":
                    updated += coefficient * delta
                else:
                    updated[:, -1, :] += coefficient * delta
                return (updated, *output[1:]) if isinstance(output, tuple) else updated
            handle = self.layers[layer_id].register_forward_hook(modify)
        observation_handle = None
        observed = []
        try:
            if capture_layer is not None:
                def observe_prefill(_module, _inputs, output):
                    if not observed:
                        hidden = output[0] if isinstance(output, tuple) else output
                        observed.append(hidden[0, -1, :].detach().float().cpu().numpy().copy())
                # Registered after the intervention: the snapshot is the actual modified output.
                observation_handle = self.layers[capture_layer].register_forward_hook(observe_prefill)
            inputs = self.inputs(prompt)
            # Reset RNG for identical sampling seeds between baseline and interventions.
            self.torch.manual_seed(int(seed))
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(int(seed))
            kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True,
                          do_sample=temperature > 0,
                          pad_token_id=self.tokenizer.eos_token_id)
            if temperature > 0:
                kwargs.update(temperature=float(temperature), top_p=float(top_p))
            tic = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **kwargs)
            elapsed = time.perf_counter() - tic
            new_ids = generated[0][inputs["input_ids"].shape[-1]:]
            result = {"text": self.tokenizer.decode(new_ids, skip_special_tokens=True),
                    "duration_seconds": elapsed, "new_tokens": int(new_ids.numel()),
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "scope": intervention_scope, "seed": int(seed)}
            if capture_layer is not None:
                if not observed or not np.isfinite(observed[0]).all():
                    raise RuntimeError("Missing or nonfinite generation activation snapshot")
                result.update(observed_prefill_activation=observed[0].tolist(),
                              observed_prefill_layer_0based=capture_layer,
                              activation_origin="real_prefill_decoder_block_output_after_intervention")
            return result
        finally:
            if observation_handle is not None:
                observation_handle.remove()
            if handle is not None:
                handle.remove()

    def describe(self):
        from .inspection import describe_backend
        return describe_backend(self)

    def inspect_weights(self, layers):
        from .inspection import inspect_weights
        return inspect_weights(self,layers)

    def close(self):
        """Release weights before another checkpoint is loaded."""
        import gc
        self.model = None
        self.layers = []
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
