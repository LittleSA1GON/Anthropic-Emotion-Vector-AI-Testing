import csv
import io
import json
from pathlib import Path
import numpy as np
import pytest
from mm_emotion.workbench import (parse_pairs, pair_to_jsonl, make_quick_vector,
    save_quick_vector, load_vector, sweep_grid, validate_test_prompts, serialize_results)

PAIRS = [
    {"neutral_prompt": "status a", "emotion_prompt": "happy a"},
    {"neutral_prompt": "status b", "emotion_prompt": "happy b"},
    {"neutral_prompt": "status c", "emotion_prompt": "happy c"},
]

class FakeBackend:
    model_id = "fake/toy"
    precision = "fp32"
    layers = list(range(4))
    def capture(self, prompt, layer_ids):
        base = np.array([1.0 if prompt.startswith("happy") else 0.0, len(prompt)*0.01, 0], dtype=np.float32)
        return np.stack([base + np.array([i*0.1, i*0.05, 0], dtype=np.float32) for i in layer_ids])


def test_import_csv_and_roundtrip():
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=["neutral_prompt", "emotion_prompt"])
    w.writeheader(); w.writerows(PAIRS)
    assert parse_pairs(b.getvalue(), "csv") == PAIRS
    assert parse_pairs(pair_to_jsonl(PAIRS), "jsonl") == PAIRS
    assert parse_pairs(json.dumps({"pairs": PAIRS}), "json") == PAIRS


def test_import_rejects_same_prompt_and_missing_values():
    with pytest.raises(ValueError):
        parse_pairs('[{"neutral_prompt":"a","emotion_prompt":"a"}]', "json")
    with pytest.raises(ValueError):
        parse_pairs('[{"neutral_prompt":"a"}]', "json")
    with pytest.raises(ValueError):
        parse_pairs('[]', "txt")


def test_quick_fit_and_save_with_provenance(tmp_path):
    v, meta = make_quick_vector(FakeBackend(), PAIRS, [0, 2, 3], label="test")
    assert v.shape == (3, 3)
    assert meta["evaluation"] == "training_only_not_held_out"
    assert meta["origin"] == "real_model_inference"  # fake backend used ONLY in a unit test
    assert all(np.isclose(np.linalg.norm(x), 1) for x in v)
    p = save_quick_vector(tmp_path, "test", v, meta)
    restored, layers, provenance = load_vector(p, model_id="fake/toy", precision="fp32",
                                                num_layers=4, hidden_size=3)
    assert np.allclose(restored, v)
    assert layers == [0, 2, 3]
    assert provenance["pairs_sha256"]
    with pytest.raises(FileExistsError):
        save_quick_vector(tmp_path, "test", v, meta)


def test_rejects_model_precision_or_size_mismatch(tmp_path):
    v, meta = make_quick_vector(FakeBackend(), PAIRS, [1, 2])
    p = save_quick_vector(tmp_path, "vectors", v, meta)
    for kwargs in [{"model_id": "different"}, {"precision": "nf4"},
                   {"hidden_size": 40}, {"num_layers": 2}]:
        with pytest.raises(ValueError):
            load_vector(p, **kwargs)


def test_rejects_synthetic_metadata_and_nonfinite_vectors(tmp_path):
    v, meta = make_quick_vector(FakeBackend(), PAIRS, [1, 2])
    p = save_quick_vector(tmp_path, "v", v, meta)
    meta["origin"] = "synthetic"
    p.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_vector(p)
    with pytest.raises(ValueError):
        save_quick_vector(tmp_path, "bad", np.full((2, 3), np.nan), meta)


def test_dose_grid_injects_baseline_and_rejects_nans():
    assert sweep_grid("-3,3,3") == [0.0, -3.0, 3.0]
    for c in ["nan", "inf", "200", "0,1,2,3,4,5,6,7,8,9"]:
        with pytest.raises(ValueError):
            sweep_grid(c)
    assert validate_test_prompts("hello\nworld") == ["hello", "world"]
    with pytest.raises(ValueError):
        validate_test_prompts("  ")


def test_session_json_is_append_only(tmp_path):
    p = tmp_path / "session.json"
    serialize_results(p, [{"coefficient": 0, "text": "ok"}], {"model_id": "fake/toy"})
    assert json.loads(p.read_text())["origin"] == "real_model_generation"
    with pytest.raises(FileExistsError):
        serialize_results(p, [], {})


def test_isolated_hook_changes_only_chosen_phase_and_is_removed():
    import torch
    from mm_emotion.hf_backend import HFBackend

    class Block(torch.nn.Module):
        def forward(self, x):
            return x

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = type("Cfg", (), {"hidden_size": 3})()
            self.block = Block()
            self.recorded = []

        def generate(self, input_ids, **kwargs):
            self.recorded = []
            for tokens in [4, 1]:
                output = self.block(torch.ones((1, tokens, 3)))
                self.recorded.append(output.detach().clone())
            return torch.cat([input_ids, torch.ones((1, 2), dtype=torch.int64)], dim=1)

    class Tokenizer:
        eos_token_id = 0
        def decode(self, *args, **kwargs):
            return "ok"

    demo = HFBackend.__new__(HFBackend)
    demo.torch = torch
    demo.model = DummyModel()
    demo.tokenizer = Tokenizer()
    demo.layers = [demo.model.block]
    demo.device = torch.device("cpu")
    demo.inputs = lambda prompt: {"input_ids": torch.tensor([[4, 5]], dtype=torch.int64)}
    direction = np.array([1., 0., 0.], dtype=np.float32)
    before = demo.model.weight.clone()

    demo.generate("same", 0, direction, 0.0, intervention_scope="all_tokens")
    assert all(torch.all(t == 1.) for t in demo.model.recorded)
    assert not demo.model.block._forward_hooks

    demo.generate("same", 0, direction, 2.0, intervention_scope="prefill_only")
    assert demo.model.recorded[0][0, -1, 0].item() == 3.0
    assert demo.model.recorded[1][0, -1, 0].item() == 1.0
    assert not demo.model.block._forward_hooks

    demo.generate("same", 0, direction, 2.0, intervention_scope="decode_only")
    assert demo.model.recorded[0][0, -1, 0].item() == 1.0
    assert demo.model.recorded[1][0, -1, 0].item() == 3.0
    assert not demo.model.block._forward_hooks

    demo.generate("same", 0, direction, -2.0, intervention_scope="all_tokens")
    assert all(torch.all(t[..., 0] == -1.) for t in demo.model.recorded)
    assert not demo.model.block._forward_hooks
    assert torch.equal(before, demo.model.weight)

    # Even a generation exception must leave no intervention installed.
    def fail_generate(**kwargs):
        demo.model.block(torch.ones((1,1,3)))
        raise RuntimeError('generation failed')
    demo.model.generate=fail_generate
    with pytest.raises(RuntimeError,match='generation failed'):
        demo.generate('same',0,direction,2.0,intervention_scope='prefill_only')
    assert not demo.model.block._forward_hooks
