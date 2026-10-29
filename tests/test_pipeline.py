import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mm_emotion.pipeline import run_experiment, compare_runs
from mm_emotion.model_mountain_bridge import export_evidence


class FakeBackend:
    """Only for unit tests; never a source of empirical evidence."""
    def __init__(self):
        self.call_counter = 0
    def capture(self, text, layers):
        self.call_counter += 1
        e = ["delighted", "steady", "angry", "worried"]
        signal = next((i for i, w in enumerate(e) if w in text), None)
        v = np.zeros((len(layers), 32), dtype=np.float32)
        if signal is not None:
            for k in range(len(layers)):
                v[k, signal] = 1. + .1 * k
        # real neutral prompts have no emotion tokens from the list
        return v


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_artifacts_and_checks(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a"
            b = Path(d) / "b"
            with patch("mm_emotion.pipeline.build_backend", side_effect=lambda *args, **kwargs: FakeBackend()):
                run_experiment(backend_kind="hf", model="FAKE_TEST_MODEL", precision="fp32",
                               data_path="data/emotion_pairs_v1.jsonl", output=a, layers=[0,1,2,3],
                               limit_per_split=2)
                run_experiment(backend_kind="hf", model="FAKE_TEST_MODEL", precision="nf4",
                               data_path="data/emotion_pairs_v1.jsonl", output=b, layers=[0,1,2,3],
                               limit_per_split=2)
            self.assertEqual(json.loads((a / "manifest.json").read_text())["status"], "complete")
            rows = compare_runs(a, b, Path(d) / "comparison.json")
            self.assertEqual(len(rows), 16)
            self.assertAlmostEqual(rows[0]["direction_cosine"], 1)
            evidence = export_evidence(a, Path(d) / "evidence.json")
            self.assertTrue(evidence["not_a_security_verdict"])
            with self.assertRaises(FileExistsError):
                with patch("mm_emotion.pipeline.build_backend", return_value=FakeBackend()):
                    run_experiment(backend_kind="hf", model="FAKE_TEST_MODEL", precision="fp16",
                                   data_path="data/emotion_pairs_v1.jsonl", output=a, layers=[0,1,2,3])
