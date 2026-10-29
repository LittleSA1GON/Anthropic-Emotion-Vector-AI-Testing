import json
import tempfile
import unittest
from pathlib import Path
from mm_emotion.model_mountain_bridge import export_evidence


class BridgeTests(unittest.TestCase):
    def test_refuses_synthetic_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "origin": "synthetic"}))
            (root / "summary.json").write_text(json.dumps({"metrics": []}))
            with self.assertRaises(ValueError):
                export_evidence(root, root / "bridge.json")
