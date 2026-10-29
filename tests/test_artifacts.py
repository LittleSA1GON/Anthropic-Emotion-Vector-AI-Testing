import tempfile
import unittest
from pathlib import Path
import numpy as np
from mm_emotion.artifacts import load_capture, save_capture


class ArtifactTests(unittest.TestCase):
    def test_roundtrip_and_checksum(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "test.npz"
            a = np.ones((2, 2, 3), dtype=np.float32)
            save_capture(file, ["a", "b"], [1, 3], a, a * 0, metadata={"origin": "synthetic_unit_test"})
            ids, layers, x, y, meta = load_capture(file)
            self.assertEqual(ids, ["a", "b"])
            self.assertEqual(layers, [1, 3])
            self.assertEqual(meta["origin"], "synthetic_unit_test")
            np.testing.assert_allclose(x, a)
            with file.open("ab") as f:
                f.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_capture(file)
