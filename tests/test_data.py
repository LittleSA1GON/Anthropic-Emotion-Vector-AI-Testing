import copy
import unittest
from mm_emotion.dataset import load_pairs, validate_pairs


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_pairs("data/emotion_pairs_v1.jsonl")

    def test_counts(self):
        counts = validate_pairs(self.rows)
        self.assertEqual(len(self.rows), 48)
        for emotion in ("joy", "calm", "anger", "anxiety"):
            self.assertEqual(counts["train/" + emotion], 6)
            self.assertEqual(counts["validation/" + emotion], 3)
            self.assertEqual(counts["test/" + emotion], 3)

    def test_leakage(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_pairs(rows)

    def test_duplicate_id(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["id"] = rows[0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_pairs(rows)
