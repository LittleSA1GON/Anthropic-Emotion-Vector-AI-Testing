import unittest
import numpy as np
from mm_emotion.vectors import normalized_difference, pairwise_pca_direction, layer_metrics, select_layer, cosine


class VectorTests(unittest.TestCase):
    def test_direction_and_selection(self):
        neutral = np.zeros((6, 3, 8), dtype=np.float32)
        positive = neutral.copy()
        positive[:, 0, 0] = 1
        positive[:, 1, 1] = 2
        positive[:, 2, 2] = -1
        vectors = normalized_difference(positive[:3], neutral[:3])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-6)
        m = layer_metrics(positive[3:], neutral[3:], vectors)
        self.assertEqual(select_layer(m, [2, 4, 6]), 4)
        self.assertEqual(m[1]["positive_fraction"], 1.0)

    def test_directions_reject_nonfinite(self):
        p = np.ones((2, 2, 3), dtype=np.float32)
        n = np.zeros_like(p)
        p[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            normalized_difference(p, n)

    def test_pairwise_pca(self):
        rng = np.random.default_rng(123)
        neutral = rng.normal(size=(6, 2, 8)).astype(np.float32)
        positive = neutral.copy()
        positive[:, :, 0] += 2
        positive[:, :, 1] += np.arange(6)[:, None] * .1
        vec = pairwise_pca_direction(positive, neutral)
        self.assertEqual(vec.shape, (2, 8))
        np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 1, rtol=1e-6)

    def test_cosine(self):
        self.assertAlmostEqual(cosine(np.array([1, 0]), np.array([1, 0])), 1)
        self.assertAlmostEqual(cosine(np.array([1, 0]), np.array([0, 1])), 0)
