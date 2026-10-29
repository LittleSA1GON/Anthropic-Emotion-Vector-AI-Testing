"""Pure NumPy layer selection and evaluation; no model or GPU dependency."""
from __future__ import annotations
import numpy as np


def normalized_difference(positive: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """Inputs [pairs,layers,hidden], output unit vectors [layers,hidden]."""
    if positive.shape != neutral.shape or positive.ndim != 3:
        raise ValueError("Paired inputs must match and have shape [N,L,H]")
    if positive.shape[0] < 2:
        raise ValueError("At least 2 paired observations required")
    if not np.isfinite(positive).all() or not np.isfinite(neutral).all():
        raise ValueError("Nonfinite activations")
    raw = np.mean(positive.astype(np.float64) - neutral.astype(np.float64), axis=0)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero or near-zero emotion contrast at one or more layers")
    return (raw / norms).astype(np.float32)


def pairwise_pca_direction(positive: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """Principal contrast variability direction, sign-aligned with mean contrast.

    This is exploratory and is not equivalent to a causal emotion direction.
    """
    if positive.shape != neutral.shape or positive.ndim != 3 or positive.shape[0] < 2:
        raise ValueError("Paired inputs must match [N,L,H] and N>=2")
    if not np.isfinite(positive).all() or not np.isfinite(neutral).all():
        raise ValueError("Nonfinite activations")
    differences = positive.astype(np.float64) - neutral.astype(np.float64)
    means = differences.mean(axis=0)
    vectors = []
    for layer in range(differences.shape[1]):
        centered = differences[:, layer, :] - means[layer]
        # Economical SVD works when N << hidden_size.
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        if singular_values[0] < 1e-10:
            direction = means[layer]
        else:
            direction = vt[0]
        if np.dot(direction, means[layer]) < 0:
            direction = -direction
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("Zero contrast direction")
        vectors.append((direction / norm).astype(np.float32))
    return np.stack(vectors)


def layer_metrics(positive: np.ndarray, neutral: np.ndarray, directions: np.ndarray) -> list[dict]:
    if positive.shape != neutral.shape or positive.ndim != 3:
        raise ValueError("Invalid activation shapes")
    if directions.shape != positive.shape[1:]:
        raise ValueError("Direction/layer/hidden dimension mismatch")
    delta = positive.astype(np.float64) - neutral.astype(np.float64)
    margin = np.einsum("nlh,lh->nl", delta, directions.astype(np.float64))
    results = []
    for layer in range(margin.shape[1]):
        m = margin[:, layer]
        sd = float(m.std(ddof=1)) if len(m) > 1 else 0.0
        results.append({"layer_position": layer, "n_pairs": len(m), "mean_margin": float(m.mean()),
                        "median_margin": float(np.median(m)), "positive_fraction": float(np.mean(m > 0)),
                        "margin_std": sd, "standardized_margin": float(m.mean() / max(sd, 1e-6))})
    return results


def select_layer(validation_metrics: list[dict], layer_ids: list[int]) -> int:
    """Layer is frozen from validation; never select using the test set."""
    if len(validation_metrics) != len(layer_ids):
        raise ValueError("Layer ID count mismatch")
    # Minimize chance of picking an overfit high-ratio layer: rank by directional consistency,
    # then the absolute margin with deterministic lower-index tie-break.
    winner = max(range(len(layer_ids)), key=lambda i:
                 (validation_metrics[i]["positive_fraction"], validation_metrics[i]["mean_margin"], -layer_ids[i]))
    return layer_ids[winner]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vector dimensionality mismatch")
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        raise ValueError("Zero vector")
    return float(np.dot(a.ravel().astype(float), b.ravel().astype(float)) / den)
