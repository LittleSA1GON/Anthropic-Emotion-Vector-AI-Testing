## v0.2 — Interactive Emotion Vector Workbench

- Editable custom prompt pairs; CSV/JSONL/JSON import/export; starter dataset templates.
- Streamlit 4-tab interface with model load/unload and isolated baseline-versus-steering tests.
- Explicit decoder layer, vector sign, strength, intervention scope, precision, sampling and batch sweep knobs.
- Vectors fit over selected layers, display training-only directional diagnostics, saved with provenance JSON.
- Matching model ID required on vector load; cross-precision transfer explicitly opt-in; mismatched hidden dimensions blocked.
- Kaggle-native ipywidgets 4-tab interface bundled into fully self-contained notebook; no public Streamlit server needed.
- Read-only run session history and JSON/CSV exports; zero-coefficient baseline always hook-free.
- New workbench tests and operator/agent documentation; no claims of GPU validation.

# Release notes — Model Mountain Emotion Lab v0.1 (2026-10-28)

## Created

- New separate research package for actual decoder-block activations using Transformers forward hooks, CPU-stored float32 tensors and normalized contrast vectors.
- Independent IBM MIA capture/steer runners conforming to public MIA configuration/API documentation (not GPU-tested).
- Quantized bitsandbytes NF4 loading path and controlled same-model FP16/NF4 direction comparison.
- Mean-difference and optional pairwise-PCA vector fitting; validation-selected layers, fixed held-out test protocol.
- Deterministic synthetic 48 matched-pair dataset (12 scenarios x 4 emotions, split by scenario).
- Optional causal generation at multiple steering coefficients, randomized blinded human annotation template, rating summaries.
- Model Mountain research-only evidence export and no-overwrite extension copier.
- Machine-readable handoff JSON and extensive agent/dataset/experiment cards.
- Single-upload Kaggle notebook embedding complete source and data.
- CPU unit tests and synthetic pipeline smoke; notebook CPU execution verified.

## Verification

10 CPU tests passing. Notebook restored project, executed all default cells without errors, validated data, and produced explicitly synthetic smoke evidence in a separate local test location. No GPU available in authoring environment. **No true Qwen model activations, NF4 result, MIA capture or emotion steering result is included or asserted.** Original Model Mountain v4 ZIP cannot be raw-materialized in the authoring environment; this is a companion, not a directly patched v4 release.
