# Model Mountain Emotion Lab v0.2 — Visual Workbench Operator Guide

## Goal
Load one model, enter or import neutral/emotional prompt pairs, extract directions for chosen decoder blocks, and compare **baseline vs one isolated activation intervention** with adjustable layers, strength and scope. Save results with their settings and provenance.

## Local GTX 1080 Ti / modern NVIDIA GPU: visual GUI

1. Unzip the suite. Install appropriate PyTorch **first** (GTX 1080 Ti: CUDA 12.x build retaining Pascal sm_61; follow `scripts/setup_gtx1080ti.ps1` or `.sh`).
2. `python -m pip install -r requirements-hf.txt -r requirements-ui.txt`
3. `python -m streamlit run app_emotion.py` (or `scripts/launch_workbench.ps1` / `.sh`).
4. In sidebar select checkpoint, precision **FP16 or NF4** on GPU, then **Load model**. Switching precision unloads previous weights; it does **not** merge GPU memory.
5. In **Enter / import data**: load example emotion pairs, type/edit in the grid, or import `.csv`, `.jsonl`, `.json`. Click **Use these pairs for discovery**. At least two distinct pairs are needed.
6. In **Find / inspect vectors**: select 0-based decoder layers, fit mean difference, compare training-only layer scores, and save a vector. Or select a vector from a full CLI run's `runs/**/vectors/*.npz`.
7. In **Isolate / steer model**: choose **layer**, **normal/reversed** direction, strength, intervention scope, generation tokens, temperature, top-p, seed. Paste a test prompt. Click **Run real baseline + intervention**. Includes coefficient 0 with no hook. Use **Dose-response sweep** for curves.
8. Review side-by-side text and the result table. In **Runs & exports**, filter and download session JSON/CSV.

## Kaggle T4: notebook-native widgets (no exposed public server)

1. Upload only `notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb` to Kaggle; enable T4 accelerator and Internet.
2. Set `OPEN_INTERACTIVE_WORKBENCH=True` in the first code cell, run restore and validation, then run the **interactive controls** cell. UI packages are installed if missing.
3. In **1 Model** load the checkpoint/precision. In **2 Data** load example pairs or enter individual contrasts or paste CSV/JSONL. In **3 Find vectors** choose block IDs, fit, and select the vector. In **4 A/B steer** choose layer, sign, coefficient, scope and prompt, then test or sweep.
4. Sessions save under `runs/workbench_sessions`. Re-run the notebook **results export** cell *after* interactive tests to include new sessions in Kaggle Outputs.

## Isolation: what changes, what does not

- For the same loaded model: **the only experimental intervention is adding `coefficient × direction` at the chosen decoder block output**. Coefficient 0 has NO hook. A hook is detached in `finally` after each generation.
- `last_token`: add to the most recent token in each forward pass (recommended default).
- `prefill_only`: only when initial prefill has more than one token. One-token initial prompts are not distinguished from decode by this heuristic.
- `decode_only`: only one-token decode passes. Does not affect prefill.
- `all_tokens`: add to every token in prefill and decode (strong; likely larger behavior disruption).
- `temperature=0` and same seed maintain a deterministic decoding setting. Sampling seeds reset per A/B generation; exact bitwise reproducibility across kernels is not guaranteed.
- `direction_sign=-1` reverses vector; choose negative coefficient for another way to reverse its effect (avoid simultaneously reversing both by accident).
- Model weights and optimizers are **never modified**, and vector scope does **not** alter layers outside the selected block.
- For precision comparisons, first capture and fit vector in **FP16**, then unload FP16, load same model in **NF4**, and explicitly enable cross-precision transfer before selecting the FP16 vector. The UI refuses any different model ID and all incompatible vector dimensions.
- For true apples-to-apples comparisons, hold prompt, tokenizer, model revision, chosen layer, sign, scope, max tokens and decoding settings fixed; change only the variable under test.

## Quick-fit versus rigorous evidence

**Quick fit:** no separate validation set; its high layer scores are biased in-sample. Save it as training-only exploration and use fresh held-out prompts for causal A/B examples. Do not report it as a validated emotional classifier.

**Rigorous:** use the main CLI `python -m mm_emotion.cli run ...` to train on train split, choose layer on validation, and evaluate on test; your custom full dataset must satisfy `mm_emotion.dataset.validate_pairs` (2+ pairs per each of the 4 emotions × each of 3 splits, with no cross-split scenario leakage).

**Research caveat:** a visible difference in output tone is not evidence of a model feeling emotions or a specific discovered causal mechanism. Include lexical controls, negative directions, multiple seeds/prompts, blinded annotations and held-out controls.

## Input formats

For quick fit use `.csv` with headers `neutral_prompt,emotion_prompt` (other fields optional), or `.jsonl` with one object per row; `.json` supports an array or `{"pairs": [...]}`. A starter exists at `data/templates/pairs_template.csv`. Exported custom datasets are in `runs/workbench_datasets` and real session files in `runs/workbench_sessions`. Never upload private or sensitive material to hosted third-party model services without authorization.

## Fast automated checks

```bash
python -m pytest -q
python -m mm_emotion.cli validate
python -m mm_emotion.cli smoke --out runs/synthetic_smoke.json
python scripts/build_notebook.py
```

## Troubleshooting

- **No real model loaded:** choose a small model, correct precision, and click Load model. CPU smoke tests alone do not load weights.
- **CUDA OOM:** unload; choose 0.5B/1.5B, NF4, short contexts, few layers/prompts and small token budget. Do not run two GPU model instances concurrently.
- **Layer error:** decoder blocks are 0-based in HF UI; IBM MIA capture JSON uses its own 1-based configuration mapping.
- **Vector mismatch:** only identical model IDs; cross-precision use requires explicit toggle; cross-model vector transfer is intentionally blocked.
- **No MIA in GUI:** MIA's separate-vLLM worker requires its own process. Use existing Kaggle `RUN_MIA` path after the HF pilot, rather than mixing runtimes in one long-lived Streamlit session.
