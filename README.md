# Model Mountain Emotion Lab

The [dynamic research workspace](docs/DYNAMIC_VISUALIZATION.md) adds evidence-backed compute states, a Blender-authored neural scene, synchronized layer inspection, real parameter samples, exact vector subtraction, and recorded steering comparisons. The UI distinguishes provider status, recorded data, and demonstration content. Both the scene and default PCA replay work without WebGL.

## Run the lab

Install Python 3.11 or newer, then **double-click `run.bat` on Windows**. On macOS/Linux, run `bash run.sh`.

The launcher creates the project environment, installs the lightweight UI dependencies, and opens [the lab](http://127.0.0.1:8501). Later launches reuse that environment.

1. Paste a Hugging Face model link and choose **Check and add model**.
2. Choose the Anthropic-inspired research pilot or course matrix, or upload your own train/test CSV. Download the selected example to edit it.
3. Enter your Kaggle username and API token, then choose **Run experiment**.
4. Open **Results & steering** for neutral-subtraction playback, color-coded emotion gauges, generated text, and reusable-vector inference controls.

The research pilot includes twelve training pairs and eight test/control prompts for happy, calm and desperate; see [the pilot guide](docs/ANTHROPIC_PILOT.md). The course example retains the original eighteen cases. Each account runs its own private Kaggle GPU jobs. See [the integrated UI guide](docs/KAGGLE_UI.md).

Run your own experiments to discover, inspect, and steer emotion-related activation vectors. Includes a local Streamlit workbench, a self-contained Kaggle notebook, sequential multi-model inference, native vector checks, signed coefficient sweeps with zero controls, repeated sampling seeds, blinded annotation export, and the CMPE 187 testing/report pipeline.

**Start with [the quickstart](docs/QUICKSTART.md).** Each researcher supplies their own Kaggle credentials in the integrated UI. Gated checkpoints additionally require an attached Kaggle `HF_TOKEN` secret; public Qwen weights need no model token. Internal-vector steering uses downloadable model weights; hosted text API keys alone do not provide activation access.

**Verification:** CPU validation and report artifacts are provided in `evidence/`. Real Kaggle T4 testing completed for Qwen 0.5B FP16, Qwen 1.5B FP32, and SmolLM2 135M FP32: 54 course cases and 18 selected controlled steering generations. FP16 1.5B activation overflow was detected; that failed run is retained. Course thresholds remain unapproved and ungraded. The matched Qwen 0.5B FP16/NF4 experiment and second-family pilot passed. IBM MIA and GTX 1080 Ti execution remain unverified.

The full source ZIP is built with `python scripts/build_release.py`. Download/import `notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb` on Kaggle, enable T4 and Internet, attach your `HF_TOKEN` secret if required, then use the interactive controls. Model loading requires a button click. For local GTX 1080 Ti setup, use the scripts documented in the quickstart.

This is a standalone research extension. The original Model Mountain v4 source has not been merged. Activation correlations are exploratory evidence, not claims that models experience emotions.

## Historical bundle documentation

The notes below describe earlier versions. The quickstart, progress log, and evidence directory describe the current release.

# Model Mountain — Emotion Vector Research Suite (v0.2 Workbench)

**NEW: Interactive test and discovery workbench** — editable prompt-pair table / CSV-JSONL import, scan decoder layers, save and browse vectors, adjust sign/strength/scope, A/B baseline and intervention, dose-response sweep, per-run JSON export. See [`docs/WORKBENCH_OPERATOR_GUIDE.md`](docs/WORKBENCH_OPERATOR_GUIDE.md).

**Visual local UI:** `python -m pip install -r requirements-hf.txt -r requirements-ui.txt` then `python -m streamlit run app_emotion.py` (install compatible GPU PyTorch beforehand). **Kaggle T4:** set `OPEN_INTERACTIVE_WORKBENCH=True` in the notebook, run the interactive panel cell; it uses Jupyter widgets directly.

**Important:** CPU tests and notebook reconstruction passed; GPU model inference and IBM MIA hooks have not been run here. Output scores are training-only for manual quick fits; full CLI has held-out splits.

---

# Model Mountain — Emotion Vector Research Suite (v0.1)

**Status (2026-10-28): source code, dataset, notebook and CPU tests prepared. Real GPU model captures and IBM MIA execution have NOT been run or verified in this environment.**

Use this suite to study whether emotion-related directions in transformer decoder activations persist when a model's weights are quantized, and whether activation interventions cause reproducible behavioral differences. It is a separate **research extension** to Model Mountain v4, not a bypass of Model Mountain's prompt/RAG/agent/MCP/extraction security enforcement.

## Start here

- **Coding agents:** read `AGENTS.md` then `docs/AGENT_DATASHEET.md`; no exploratory repository browsing needed to start.
- **Kaggle, single upload:** import `notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb`. It restores the complete suite embedded within the notebook.
- **GTX 1080 Ti:** use Hugging Face Transformers backend (FP16 + bitsandbytes NF4), NOT IBM MIA.
- **Kaggle T4:** use the same Transformers backend first. IBM vLLM-Hook MIA is an optional, separate, unverified environment.

## Quick local smoke test (no model download)

```bash
python -m pip install -r requirements-core.txt
python -m pytest -q
python -m mm_emotion.cli doctor
python -m mm_emotion.cli validate
python -m mm_emotion.cli smoke --out runs/synthetic_smoke.json
```

**The smoke JSON is synthetic and MUST NOT be reported as experimental model evidence.**

## Real GPU experiment on Kaggle T4 or compatible local GPU

Install `torch` from the official PyTorch wheel index matching your driver and GPU architecture before installing HF requirements. On GTX 1080 Ti, use a CUDA 12.x PyTorch binary retaining Pascal (sm_61) support; not CUDA 13 builds. Convenience setup: Windows PowerShell `scripts/setup_gtx1080ti.ps1`, or Linux/WSL2 `bash scripts/setup_gtx1080ti.sh`; these install `torch==2.14.1` from the official CUDA 12.6 wheel index. Then:

```bash
python -m pip install -r requirements-hf.txt
# Small pilot: two pairs per emotion per split
python -m mm_emotion.cli run --backend hf --model Qwen/Qwen2-1.5B-Instruct \
  --precision fp16 --data data/emotion_pairs_v1.jsonl \
  --layers 5,11,17,23 --limit-per-split 2 --generate-samples 2 \
  --out runs/qwen1p5b_fp16_pilot

python -m mm_emotion.cli run --backend hf --model Qwen/Qwen2-1.5B-Instruct \
  --precision nf4 --data data/emotion_pairs_v1.jsonl \
  --layers 5,11,17,23 --limit-per-split 2 --generate-samples 2 \
  --out runs/qwen1p5b_nf4_pilot

python -m mm_emotion.cli compare --run-a runs/qwen1p5b_fp16_pilot \
  --run-b runs/qwen1p5b_nf4_pilot --out runs/pilot_comparison.json
```

Then rerun both precisions with `--limit-per-split 0` in NEW `runs/` directories for the 48-pair pilot corpus (the full pilot is still not a sufficiently powered publication-quality dataset). `--generate-samples 2` records deterministic baseline and steered generations for human review; the code does not claim an automatic emotion-scoring evaluator.

## Genuine IBM vLLM-Hook MIA on Kaggle (experimental; NOT locally validated)

Use NVIDIA T4 (compute capability 7.5), CUDA 13-compatible driver, isolated environment. The install script is `scripts/install_mia_kaggle.sh`; exact MIA worker API tested against current public IBM documentation, not against actual GPU execution here. Run from suite root after installation:

```bash
# This uses a separate capture worker; requires genuine MIA installed in the environment.
python -m mm_emotion.cli run --backend mia --model Qwen/Qwen2-1.5B-Instruct \
  --precision fp16 --mia-config configs/mia_capture_qwen2_1p5b.json \
  --layers 5,11,17,23 --limit-per-split 2 --out runs/mia_fp16_pilot

# Different worker / separate process after capture; uses MIA steering config.
python -m mm_emotion.cli mia-steer --run runs/mia_fp16_pilot \
  --emotion anxiety --out runs/mia_steering_anxiety
```

**MIA is NOT required** to carry out FP16 versus NF4 representation experiments. Do not compare MIA and HF activations as interchangeable until verifying identical prompt tokenization, block hook positions, capture phase and numerical conventions.

## Research outputs

Each real run writes:

- `manifest.json`: model ID, precision, backend, runtime, data checksum, selected layers, run state
- `captures/<emotion>_<split>.npz` and `.json`: real last-token block activations + SHA256 provenance
- `vectors/<emotion>.npz`, `.json`, `_mia.pt`: fitted layer vectors, validation-selected layer, MIA export
- `summary.json` / `summary.csv`: held-out **synthetic-prompt** metrics, never clinical/psychological scores
- `generations/<emotion>.jsonl`: optional baseline/steered HF results
- `comparison.json`: direction cosines between matched model-and-dataset runs

`python -m mm_emotion.cli export-evidence --run runs/... --out runs/.../model_mountain_evidence.json` emits a read-only Model Mountain research evidence record. It explicitly cannot authorize an agent action or alter a security policy decision.

## What this experiment DOES and DOES NOT establish

**Does:** collect actual per-layer activations from controlled, matched prompt pairs; build normalized mean-difference (or exploratory pairwise-PCA) directions; select intervention layers only on validation; report held-out contrastive margins; compare directions across weight precision; produce output samples for blinded human annotation.

**Does not:** prove the model has emotions; isolate emotion from all lexical or task-style confounds; validate universal transferability; supply population or statistical confidence with only 3 test scenarios per emotion; guarantee IBM MIA on Kaggle; deliver real GPU results from this archive.

## Project tree

```text
mm_emotion/            Data validation, capture backends, vector fitting, steering and evidence
configs/               Layer IDs and documented MIA hidden-state capture settings
data/                  Synthetic matched prompt pairs with leakage-safe splits
tests/                 CPU unit/integration tests (no fake GPU claims)
notebooks/             Self-contained Kaggle notebook, all source embedded
scripts/               Dataset generator, optional MIA install, overlay helper
integrations/          Model Mountain v4 research-only adapter guidance
app_emotion.py         Optional read-only Streamlit research dashboard
docs/AGENT_DATASHEET.md Full agent handoff, next steps, known blockers, contracts
AGENTS.md              Minimal operational starting instructions
```

## Public technical references

- IBM vLLM-Hook: https://github.com/IBM/vLLM-Hook
- MIA configuration and layer numbering: https://github.com/IBM/vLLM-Hook/blob/main/docs/configs.md
- bitsandbytes hardware matrix: https://huggingface.co/docs/bitsandbytes/main/en/installation
- NVIDIA legacy GPU compute capabilities: https://developer.nvidia.com/cuda/gpus/legacy
- Qwen2 1.5B Instruct: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct

## Blinded human annotation and reports

```bash
python scripts/blind_hf_generations.py --run runs/fp16_full --out runs/fp16_blind
# Annotators complete the blank scores in blinded_annotations.csv without seeing condition key.
python scripts/analyze_annotations.py --ratings runs/fp16_blind/blinded_annotations.csv \
  --key runs/fp16_blind/blinding_key_PRIVATE.json --out runs/fp16_blind/rating_summary.json
python scripts/render_report.py --run runs/fp16_full
```

Alternative exploratory vector extraction: append `--vector-method pairwise_pca` to both `run` commands, using new directories; comparisons require the same method. No automatic mood classifier or psychological ground truth is included.

**Pascal wheel provenance:** https://pytorch.org/get-started/previous-versions/ lists PyTorch 2.14.1 CUDA 12.6; https://github.com/pytorch/pytorch/issues/190385 documents why CUDA 12.6 is the last supported pre-Turing line in the proposed PyTorch 2.15 matrix. Verify the actual installed wheel with `doctor` and a real CUDA capture on your own machine.

## v0.3: Multi-model inference + CMPE 187 testing

- `python -m mm_emotion.multi_model --models qwen05,qwen15 --mode infer --out runs/multi.json`: run direct prompts through multiple model checkpoints in sequence.
- `python -m mm_emotion.multi_model --models qwen05 --mode matrix --out runs/cmpe187.json`: run the 18-case rubric matrix against genuine hidden-state captures, using separate training prompts.
- `python -m mm_emotion.course_report --results runs/cmpe187.json --out runs/CMPE187_Results.docx`: generate a rubric-aligned editable report.
- `python -m mm_emotion.drawio course/drawio`: create native editable classification, context, event and decision-table diagrams.
- Local UI pages: `pages/1_Multi_Model_Experiments.py` and `pages/2_CMPE187_Report_Builder.py` (launch `app_emotion.py`). Kaggle widgets: `OPEN_MULTI_MODEL_PANEL=True` in the single-upload notebook.
- Work handoff: `work_handoff/WORK_PROMPT_COPY_PASTE.md` and `work_handoff/CONTEXT_INDEX.md`. Read only what each task needs to save context/tokens.

**Provenance:** This extends Workbench v2 source, not the unavailable original Model Mountain v4 archive. All actual GPU inference and MIA behavior require external verification. 18-case thresholds from the original Deliverable #2A are **unapproved** because several original interval inequalities are reversed. No real-world model test evidence is embedded.
