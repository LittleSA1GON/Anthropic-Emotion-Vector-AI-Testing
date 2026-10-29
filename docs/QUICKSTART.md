# Run your own Emotion Lab

Each researcher runs a separate local app or a private Kaggle notebook with their own credentials and output directory. Access to this GitHub repository does not grant Kaggle GPU quota or access to gated model weights.

## Kaggle T4

1. Download `notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb` and import it into Kaggle Notebooks. The source is embedded; no ZIP upload is needed.
2. Enable Internet and a T4 accelerator in Notebook settings. Models run sequentially on one GPU; two T4 cards do not combine memory.
3. Public Qwen checkpoints need no key. For gated weights, accept the model license on Hugging Face, create a read token, add a Kaggle Secret named `HF_TOKEN`, and attach it to your notebook. Never paste a token into a cell.
4. Run the cells. The workbench and multi-model panels open by default; real weights load only when you click a model action. The automatic GPU pilot remains opt-in.
5. In the workbench, load a small model, import matched pairs, scan layers, select the saved native vector, and compare negative / zero / positive strengths on new prompts. Inspect the activation projection and generated text separately.
6. Save notebook outputs after experiments. The final cell exports the current `runs` directory. Run it again after using panel buttons; earlier ZIPs do not update automatically.

## Local Windows or Linux

Use Python 3.12. In a terminal at the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -r requirements-hf.txt -r requirements-ui.txt
```

Install a PyTorch build compatible with your GPU separately. For GTX 1080 Ti, the setup scripts install a CUDA 12.6 build in `.venv-gtx`; run `scripts/setup_gtx1080ti.ps1` or `bash scripts/setup_gtx1080ti.sh`. They check CUDA using a real tensor operation before claiming readiness. Driver/build compatibility still needs testing on your hardware. Start with Qwen 0.5B FP16 and layer 5. The `qwen15` alias uses FP32 because real T4 testing exposed FP16 activation overflow for Qwen2.5 1.5B; the failure evidence is retained. NF4 is an optional follow-up; IBM MIA remains experimental and is not the Pascal workflow.

```bash
python -m mm_emotion.cli doctor
python -m streamlit run app_emotion.py --server.address 127.0.0.1
```

Paste your optional Hugging Face token into **Your model access** in the sidebar. It stays in the current Streamlit session. CLI users set `HF_TOKEN` in their own shell. `.env.example` documents variable names but is not automatically loaded. Unload the home-page model before running a multi-model batch to free GPU memory. Use a separate local process for each researcher; this app is not a multi-tenant hosted service.

## Kaggle API credentials

The Kaggle API key authenticates notebook uploads/downloads, not model inference. Running on the Kaggle website does not require putting this key in the notebook. To upload from a local terminal:

```bash
python -m pip install kaggle
kaggle auth login
kaggle kernels init -p notebooks
```

Edit the generated `notebooks/kernel-metadata.json` with your own `username/slug`, title, `code_file` set to `Model_Mountain_Emotion_Lab_Kaggle.ipynb`, Python notebook type, Internet and GPU enabled, and `is_private: true`. Then:

```bash
kaggle kernels push -p notebooks --no-run
```

Alternatively configure your own `KAGGLE_API_TOKEN` or user-home Kaggle credentials according to the [official authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md). Do not commit personal kernel metadata or credentials. Uploading to Kaggle and spending GPU quota are researcher actions; this release does not upload or run notebooks under someone else's account. [Official kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

## Reproducible experiments

```bash
# Direct inference, sequential checkpoints and repeated sampling seeds
python -m mm_emotion.multi_model --models qwen05,qwen15 --mode infer --temperature 0.7 --seeds 42,43,44 --out runs/direct.json
# Real vector discovery: train / validation / test remain separate
python -m mm_emotion.cli run --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16 --layers 5,11,17 --limit-per-split 2 --generate-samples 2 --coefficients=-3,0,3 --out runs/qwen05
```

Create `runs/native_vectors.json` as `{"qwen05":"runs/qwen05/vectors/joy.npz"}`. Each model needs a vector fitted to its exact checkpoint and precision.

```bash
python -m mm_emotion.multi_model --models qwen05 --mode sweep --native-vectors runs/native_vectors.json --layer 5 --coefficients=-3,0,3 --temperature 0.7 --seeds 42,43,44 --out runs/sweep.json
python -m mm_emotion.blinding --run runs/sweep.json --out runs/blind
python -m mm_emotion.multi_model --models qwen05,qwen15 --mode matrix --layer 5 --continue-on-error --out runs/matrix.json
python -m mm_emotion.course_report --results runs/matrix.json --out runs/CMPE187_Results.docx
```

Give annotators `annotations.csv`; retain `condition_key_PRIVATE.json` until ratings are locked. At temperature zero, repeating seeds typically produces identical output and does not estimate sampling variability. Zero is always included in strength sweeps. Never fit vectors on evaluation prompts or silently reuse vectors from another checkpoint. Hugging Face tokens authorize weight downloads; Anthropic/OpenAI text APIs do not expose the decoder hooks used here.

The 18-case matrix preserves happiness/sadness/anger × objective/stated/enacted × standard/expressive. Original reversed cosine strings remain recorded. All thresholds stay ungraded unless explicitly approved; neutral-label semantics also need instructor review.

## Validation and release

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m mm_emotion.cli validate
python -m mm_emotion.drawio course/drawio
python -m mm_emotion.course_report --results evidence/course-matrix.json
python scripts/release_manifest.py
python scripts/build_notebook.py
python scripts/build_release.py
```

Use LibreOffice and the provided document-rendering environment to render DOCX to PDF. `python scripts/export_report_pdf.py INPUT.docx OUTPUT.pdf` provides a portable content-equivalent PDF when LibreOffice is unavailable; its pagination differs from Word. CPU plumbing tests and generated templates are not real model findings. See `evidence/validation.json` and `work_handoff/PROGRESS.json` for this release's actual verification status.

Release evidence: real Kaggle T4 runs completed for three checkpoints. The 54 course cases remain ungraded. Matched Qwen 0.5B FP16/NF4 direction cosines ranged from 0.7704 to 0.8062 at block 5; these exploratory representation comparisons do not establish preserved behavior.

The owner account also has a [private saved notebook](https://www.kaggle.com/code/littlesa1gon/model-mountain-emotion-lab). Collaborators can import the notebook from this repository into their own account and attach their own secrets. To inspect shipped vectors locally, copy a paired `.npz` and `.json` from `evidence/gpu-v2/.../vectors` into `runs/workbench_vectors`; keep each checkpoint and precision matched.
