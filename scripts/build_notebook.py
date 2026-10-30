"""Build a SINGLE-UPLOAD Kaggle notebook embedding the entire code/data suite.

The notebook can reconstruct itself's companion source; no ZIP attachment required.
"""
from __future__ import annotations
import base64
import io
import zipfile
from pathlib import Path
import nbformat as nbf
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mm_emotion.packaging import source_files
NOTEBOOK = ROOT / "notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".venv-mia", "external", "runs", ".git", ".ipynb_checkpoints", "generated", "diagram_previews", "reference"}
EXCLUDE_FILES = {"MIA_COMMIT_PIN.txt", "Model_Mountain_Emotion_Lab.zip", NOTEBOOK.name}


def build_zip():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(source_files(ROOT)):
            info = zipfile.ZipInfo((Path('Model_Mountain_Emotion_Lab') / path.relative_to(ROOT)).as_posix(), (2026,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    return memory.getvalue()


def code(src):
    return nbf.v4.new_code_cell(src)


def main():
    payload = base64.b64encode(build_zip()).decode("ascii")
    chunks = [payload[i:i+100] for i in range(0,len(payload),100)]
    notebook = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python"},
                                             "model_mountain_status": "CPU verified; measured Kaggle T4 evidence in release; IBM MIA unverified"})
    notebook.cells = [
        nbf.v4.new_markdown_cell('''# Model Mountain — Emotion Vector Research (Kaggle single-upload edition)

**Includes all source code, synthetic data, tests, agent handoff, and optional MIA adapter embedded inside this notebook.** No additional ZIP needed. Import this `.ipynb` into Kaggle → Settings → Accelerator → GPU T4 ×2 for real experiments; Internet must be on for model/packages. (Two T4s do not pool VRAM.)

**Integrity notice:** offline smoke tests use simulated vectors and are **not** real emotion evidence. `RUN_REAL_HF`, `RUN_NF4` and `RUN_MIA` are opt-in. Separate measured T4 experiments are documented in the release's `evidence/` directory; importing this notebook does not reproduce them automatically. MIA and GTX 1080 Ti execution remain unverified. On a GTX 1080 Ti use the local HF workflow.

For the integrated model-link, CSV, emotion-gauge and mixed-steering UI, run `run.bat` (Windows) or `bash run.sh` (macOS/Linux) from the GitHub project. It uses your Kaggle API account for private GPU jobs; no local GPU is required.

For multi-model direct inference and CMPE 187 classification tests, set `OPEN_MULTI_MODEL_PANEL = True`; for an opt-in GPU pilot set `RUN_MULTI_MODEL_PILOT = True`. Both panels default to True; the automatic GPU pilot defaults to False. See `course/` and `work_handoff/` after extraction.

For interactive vector testing (no CLI), set `OPEN_INTERACTIVE_WORKBENCH = True`: a native Jupyter interface lets you add/import prompt pairs, choose model/precision, scan layers, and compare steered vs unsteered generations. The model is not loaded until you click Load model.

Agent entrypoint after restoration: `AGENTS.md` → `docs/AGENT_DATASHEET.md` → `docs/AGENT_HANDOFF.json`.'''),
        code('''# Control switches; default execution is safe and CPU-only.
RUN_REAL_HF = False          # True: actual FP16 Qwen GPU forward hooks and limited steering
RUN_NF4 = False              # True: also run identical-model NF4 weight quantization
RUN_MIA = False              # True: separate experimental IBM MIA environment, only after HF works
OPEN_INTERACTIVE_WORKBENCH = True    # Panels open by default; model loads only after a button click
OPEN_MULTI_MODEL_PANEL = True        # Native direct inference and CMPE 18-case matrix UI
RUN_MULTI_MODEL_PILOT = False         # True: GPU Qwen 0.5B sequential inference + matrix
PILOT = True                 # True: 2 pairs/split/emotion; False: all 48 pairs
GENERATE_SAMPLES = 2         # 0 disables intervention generations
COEFFICIENTS = "0,3,6"       # matched causal dose-response levels
MODEL = "Qwen/Qwen2-1.5B-Instruct"
LAYERS = "5,11,17,23"        # 0-based block IDs
import os, sys, time, subprocess, json, shutil
from pathlib import Path
RUN_ID = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
print("Run ID", RUN_ID, "selected real HF:", RUN_REAL_HF, "NF4:", RUN_NF4, "MIA:", RUN_MIA)'''),
        code('''# Reconstruct the complete project from this notebook's embedded ZIP.
import base64, zipfile, io, os
from pathlib import Path
_PACKED_BASE64 = (
''' + "\n".join('    "' + x + '"' for x in chunks) + '''
)
base = Path(os.environ.get("MM_EMOTION_NOTEBOOK_WORKDIR", "/kaggle/working" if Path("/kaggle/working").exists() else "/mnt/data"))
base.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_PACKED_BASE64))) as package:
    for name in package.namelist():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise RuntimeError(f"Unsafe archive member: {name}")
    package.extractall(base)
PROJECT = base / "Model_Mountain_Emotion_Lab"
os.chdir(PROJECT)
print("Restored project at", PROJECT)
print("Handoff:", PROJECT / "docs/AGENT_DATASHEET.md")'''),
        nbf.v4.new_markdown_cell('''## Your credentials
For gated models, add a Kaggle Secret named `HF_TOKEN` and grant this notebook access.
Public Qwen checkpoints need no token. Tokens are read at runtime and never embedded.
A Kaggle API token is needed only to upload/download notebooks through the local CLI;
it is not needed to execute this notebook on Kaggle. See `docs/QUICKSTART.md`.
Hosted model APIs do not expose decoder activations; this lab runs downloadable weights.'''),
        code('''# Read private credentials without displaying their values.
import sys
sys.path.insert(0, str(PROJECT))
from mm_emotion.credentials import load_kaggle_model_secret
print("Hugging Face credential available:", load_kaggle_model_secret())'''),
        code('''# CPU validation: genuine code/dataset checks, plus explicitly SYNTHETIC numerical smoke.
def run_command(args, *, python=sys.executable, extra_env=None):
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    if extra_env:
        env.update(extra_env)
    print("COMMAND:", python, *args)
    subprocess.run([python, *args], cwd=str(PROJECT), env=env, check=True)

run_command(["-m", "pip", "install", "-r", "requirements-core.txt"])
run_command(["-m", "pytest", "-q"])
run_command(["-m", "mm_emotion.cli", "validate"])
run_command(["-m", "mm_emotion.cli", "smoke", "--out", "runs/kaggle_synthetic_smoke.json"])
print("CPU-only tests are not evidence of real model activations")'''),
        code('''# GPU hardware and package inventory. Do NOT assume this notebook has a GPU.
run_command(["-m", "mm_emotion.cli", "doctor"])
if RUN_REAL_HF or RUN_MIA:
    import torch
    assert torch.cuda.is_available(), "Kaggle GPU not enabled; use Settings -> Accelerator -> GPU T4"
    print("Selected GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))'''),
        code('''# Optional real multi-model direct inference and CMPE187 rubric matrix.
# Real GPU calls are opt-in; never confuse default CPU smoke tests with actual evidence.
if RUN_MULTI_MODEL_PILOT:
    run_command(["-m", "pip", "install", "-r", "requirements-hf.txt"])
    run_command(["scripts/gpu_acceptance.py", "--models", "qwen05,qwen15",
                 "--out", f"runs/kaggle_{RUN_ID}_acceptance"])
else:
    print("Optional GPU multi-model pilot disabled")'''),
        code('''# REAL HF FP16 / NF4 capture-and-steering pilot; always separate subprocesses.
# Kaggle's existing PyTorch is used; install Hugging Face stack, not the MIA venv here.
HF_RUNS = {}
if RUN_REAL_HF:
    run_command(["-m", "pip", "install", "-r", "requirements-hf.txt"])
    suffix = "pilot" if PILOT else "full"
    limit = "2" if PILOT else "0"
    for precision in (["fp16", "nf4"] if RUN_NF4 else ["fp16"]):
        out = f"runs/kaggle_{RUN_ID}_{suffix}_{precision}"
        run_command(["-m", "mm_emotion.cli", "run", "--backend", "hf",
                     "--model", MODEL, "--precision", precision, "--layers", LAYERS,
                     "--limit-per-split", limit, "--generate-samples", str(GENERATE_SAMPLES),
                     "--coefficients", COEFFICIENTS, "--out", out])
        HF_RUNS[precision] = out
        run_command(["scripts/render_report.py", "--run", out])
    if "nf4" in HF_RUNS:
        run_command(["-m", "mm_emotion.cli", "compare", "--run-a", HF_RUNS["fp16"],
                     "--run-b", HF_RUNS["nf4"], "--out", f"runs/kaggle_{RUN_ID}_{suffix}_comparison.json"])
        print("Saved controlled weight-precision comparison")
else:
    print("HF experiments disabled; set RUN_REAL_HF=True after verifying GPU and Internet")'''),
        code('''# OPTIONAL: MIA is an independent CUDA13-compatible environment and separate vLLM worker.
# Run only after confirmed HF evidence. Installs IBM dependencies in .venv-mia.
MIA_RUN = None
if RUN_MIA:
    import torch
    assert tuple(torch.cuda.get_device_capability(0)) >= (7, 5), "IBM MIA requires Turing or newer"
    subprocess.run(["bash", "scripts/install_mia_kaggle.sh"], cwd=str(PROJECT), check=True)
    mia_python = str(PROJECT / ".venv-mia/bin/python")
    MIA_RUN = f"runs/kaggle_{RUN_ID}_mia_capture"
    run_command(["-m", "mm_emotion.cli", "run", "--backend", "mia", "--model", MODEL,
                 "--precision", "fp16", "--layers", LAYERS,
                 "--mia-config", "configs/mia_capture_qwen2_1p5b.json",
                 "--limit-per-split", "2", "--out", MIA_RUN], python=mia_python,
                extra_env={"PYTHONPATH": str(PROJECT)})
    run_command(["-m", "mm_emotion.cli", "mia-steer", "--run", MIA_RUN,
                 "--emotion", "anxiety", "--out", f"runs/kaggle_{RUN_ID}_mia_steering_anxiety"],
                python=mia_python, extra_env={"PYTHONPATH": str(PROJECT)})
else:
    print("IBM MIA not attempted; not a failure of the HF research suite")'''),
        code('''# OPTIONAL visual controls: check the switch in the first configuration cell.
# Buttons and sliders work inside Kaggle/Jupyter; no Streamlit server or port forwarding.
if OPEN_INTERACTIVE_WORKBENCH:
    import importlib.util
    if importlib.util.find_spec("ipywidgets") is None:
        run_command(["-m", "pip", "install", "ipywidgets>=8,<9"])
    # Ensure both FP16 and NF4 support; Kaggle images may already have Transformers but lack bitsandbytes.
    run_command(["-m", "pip", "install", "-r", "requirements-hf.txt"])
    from mm_emotion.notebook_ui import display_workbench
    ui = display_workbench(PROJECT)
    print("Click the tabs in the control panel; experiment files go to runs/workbench_sessions/")
else:
    print("For visual A/B steering controls, set OPEN_INTERACTIVE_WORKBENCH=True in cell 2 and rerun this cell")'''),
        code('''# Kaggle-native direct model inference and CMPE classification test controls.
if OPEN_MULTI_MODEL_PANEL:
    run_command(["-m", "pip", "install", "-r", "requirements-hf.txt"])
    import importlib.util
    if importlib.util.find_spec("ipywidgets") is None:
        run_command(["-m", "pip", "install", "ipywidgets>=8,<9"])
    from mm_emotion.course_notebook_ui import display_course_panel
    cmpe_ui = display_course_panel(PROJECT, before_run=getattr(globals().get("ui"), "_release_model", None))
else:
    print("Set OPEN_MULTI_MODEL_PANEL=True to open the multi-model research panel")'''),
        code('''# Export recorded results. Download from Kaggle's notebook Output tab.
from pathlib import Path
results_dir = PROJECT / "runs"
result_files = sorted(str(p.relative_to(PROJECT)) for p in results_dir.rglob("*") if p.is_file())
print("Result files:", *result_files[:100], sep="\\n  ")
archive_path = base / f"Model_Mountain_Emotion_Results_{RUN_ID}.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in results_dir.rglob("*"):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(PROJECT)))
print("Export bundle:", archive_path)
print("WARNING: If blinded annotation private keys exist, keep the export private until scoring is locked.")''')
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(f"Built {NOTEBOOK} ({NOTEBOOK.stat().st_size:,} bytes) with embedded {len(payload):,} base64 chars")


if __name__ == "__main__":
    main()
