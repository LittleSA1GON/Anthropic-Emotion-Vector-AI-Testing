# v0.3 WORK ENTRYPOINT — READ THIS FIRST

**This is the current active entrypoint.** For ChatGPT Work, use `work_handoff/WORK_PROMPT_COPY_PASTE.md`. Next read `work_handoff/TASK_QUEUE.json`, then `work_handoff/CONTEXT_INDEX.md`. Do NOT survey the whole repository or duplicate implementation. Use `python -m pytest -q` for CPU validation; optional Kaggle notebook uses `OPEN_MULTI_MODEL_PANEL=True` for direct multi-model inference and the CMPE187 matrix. See `course/QUICK_START.md`. Research-only, no GPU evidence generated here.

**Core change:** The CMPE187 18-case matrix has provisional cosine thresholds due to inverted inequalities in the supplied Deliverable #2A PDF. Never silently grade them. Do not treat the current extension as directly merged into Model Mountain v4.

---

# IMMEDIATE START FOR ANY CODING AGENT (v0.2)

**First read** `docs/WORKBENCH_OPERATOR_GUIDE.md`, then `docs/AGENT_DATASHEET.md` for full experiments. User-facing GUI: `app_emotion.py`. Notebook-native Kaggle widgets: `mm_emotion/notebook_ui.py`. Input/fit/save/guardrails: `mm_emotion/workbench.py`. Intervention hook controls: `mm_emotion/hf_backend.py:HFBackend.generate`. Build embedded notebook: `python scripts/build_notebook.py`.

**Do next**: run `python -m pytest -q`; on hardware run the smallest actual FP16 prompt pair, then a negative/positive/zero sweep at one layer, document the outputs; on Kaggle start with notebook `OPEN_INTERACTIVE_WORKBENCH=True`. Never call synthetic outputs real evidence or in-sample scores held-out performance. Do not merge experiment output into security authorization.

# AGENT ENTRYPOINT — MODEL MOUNTAIN EMOTION LAB

**Read `docs/AGENT_DATASHEET.md` next.** It is the canonical handoff: task, repository paths, verified status, commands, acceptance criteria, GPU decision tree, and remaining work. Do not spend time re-discovering the project.

Current state 2026-10-28: files created; dataset generated; CPU unit tests pass; CPU-only synthetic smoke passes; NO real GPU inference; NO MIA validation; NO empirical emotion behavior findings.

**Immediate objective:** run a real Qwen2 1.5B FP16 vs NF4 pilot on a Kaggle T4 with the `hf` backend, validate artifact provenance, and then expand. Test optional MIA only afterward.

**Constraints:** Never claim synthetic = real evidence. Never use an uncalibrated emotion score as a Model Mountain security authorization signal. Do not silently modify the existing Model Mountain v4 gateway. Preserve train/validation/test leakage prevention and match model IDs/dataset hashes for quantization comparisons.

**First commands:** `python -m pytest -q && python -m mm_emotion.cli validate && python -m mm_emotion.cli doctor`.

**Archive limitation:** Model Mountain v4 (`Model_Mountain_v4_Multiagent_Workflow_COMPLETE.zip`) was discoverable in prior-project Library but could not be materialized to inspect its source bytes in the authoring environment. This project is a **standalone compatible research extension**, not a verified patch into that archive. A future agent with the original archive must inspect v4's module signatures before creating a direct import bridge.
