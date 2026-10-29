# Course / multi-model quick start

1. Run `python -m pytest -q` to verify the project; no GPU required.
2. Run `python -m mm_emotion.drawio course/drawio` to rebuild all editable draw.io diagrams.
3. Run `python -m mm_emotion.course_report --out course/generated/CMPE187_Deliverable_2_DRAFT.docx` to generate the rubric-aligned report without inventing results.
4. To use a local Streamlit interface: `python -m pip install -r requirements-ui.txt -r requirements-hf.txt python-docx pillow`; `python -m streamlit run app_emotion.py`. Sidebar: Multi Model Experiments / CMPE187 Report Builder.
5. For direct real GPU inference: `python -m mm_emotion.multi_model --models qwen05,qwen15 --mode infer --out runs/real_multi_inference.json`. Models load **sequentially**.
6. For the course matrix: `python -m mm_emotion.multi_model --models qwen05 --mode matrix --cases course/18_test_cases.csv --layer 5 --out runs/real_course_matrix.json` (actual GPU; 18 cases; provisional UNGRADED default).
7. For a results-backed report: `python -m mm_emotion.course_report --results runs/real_course_matrix.json --out runs/CMPE187_Deliverable_2_Results.docx`.
8. For vector sweeps across multiple models, first create a **separate** vector NPZ for each model/precision using the existing workbench and put their paths in `course/native_vectors.example.json` (edit it). Run `python -m mm_emotion.multi_model --models qwen05,qwen15 --mode sweep --native-vectors course/native_vectors.example.json --coefficients=-6,-3,0,3,6 --layer 5 --out runs/real_sweep.json`.
9. For Kaggle: import `notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb`, enable GPU T4 (internet on), set `OPEN_MULTI_MODEL_PANEL=True` for notebook widgets. Use `RUN_MULTI_MODEL_PILOT=True` for an opt-in scripted GPU smoke.

**Requirements:** Supported CUDA/PyTorch build + HF stack for actual inference. GTX 1080 Ti cannot run the current IBM vLLM-MIA engine; Kaggle T4 can attempt the MIA path separately but this remains unverified. GPU runtimes are not bundled. Model checkpoints are fetched from authorized public model registries at execution time. Model-specific gates/licenses may apply.

**Never confuse course outputs:** No GPU => DOCX says Not Run. Unapproved cosine thresholds => UNGRADED rather than fake passes. A raw prompt mentioning an emotion is not proof the model internally represents it.
