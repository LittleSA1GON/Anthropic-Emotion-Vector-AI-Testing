# Read-by-task index — avoid context and token waste

| Task | Only open these files initially |
|---|---|
| Project map | `AGENTS.md`, `docs/STATUS.md`, `work_handoff/TASK_QUEUE.json` |
| Existing vector discovery and steering | `mm_emotion/workbench.py`, `mm_emotion/hf_backend.py`, `app_emotion.py` |
| Multi-model direct inference | `mm_emotion/multi_model.py`, `pages/1_Multi_Model_Experiments.py`, `course/model_registry.example.json` |
| Kaggle multi-model | `mm_emotion/course_notebook_ui.py`, `scripts/build_notebook.py` |
| 18-case classifier, data | `mm_emotion/course_tests.py`, `course/18_test_cases.csv`, `course/training_pairs.json` |
| Draw.io test models | `mm_emotion/drawio.py`, `course/drawio/CMPE187_All_Editable_DrawIO.drawio` |
| Report + evidence | `mm_emotion/course_report.py`, `pages/2_CMPE187_Report_Builder.py`, `course/generated/CMPE187_Deliverable_2_DRAFT.docx` |
| Existing IBM MIA | `mm_emotion/mia_backend.py`, `mm_emotion/mia_steering.py`, `configs/mia_capture_qwen2_1p5b.json` |
| Model Mountain v4 bridge | `mm_emotion/model_mountain_bridge.py`, `integrations/README.md` |
| Existing validation | `tests/test_course.py`, `tests/test_multi_model.py`, plus nearest relevant test file |

**Do not read all source files into context at once.** Search local source by symbol (`rg 'symbol' mm_emotion`) and open only required functions or lines. Reuse existing APIs and record changed paths in progress log. Default: no external web calls unless dependency compatibility requires verification.

## Work-specific decisions already made
- Current files come from `Model_Mountain_Emotion_Lab_WORKBENCH_v2_COMPLETE.zip` and this update, not directly from the latest v4 archive (the latter's raw bytes were inaccessible).
- Course test classification is exactly happiness/sadness/anger × objective/stated/enacted × standard/expressive = 18 primary cases; intensity is a derived tag, not a fourth independent crossed factor.
- The source PDF's cosine bounds are traceable via `source_range_text`; `proposed_min`/`proposed_max` are provisional and ungraded by default.
- This is model-internal interpretability research, not a claim of subjective feelings in language models and not a security gate.
- GTX 1080 Ti: HF hooks and NF4 candidate, no current vLLM-MIA; Kaggle T4: HF first, MIA compatibility experimental.
- Multi-model GPU runs use sequential model loading to prevent VRAM accumulation.
