# Deliver this ZIP to ChatGPT Work

1. Attach `Model_Mountain_Emotion_Lab_WORK_READY_v3_FULL.zip` to a new ChatGPT Work task.
2. Copy `work_handoff/WORK_PROMPT_COPY_PASTE.md` into the task. The prompt instructs Work to open only task-relevant files and reuse the code.
3. For the initial real experiment use Kaggle T4, enable GPU, and run `notebooks/Model_Mountain_Emotion_Lab_Kaggle.ipynb` with `OPEN_MULTI_MODEL_PANEL=True`.
4. The draw.io XMLs are under `course/drawio`, the 18-case CSV under `course`, the prefilled draft DOCX/PDF under `course/generated`, and the original source PDF under `reference/`.
5. CPU tests and the generated report are verified. Real GPU inference, valid cosine thresholds, and MIA execution are not yet verified.

Task-level index: `work_handoff/CONTEXT_INDEX.md`. Machine-readable status: `work_handoff/TASK_QUEUE.json` and `work_handoff/PROGRESS.json`.

## For course submission
The included `course/generated/CMPE187_Deliverable_2_DRAFT.docx` is a **template**, not an empirical-results submission: it explicitly says Not Run. Use the real multi-model matrix results JSON to regenerate the report. Review thresholds with instructor/course group before enabling pass/fail criteria. Edit diagrams in diagrams.net.
