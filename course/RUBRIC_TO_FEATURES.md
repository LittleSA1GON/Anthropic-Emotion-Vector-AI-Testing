# Deliverable #2 rubric -> exact implemented software features

Source: `CMPE187 - Group (v'-')b - Deliverable #2A.pdf`, pages 1-8. Prompt *examples* are newly authored; categories and thresholds reflect the source, with issues flagged.

| Required section | App / code | Generated evidence |
|---|---|---|
| 1 AI Function Test Requirement Analysis | `course_tests.py`, report generator | DOCX Section 1: 3 target emotions and methodological requirements |
| 2.1 Context Modeling | `drawio.py` | `course/drawio/2.1_Context_Model.drawio` |
| 2.2 Input Classifications | `drawio.py`, editable matrix | `2.2_Input_Classification.drawio`, `18_test_cases.csv` |
| 2.3 Output/Event/Action Classifications | `course_tests.py` scoring | `2.3_Output_Event_Action.drawio` |
| 2.4 Classification Decision Table | `course_tests.py` source/proposed columns | `2.4_Decision_Table.drawio`, DOCX table |
| 3.1 Test data models | 18-case CSV/JSON, training pairs, class validation | Prompt text, expected emotion, context, expected bounds |
| 3.2 Test case reports | `multi_model.py --mode matrix` | Real JSON per case per model, DOCX observed results |
| 3.3 Test case analysis/statistics | `summary()` | Counts, mean cosine, per-dimension status, coverage |
| 4.1 Test results/complexity/statistics | `course_report.py` | Per-model results, complexity (18 primary cases/model) |
| 4.2 Bug analysis | `course_report.py` | Known requirements defects and recorded real failure cases |
| 4.3 Test quality assessment | `course_report.py` | Leakage rules, calibration, repeatability, traceability |

**Important threshold discrepancy:** Page 5-7 bounds `0.2 >= cos(theta) >= 0.5`, etc. cannot hold. The software keeps the exact strings in `source_range_text`, stores proposed ascending bounds, and leaves all judgments ungraded unless explicitly reviewed by the course team/instructor.

**Neutral classification discrepancy:** The original PDF's neutral rows still say the expected emotion is Happy/Sad/Angry, although low cosine is expected. The tool preserves expected labels and flags ambiguity. No claim that neutral is a genuine positive emotion case.

**Scientific caution:** High cosine with a fitted direction is a proxy for vector alignment, not proof of emotion recognition or consciousness. The prefilled DOCX is a draft; no empirical values until real model runs.
