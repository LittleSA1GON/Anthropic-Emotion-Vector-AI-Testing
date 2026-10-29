# Model Mountain v4 integration contract

The user requested reuse of Model Mountain v4's behavioral-vector / MIA work. v4's archive is named `Model_Mountain_v4_Multiagent_Workflow_COMPLETE.zip`, but its raw bytes were not accessible to the authoring session, so no source-level merge was performed. This package is a **read-only research extension** with its own implementations of mean-difference vectors, layer selection, real HF forward hooks, MIA `capture_hs`, and `steer` export.

To install beside a locally extracted v4 project:

```bash
python scripts/install_into_model_mountain.py --root /path/to/model-mountain-v4   # dry run
python scripts/install_into_model_mountain.py --root /path/to/model-mountain-v4 --apply
```

To bridge research evidence into Model Mountain's evaluation/dashboard layer:

```bash
python -m mm_emotion.cli export-evidence --run runs/qwen1p5b_fp16_full \
    --out runs/qwen1p5b_fp16_full/model_mountain_evidence.json
```

Contract: `model-mountain.research.evidence.v1`; read-only structured evidence. `not_a_security_verdict=true`, `security_policy_effect=none`. No emotion vector may override authentication, document integrity, agent authorization, MCP validation, or extraction limits. Before adding a direct v4 module import, inspect actual v4 `model_mountain` vector and MIA APIs and add an integration test. Do not assume class or module names based on previous version summaries.
