# DATASET DATASHEET — emotion_pairs_v1.jsonl

**Dataset type:** fully synthetic human-authored structured paired prompts (generated deterministically from a short template). **Size:** 48 paired observations = 96 prompts across 12 distinct scenario topics, 4 emotion conditions and 3 disjoint splits. **License:** the suite's own templates; no external personal or private records. **Purpose:** smoke/pilot research on activation contrasts, NOT a benchmark for inferring internal feelings, psychology, or real-world affect.

## Composition

Each row: `id` unique pair; `scenario_id` groups related prompts; `split` (`train`, `validation`, `test`); `emotion` (`joy`, `calm`, `anger`, `anxiety`); `neutral_prompt` and `emotion_prompt` holding a fixed scenario but differing by a short first-person emotion framing; `source=author_created_synthetic_v1`; `annotation_note` guidance (not training label). Scenarios assigned 6 train, 3 validation, 3 test, shared consistently across all four emotions. There are thus 24/12/12 paired rows across splits.

## Known weaknesses

- Directly names affective states in positive prompts, so vectors can reflect lexical words rather than a latent affective function.
- Synthetic first-person prompts can reflect style instruction/assistant formatting; not necessarily a general emotion mechanism.
- Only 3 held-out scenarios per emotion; descriptive metrics cannot establish stable generalization.
- No blinded label annotation or natural conversational examples. The data must not be called benchmark-standard or psychologically validated.
- Some neutral prompts are repeated across emotions for the same scenario by design; the splitting guarantee is by `scenario_id`, not row. This is appropriate for controlled contrasts but can induce common neutral direction biases.

## Auditing and versioning

Run `python scripts/make_dataset.py` then `python -m mm_emotion.cli validate`. Hash of initial generated v1 content: `3f71a11c2c9a6d31f0837089cdb2f3a95305ec84c3501c29bac0a34db8ff235e`. Any edit changes this hash and invalidates direct comparability. Before publication, create version 2 with longer controlled, non-lexical emotions, human annotation, and leakage-safe unseen scenario groups; keep v1 for reproducibility.
