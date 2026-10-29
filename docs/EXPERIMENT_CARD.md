# EXPERIMENT CARD

Question: Does 4-bit quantization preserve emotion-related activation directions and steering effects for the same language-model checkpoint?

Primary experiment: Qwen/Qwen2-1.5B-Instruct; Hugging Face decoder hooks; FP16 versus bitsandbytes NF4; four fixed 0-based blocks 5,11,17,23; sample 48 synthetic matched pairs; train on 6 per emotion, select block on 3 per emotion, describe held-out performance on 3 per emotion; steer using 0/3/6 coefficients; compare directional cosine and blind human-rated response behavior. All metrics descriptive.

Optional replication: IBM vLLM-Hook MIA on Kaggle T4, same model, `capture_hs` with 1-based decoder indices and a distinct `steer` worker, 512 MiB capture aperture. MIA backend must be verified to be semantically aligned with HF hooks before cross-backend metrics are interpreted. Direct NF4 quantization under vLLM may need supported prequantized AWQ/GPTQ checkpoint, not a converted NF4 checkpoint.

Data source: synthetic `data/emotion_pairs_v1.jsonl`. Do not call outcomes 'emotional intelligence' or evidence of sentience; the intervention changes numerical internal states and perhaps outputs, nothing more.

Artifacts: per-run manifest, checked .npz layers, normalized vectors, validation selection, test report, generation samples, comparison JSON and blinded annotation sheets. Disclose unsuccessful attempts and missing MIA runs explicitly.

Pre-registered pilot selection: choose the validation-best layer by positive fraction then mean margin, no test-informed tuning. Model weights and dataset must remain the same across precision comparisons; track device/library versions. Secondary evaluation: task quality degradation, run-to-run reproducibility, negative controls, human blind ratings, dose response, latency and memory.
