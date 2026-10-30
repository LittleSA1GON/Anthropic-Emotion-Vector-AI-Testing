# Model link to emotion steering

## Start

Install Python 3.11 or newer. Double-click `run.bat` on Windows; on macOS/Linux run `bash run.sh`. The launcher installs the UI dependencies on first use and opens http://127.0.0.1:8501. It reuses a running instance rather than opening a duplicate server.

Command-line alternatives:

```bash
python launch.py
python launch.py --port 8502
python launch.py --check
```

Model loading and GPU execution happen in your Kaggle account. The local client needs no model-weight downloads.

The [dynamic visualization guide](DYNAMIC_VISUALIZATION.md) explains live-status limits, recorded event playback, weight/activation inspection, and Blender regeneration. Detailed runtime events become available when Kaggle publishes outputs; the UI does not infer GPU allocation from a submitted or running job.

## Setup

1. Paste a Hugging Face checkpoint link, such as `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`, and choose **Check and add model**. Add up to three checkpoints. The preflight checks the architecture and records the revision; the actual Kaggle run verifies weight loading and decoder hooks. Unsupported architectures report an error rather than returning simulated results.
2. Choose **Anthropic research pilot** for twelve training pairs and eight test/control prompts, or **CMPE 187 course matrix** for the original eighteen course cases. Run the selected example directly, save it with **Download example CSV**, or upload a single CSV to replace it. See [the research pilot guide](ANTHROPIC_PILOT.md).
3. Enter your Kaggle username and API token. A blank token uses an existing local Kaggle CLI login. Tokens remain in the UI session and subprocess environment, outside notebooks, requests, output files, and Git.
4. Click **Run experiment**. Settings such as layer, seed, decoding temperature, and isolation examples are under **Experiment settings**.

Public Qwen and SmolLM weights need no model token. Gated checkpoints require model access and an attached Kaggle `HF_TOKEN` secret; model tokens are not embedded in notebook source. Each user runs a separate local UI with their own Kaggle credentials.

## CSV format

Required columns: `row_type,case_id,emotion,prompt,neutral_prompt`.

- `train` rows: at least two distinct emotional/neutral prompt pairs per emotion. These learn that emotion's native direction.
- `test` rows: new prompts and matched neutral references. They must be separate from training prompts. The emotion column can record an expected label.
- `case_id`: a unique identifier for every row.

The template also preserves the course's abstraction, diction, intensity, expected-label, original range, and provisional-threshold columns. These classifications are not changed or silently graded. Up to six emotion labels and fifty test rows are supported.

## Results and steering

Use **Update results** to check the queued batch and retrieve its outputs. Select the model and test case in the same workspace.

- **Emotion gauges** show raw cosine alignment of the neutral-subtracted activation with each learned direction, on a -1 to +1 scale. The closest positively aligned direction is reported; zero differences are unscorable. These scores are not probabilities or calibrated psychological labels.
- **Experiment playback** shows input activation, matched neutral, subtraction, vector comparison, and recorded steering states. PCA axes are fitted on training captures. Direction arrows are scaled for visibility, while the activation points and subtraction data come from real model captures.
- **Generated text** compares the zero baseline and measured isolation conditions. **All test results** exports the classification table.
- Each emotion has a distinct, consistent color and a signed steering slider. Set several sliders to combine vectors, enter an inference prompt and matched neutral reference, and click **Generate with these emotion settings**.

Inference uploads the exact native vector bundle and pins the checkpoint revision. The worker checks the model ID, precision, revision, training hash, and vector checksum before reuse. It then records an unsteered response and the requested combined intervention with the same seed. Model weights are not fine-tuned. Hooks are removed after generation.

Kaggle provides queued GPU batches, so new prompts and steering settings create another private job. The animations replay captured measurements after the job finishes. Download the experiment ZIP for vectors, activations, configuration, model outputs, scores, and exceptions.

Advanced local-GPU, legacy batch, and course-report tools remain available in the collapsed **Advanced tools** section. The UI follows the supplied StyleKit and Impeccable references through progressive disclosure, a clear primary workflow, restrained styling, and semantic emotion colors.
