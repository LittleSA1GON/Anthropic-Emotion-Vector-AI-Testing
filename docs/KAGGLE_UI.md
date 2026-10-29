# Run experiments from the UI using your Kaggle account

The **Kaggle Experiment Lab** page is a local control panel for private GPU jobs in your Kaggle account. It supports your own training pairs, test prompts, checkpoints, layer choices, signed strengths, and repeated seeds. Your computer does not need CUDA or downloaded model weights.

## Open the lab

```bash
python -m pip install -r requirements-ui.txt
python -m streamlit run app_emotion.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501/Kaggle_Experiment_Lab`. Windows users with the project virtual environment can run `scripts/launch_kaggle_lab.ps1`.

## Your account

Enter your Kaggle username and API token under **Compute account**. Use **Check Kaggle credentials** to verify access. Leave the token blank if your local Kaggle CLI is already authenticated.

Each user's token is held in their current Streamlit session and passed through a subprocess environment. It is not saved in requests, notebooks, experiment outputs, or Git. Run your own local UI process; this is not a shared hosted service with user authentication.

Public Qwen and SmolLM weights need no Hugging Face key. Gated models additionally require permission for the weights and an attached Kaggle Secret named `HF_TOKEN`; the UI does not embed model tokens in notebook source.

## Submit an experiment

1. Select one or more checkpoints. They load sequentially on the Kaggle worker.
2. Edit or import at least two matched neutral/emotional training pairs. Name the direction you want to study.
3. Choose decoder layers to scan, then the scanned layers to intervene on. Each intervention changes one layer at a time.
4. Enter test prompts and a neutral reference that are separate from the training pairs.
5. Choose coefficients such as `-3,0,3`, a seed, decoding settings, and an intervention scope. The zero control is mandatory.
6. Optionally enable the original 18-case course matrix. Its thresholds remain ungraded.
7. Click **Run on my Kaggle T4**. This sends the selected prompts, settings, and lab source to a private notebook under your account, using your GPU quota.

Kaggle runs queued batches. New prompts or settings create another job rather than a continuous chat connection. The existing saved lab notebook is not overwritten.

## Inspect results

Open **Results**, refresh the status, and retrieve the completed outputs. Choose a checkpoint, intervention layer, prompt, and seed.

- Compare actual unsteered and steered generations under the same checkpoint and sampling seed.
- Rotate the 3D PCA plot and press Play to animate the measured prefill activation across strengths. Axes are fitted on training/reference captures only.
- Inspect the projection and isolation table. `added_projection` is the activation change along the selected vector relative to zero. `orthogonal_delta_norm` is the change outside that vector; finite precision can produce a small residual.
- Download the generation CSV, raw JSON, and full ZIP containing native vectors and activation captures.

The snapshots come from a forward hook during real generation, after the intervention. With `decode_only` scope, the prefill snapshot correctly stays unchanged because that intervention starts later. These are reduced-dimensional model measurements, not emotion probabilities or evidence of subjective feelings.

## Interface direction

The UI follows the [StyleKit anti-template guide](https://www.stylekit.top/en/avoid-ai-slop): an editorial hierarchy, warm paper background, dark text, forest-green actions, small corner radii, visible focus states, and responsive controls. Animation is used to compare recorded activations.
