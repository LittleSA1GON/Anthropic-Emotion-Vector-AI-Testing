# Dynamic visualization and measured inspection

Run `run.bat` on Windows or `bash run.sh` on macOS/Linux, then open http://127.0.0.1:8501. Ordinary users need neither Blender nor a local GPU. The existing model-link, CSV, per-user Kaggle API, course report, and local-GPU workflows remain available.

## Compute states and evidence

`mm_emotion/lifecycle.py` defines the shared states, append-only JSONL events, and the snapshot used by native status indicators and the scene. The worker records model loading, weight inspection, activation capture, averaging, subtraction, steering, comparison, completion, and failures. A usable GPU requires a successful CUDA tensor operation; selecting an accelerator or receiving RUNNING is insufficient.

The Kaggle client polls notebook status in an isolated Streamlit fragment every eight seconds while active. It enters QUEUED only when Kaggle reports that state. Connection failures preserve the last confirmed status. Submission failures are distinct from a failed running notebook. No queue position, allocation percentage, free-capacity estimate, or completion ETA is generated.

**Kaggle limitation:** the existing API workflow returns saved notebook outputs, not a live model telemetry stream. While a job runs, the UI reports only confirmed provider status. Detailed GPU/model events become available as **recorded** events after Kaggle publishes the outputs. The timeline replays those timestamped events; it never pretends they are live. A new prompt creates a new private batch.

| Display mode | Meaning |
|---|---|
| Live status | Confirmed client or Kaggle status; GPU may still be unverified |
| Recorded data | Published runtime events, architecture, tensors, and generation snapshots |
| Demonstration | Explicitly synthetic geometry and architecture; no model loaded or results generated |

Demonstration mode disables the setup's real submission button. Tokens stay in the UI session or local Kaggle login and subprocess environment. Decorative Blender assets are excluded from remote inference uploads; a separate source manifest verifies the worker files.

## Scientific inspection

Select a recorded model, case, and captured layer. Scene layer buttons and the native layer inspector share the same selection. Architecture counts and dimensions come from the loaded checkpoint, not the illustrative network template. Older recordings without architecture or weight metadata explicitly show those fields as unavailable.

- **Extract the direction:** recompute the emotional and matched-neutral means from saved training captures. Display the raw difference and its L2-normalized native direction, largest-changing dimensions, exact individual values, and a full numerical CSV download. The configured method remains final-prefill-token matched-pair contrasts.
- **Activation data:** inspect bounded heatmaps and histograms of actual hidden states. PCA axes in playback are fitted on training captures. The default 2D replay needs no WebGL; optional 3D PCA uses the same measurements. Generation snapshots appear only at the layer where they were actually captured.
- **Architecture & weights:** inspect recorded model configuration, component types, capabilities, and memory. For floating-point models, sample at most two parameter matrices at each scanned layer, with at most 2,048 uniformly indexed entries per matrix. Statistics describe these samples; up to 64 exact values and their indices are exposed. Attention parameter matrices are not attention probabilities. NF4 floating-point weights are unavailable here: packed storage bytes are not presented as weights and whole checkpoints are not dequantized for display.
- **Steering arithmetic:** compare the requested sum of signed native unit vectors with the actual difference between baseline and post-hook first-prefill activations. Display the residual, which can include floating-point rounding, and the corresponding responses. Interventions add to hidden states; they do not train model weights.

Changing a captured layer updates its input-minus-neutral scores and extraction values. If no generation was captured there, the UI says so. The separate Steering layer control can launch a new inference at any layer present in the exact native vector bundle.

The neural scene is **illustrative geometry**, not a visualization of every parameter or a neuron map. Its small numerical subtraction overlay uses actual recorded means. Exact values, parameter samples, and measured comparisons remain native controls and charts adjacent to the illustration. Scores are exploratory cosine alignments, not subjective feelings, probabilities, or approved course grades.

## Blender regeneration

Blender 5.2.3 was used for the bundled assets. From the repository root:

```bash
blender --background --python scripts/build_neural_assets.py -- --out assets/neural
```

On Windows, if Blender is not on PATH:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python scripts/build_neural_assets.py -- --out assets/neural
```

Outputs include editable `neural_lab.blend`, animated `neural_lab.glb`, `neural_scene.json`, and `neural-preview.png`. `--skip-render` regenerates geometry and exports without rendering a new fallback image. The source uses 48 objects sharing one low-poly node mesh, sparse connections, neutral/emotional aggregation motifs, a direction motif, materials, lights, and an animated camera. The loop closes at frame 121 over a 120-frame period.

The dependency-free Streamlit component in `assets/neural/index.html` projects the Blender-authored node template with Canvas 2D; the GLB is also available for native 3D reuse. Runtime layer structures adapt to recorded architecture, sampling at most 24 displayed blocks plus the selected block. Browser rendering never loads model weights or spends Kaggle quota. Motion pauses when hidden, can be paused/replayed manually, respects reduced-motion preferences, and has an explicit Reduce motion control. A flat 2D projection and static Blender fallback preserve access without WebGL or Canvas support.

Design follows [Impeccable](https://impeccable.style/docs/) through restrained hierarchy, semantic color, native accessible controls, explicit empty/error states, and low visual noise. No third-party design hooks or frontend packages are required.

## Reproduce checks

```bash
python -m pytest -q
python scripts/check_ui.py
python -m compileall -q mm_emotion scripts tests
python scripts/release_manifest.py
python scripts/build_notebook.py
python scripts/build_release.py
```

`tests/test_lifecycle.py` checks evidence gates, disconnect recovery, journal handling, exact extraction/steering arithmetic, and bounded real PyTorch parameter samples. Integrated tests check event wiring, native-vector reuse, and failures. `evidence/visual-lifecycle/` contains the small real T4 acceptance: two 8-token generations, 12 events, actual architecture, and six sampled tensors, with submitted-source provenance. It verifies instrumentation and arithmetic, not emotion accuracy.

The Kaggle notebook embeds the research source and native notebook controls. Blender/Canvas assets remain in GitHub and the full ZIP to avoid inflating GPU notebook uploads. IBM MIA and GTX 1080 Ti hardware execution remain unverified. Existing course thresholds and original v4 integration constraints are unchanged.
