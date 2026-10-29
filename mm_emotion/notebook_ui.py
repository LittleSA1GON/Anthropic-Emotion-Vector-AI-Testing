"""Notebook-native interactive control panel for Kaggle/Jupyter.

Call display_workbench() after unpacking the suite. Requires ipywidgets,
Transformers and Torch for actual inference. No model loads until button click.
"""
from __future__ import annotations
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .dataset import load_pairs
from .workbench import (make_quick_vector, save_quick_vector, load_vector, safe_name,
                        parse_pairs, sweep_grid, serialize_results)


def display_workbench(project: str | Path = "."):
    import ipywidgets as w
    from IPython.display import display, clear_output
    root = Path(project).resolve()
    pairs_path = root / "data/emotion_pairs_v1.jsonl"
    vector_path = root / "runs/workbench_vectors"
    results_path = root / "runs/workbench_sessions"
    vector_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    state = {"backend": None, "vector": None, "pairs": []}

    model = w.Dropdown(description="Model", options=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct",
                                                     "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                                                     "Qwen/Qwen2.5-7B-Instruct"], layout=w.Layout(width="490px"))
    precision = w.Dropdown(description="Precision", options=["fp16", "nf4", "fp32"])
    load = w.Button(description="Load model", button_style="primary")
    unload = w.Button(description="Unload")
    message = w.Output(layout=w.Layout(border="1px solid #ccc", max_height="180px", overflow="auto"))

    emotion = w.Dropdown(description="Examples", options=["joy", "calm", "anger", "anxiety"])
    examples = w.Button(description="Load sample pairs")
    neutral = w.Textarea(description="Neutral", placeholder="A neutral context prompt", layout=w.Layout(width="95%", height="75px"))
    emotional = w.Textarea(description="Emotion", placeholder="Matched emotional context prompt", layout=w.Layout(width="95%", height="75px"))
    add_pair = w.Button(description="Add pair")
    paste = w.Textarea(description="Import", placeholder="Paste CSV (neutral_prompt,emotion_prompt) or JSONL records", layout=w.Layout(width="95%", height="80px"))
    paste_format = w.Dropdown(description="Format", options=["csv", "jsonl", "json"])
    import_pairs = w.Button(description="Import pasted pairs")
    pairs_view = w.Output()
    layers = w.Text(description="Scan blocks", value="5,11,17,23", layout=w.Layout(width="390px"))
    vec_label = w.Text(description="Vector", value="joy", layout=w.Layout(width="300px"))
    method = w.Dropdown(description="Fit", options=["mean_diff", "pairwise_pca"])
    fit = w.Button(description="Capture & fit", button_style="primary")
    saved = w.Dropdown(description="Use vector", options=[], layout=w.Layout(width="590px"))
    refresh = w.Button(description="Refresh vectors")
    use_vec = w.Button(description="Select vector")
    scan_view = w.Output()

    layer = w.Dropdown(description="Layer", options=[])
    scope = w.Dropdown(description="Scope", options=["last_token", "prefill_only", "decode_only", "all_tokens"])
    sign = w.Dropdown(description="Direction", options=[("Normal", 1), ("Reverse", -1)])
    coef = w.FloatSlider(description="Strength", value=3, min=-20, max=20, step=.5,
                         continuous_update=False, layout=w.Layout(width="450px"))
    sweep = w.Text(description="Sweep", value="-6,-3,0,3,6", layout=w.Layout(width="340px"))
    tokens = w.IntSlider(description="Max tokens", value=64, min=8, max=192, step=8, continuous_update=False)
    test_prompt = w.Textarea(description="Test prompt", value="Explain how to approach an upcoming exam in two sentences.",
                             layout=w.Layout(width="95%", height="105px"))
    transfer = w.Checkbox(description="Allow explicit cross-precision vector transfer (same model only)", value=False)
    one = w.Button(description="Baseline vs strength", button_style="primary")
    many = w.Button(description="Run sweep")
    outputs = w.Output()

    def log(*messages):
        with message:
            print(*messages)

    def update_pairs():
        with pairs_view:
            clear_output(wait=True)
            print(f"{len(state['pairs'])} pairs ready. First two are:")
            for p in state["pairs"][:2]:
                print(" NEUTRAL:", p["neutral_prompt"][:160])
                print(" EMOTION:", p["emotion_prompt"][:160])

    def get_backend():
        if state["backend"] is None:
            raise RuntimeError("Load a model first")
        return state["backend"]

    def choose_file(refresh_only=False):
        paths = sorted((root / "runs").glob("**/vectors/*.npz")) + sorted(vector_path.glob("*.npz"))
        paths = list(dict.fromkeys(p for p in paths if p.is_file()))
        current = saved.value
        saved.options = [(str(p.relative_to(root)), str(p)) for p in paths]
        if paths:
            if current in [str(p) for p in paths]:
                saved.value = current
            elif state["vector"] and str(state["vector"]) in [str(p) for p in paths]:
                saved.value = str(state["vector"])
        if not refresh_only and paths:
            log(f"Found {len(paths)} stored vectors")

    def select_vector(_=None):
        try:
            if not saved.value:
                raise ValueError("Fit or locate a .npz vector first")
            obj = get_backend()
            v, ls, meta = load_vector(saved.value, model_id=obj.model_id,
                                      precision=None if transfer.value else obj.precision,
                                      num_layers=len(obj.layers), hidden_size=obj.model.config.hidden_size)
            state["vector"] = Path(saved.value)
            layer.options = ls
            log("Selected", state["vector"].name, "layers:", ls,
                "; origin:", meta.get("origin"), "; vector precision:", meta.get("precision"))
        except Exception as exc:
            log("Cannot select vector:", str(exc))

    def on_load(_):
        with message:
            print("Loading actual weights; check GPU memory if failure occurs...")
        try:
            state["backend"] = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from .hf_backend import HFBackend
            obj = HFBackend(model.value, precision.value, max_length=512)
            state["backend"] = obj
            state["vector"] = None
            log("Model ready:", obj.model_id, obj.precision, "blocks:", len(obj.layers))
        except Exception as exc:
            log("Load failed:", str(exc))

    def on_unload(_):
        old = state["backend"]
        state["backend"] = None
        state["vector"] = None
        if old is not None:
            try:
                old.model.cpu()
            except Exception:
                pass
        del old
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log("Unloaded")

    def on_examples(_):
        state["pairs"] = [r for r in load_pairs(pairs_path) if r["emotion"] == emotion.value and r["split"] == "train"]
        vec_label.value = emotion.value
        update_pairs()

    def on_add(_):
        try:
            state["pairs"].extend(parse_pairs(json.dumps([{"neutral_prompt": neutral.value,
                                                             "emotion_prompt": emotional.value}]), "json"))
            neutral.value, emotional.value = "", ""
            update_pairs()
        except Exception as exc:
            log("Add pair failed:", str(exc))

    def on_import(_):
        try:
            state["pairs"] = parse_pairs(paste.value, paste_format.value)
            update_pairs()
        except Exception as exc:
            log("Import failed:", str(exc))

    def on_fit(_):
        try:
            obj = get_backend()
            selected = [int(s.strip()) for s in layers.value.split(",") if s.strip()]
            log("Running", len(state["pairs"])*2, "genuine capture passes...")
            directions, meta = make_quick_vector(obj, state["pairs"], selected, method.value,
                                                 label=vec_label.value)
            tag = safe_name(vec_label.value) + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            path = save_quick_vector(vector_path, tag, directions, meta)
            state["vector"] = path
            choose_file(refresh_only=True)
            saved.value = str(path)
            layer.options = selected
            with scan_view:
                clear_output(wait=True)
                print("TRAINING-ONLY contrast margins; these are not held-out results")
                for row in meta["layer_scores"]:
                    print("layer", row["layer_0based"], "mean", round(row["mean_margin"], 4),
                          "positive fraction", round(row["positive_fraction"], 3))
            log("Saved real vector:", str(path))
        except Exception as exc:
            log("Fit failed:", str(exc))

    def on_run(dose):
        try:
            obj = get_backend()
            if not state["vector"]:
                raise ValueError("Fit or select a vector first")
            dirs, ls, meta = load_vector(state["vector"], model_id=obj.model_id,
                                        precision=None if transfer.value else obj.precision,
                                        num_layers=len(obj.layers), hidden_size=obj.model.config.hidden_size)
            coefs = sweep_grid(sweep.value) if dose else list(dict.fromkeys([0.0, float(coef.value)]))
            direction = dirs[ls.index(layer.value)] * sign.value
            if not test_prompt.value.strip():
                raise ValueError("Enter a prompt")
            log("Running genuine baseline and", len(coefs)-1, "steering conditions...")
            results = []
            for k in coefs:
                r = obj.generate(test_prompt.value.strip(), int(layer.value), direction, k,
                                 max_new_tokens=tokens.value, intervention_scope=scope.value)
                results.append({"prompt": test_prompt.value.strip(), "coefficient": k, **r})
            settings = {"model_id": obj.model_id, "precision": obj.precision,
                        "vector_file": str(state["vector"].relative_to(root)),
                        "vector_sha256": hashlib.sha256(state["vector"].read_bytes()).hexdigest(),
                        "vector_source_precision": meta.get("precision"),
                        "explicit_cross_precision_transfer": bool(meta.get("precision") != obj.precision and transfer.value),
                        "layer_0based": layer.value, "direction_sign": sign.value, "scope": scope.value,
                        "max_new_tokens": tokens.value, "coefficients": coefs, "seed": 42,
                        "temperature": 0.0, "vector_evaluation": meta.get("evaluation", "train_val_test_pipeline")}
            path = results_path / ("steering_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + ".json")
            serialize_results(path, results, settings)
            with outputs:
                clear_output(wait=True)
                for r in results:
                    print("="*50, "coefficient", r["coefficient"], "="*10)
                    print(r["text"] or "(empty output)")
                print("Saved", path)
        except Exception as exc:
            log("Intervention failed:", str(exc))

    load.on_click(on_load)
    unload.on_click(on_unload)
    examples.on_click(on_examples)
    add_pair.on_click(on_add)
    import_pairs.on_click(on_import)
    fit.on_click(on_fit)
    refresh.on_click(lambda _: choose_file())
    use_vec.on_click(select_vector)
    one.on_click(lambda _: on_run(False))
    many.on_click(lambda _: on_run(True))
    on_examples(None)
    choose_file(refresh_only=True)

    model_panel = w.VBox([w.HTML("<h3>Model · load one precision at a time</h3>"), model, precision,
                          w.HBox([load, unload]), message])
    data_panel = w.VBox([w.HTML("<h3>Data · add two or more contrastive prompt pairs</h3>"), emotion, examples,
                         neutral, emotional, add_pair, paste_format, paste, import_pairs, pairs_view])
    vector_panel = w.VBox([w.HTML("<h3>Vector discovery · capture selected decoder layers</h3>"),
                           layers, vec_label, method, fit, scan_view, w.HBox([saved, refresh, use_vec])])
    test_panel = w.VBox([w.HTML("<h3>Vector isolation · baseline against only the chosen activation intervention</h3>"),
                         layer, scope, sign, coef, sweep, tokens, test_prompt, transfer,
                         w.HBox([one, many]), outputs])
    tabs = w.Tab(children=[model_panel, data_panel, vector_panel, test_panel])
    for idx, title in enumerate(["1 Model", "2 Data", "3 Find vectors", "4 A/B steer"]):
        tabs.set_title(idx, title)
    display(tabs)
    tabs._release_model = lambda: on_unload(None)
    return tabs
