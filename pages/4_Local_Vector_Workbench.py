"""Interactive Model Mountain Emotion Vector Workbench.

Run: streamlit run app_emotion.py
This is an experimental research UI: no model is downloaded or loaded until
"Load model" is clicked. GPU inference is never simulated.
"""
from __future__ import annotations
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from mm_emotion.dataset import load_pairs, validate_pairs
from mm_emotion.workbench import (
    parse_pairs, pair_to_jsonl, safe_name, make_quick_vector,
    save_quick_vector, load_vector, sweep_grid, validate_test_prompts,
    serialize_results,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "emotion_pairs_v1.jsonl"
VECTORS = ROOT / "runs" / "workbench_vectors"
SESSIONS = ROOT / "runs" / "workbench_sessions"
CUSTOM_DATA = ROOT / "runs" / "workbench_datasets"
for folder in (VECTORS, SESSIONS, CUSTOM_DATA):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Model Mountain | Vector Workbench", page_icon="🔬", layout="wide")
from mm_emotion.credentials import credential_panel
hf_token = credential_panel()
st.title("Model Mountain · Emotion Vector Workbench")
st.markdown('[Open the integrated Kaggle experiment lab](/)')
st.caption("Interactive real-model activation research · no synthetic outputs · research-only vectors, never security authorization")
st.info("Use the sidebar pages for **Multi-model experiments** and **CMPE 187 report builder**. This home page focuses on vector capture, isolation and steering.")

MODELS = {
    "Qwen2 0.5B (fastest)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2 1.5B (Model Mountain baseline)": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen2.5 1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5 3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5 7B (NF4 recommended)": "Qwen/Qwen2.5-7B-Instruct",
    "Custom model ID": None,
}


def release():
    old = st.session_state.pop("backend", None)
    st.session_state.pop("model_key", None)
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
            torch.cuda.ipc_collect()
    except Exception:
        pass


def backend():
    return st.session_state.get("backend")


def active_vector():
    path = st.session_state.get("active_vector")
    return Path(path) if path and Path(path).exists() else None


def dataset_editor_refresh(rows):
    st.session_state["pairs"] = rows
    st.session_state["editor_rev"] = st.session_state.get("editor_rev", 0) + 1


def unique_name(tag, ext="json"):
    return f"{safe_name(tag)}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}Z.{ext}"


with st.sidebar:
    st.header("1 · Model control")
    choice = st.selectbox("Model checkpoint", list(MODELS), index=0,
                          help="Weights are loaded only when you click Load model")
    selected_id = MODELS[choice] or st.text_input("Hugging Face model ID", "Qwen/Qwen2-1.5B-Instruct")
    precision = st.selectbox("Weight precision", ["fp16", "nf4", "fp32"],
                             help="GTX 1080 Ti: FP16 or NF4. CPU debug: FP32 only.")
    max_context = st.select_slider("Maximum input tokens", options=[128, 256, 512, 1024], value=512)
    pending_key = (selected_id, precision, max_context)
    loaded = backend()
    if loaded is not None:
        st.success(f"Loaded: {loaded.model_id} ({loaded.precision})")
        st.caption(f"{len(loaded.layers)} decoder blocks · 0-based layer numbering")
        if pending_key != st.session_state.get("model_key"):
            st.warning("Your selectors differ from the loaded model. Click Load model to switch.")
    else:
        st.info("No GPU model loaded yet")
    c1, c2 = st.columns(2)
    if c1.button("Load model", type="primary", use_container_width=True):
        if not selected_id.strip():
            st.error("Enter a model ID")
        else:
            release()
            try:
                from mm_emotion.hf_backend import HFBackend
                with st.spinner("Loading real model weights; first load may download them"):
                    obj = HFBackend(selected_id.strip(), precision=precision, max_length=max_context, token=hf_token)
                st.session_state["backend"] = obj
                st.session_state["model_key"] = pending_key
                st.success("Real model ready")
                st.rerun()
            except Exception as exc:
                st.error(f"Model load failed: {exc}")
    if c2.button("Unload", use_container_width=True):
        release()
        st.rerun()
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            st.caption(f"GPU: {p.name} · {p.total_memory/2**30:.1f} GiB · compute {p.major}.{p.minor}")
            st.caption(f"VRAM allocated: {torch.cuda.memory_allocated(0)/2**30:.2f} GiB")
        else:
            st.caption("GPU unavailable. Model execution requires appropriate CPU FP32 resources.")
    except Exception:
        st.caption("CUDA information unavailable")
    st.divider()
    st.markdown("**Evidence status**")
    st.caption("Manual quick-fit = training-only exploration. Use the full CLI experiment for validation-selected layers and held-out testing.")
    st.caption("IBM MIA remains a separate process, not a toggle inside this HF workbench.")


tab_data, tab_find, tab_steer, tab_history = st.tabs([
    "1 · Enter / import data", "2 · Find / inspect vectors", "3 · Isolate / steer model", "4 · Runs & exports",
])

with tab_data:
    st.subheader("Matched prompt pairs")
    st.write("Enter two or more neutral-versus-emotional prompt pairs, paste into the editable grid, or upload CSV / JSONL / JSON. No code changes required.")
    lc, rc = st.columns([1, 1])
    with lc:
        preset_emotion = st.selectbox("Load curated example pairs", ["joy", "calm", "anger", "anxiety"])
        if st.button("Load example pairs"):
            rows = [r for r in load_pairs(DATA) if r["emotion"] == preset_emotion and r["split"] == "train"]
            dataset_editor_refresh(rows)
            st.rerun()
    with rc:
        uploaded = st.file_uploader("Import pairs (neutral_prompt, emotion_prompt)", type=["csv", "json", "jsonl"])
        if uploaded and st.button("Import into editor"):
            try:
                rows = parse_pairs(uploaded.getvalue(), uploaded.name.rsplit(".", 1)[-1].lower())
                dataset_editor_refresh(rows)
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")
    if "pairs" not in st.session_state:
        dataset_editor_refresh([r for r in load_pairs(DATA) if r["emotion"] == "joy" and r["split"] == "train"])
    columns = ["neutral_prompt", "emotion_prompt", "emotion", "split", "scenario_id", "source", "id"]
    display_rows = [{k: x.get(k, "") for k in columns} for x in st.session_state["pairs"]]
    edited = st.data_editor(pd.DataFrame(display_rows, columns=columns),
                             num_rows="dynamic", use_container_width=True,
                             key=f"pair_editor_{st.session_state['editor_rev']}",
                             column_config={
                                 "neutral_prompt": st.column_config.TextColumn("Neutral context", width="large"),
                                 "emotion_prompt": st.column_config.TextColumn("Emotion context", width="large"),
                             })
    rows = [{k: (str(v).strip() if pd.notna(v) else "") for k, v in d.items()}
            for d in edited.to_dict("records") if any(pd.notna(v) and str(v).strip() for v in d.values())]
    try:
        parsed = parse_pairs(json.dumps(rows), "json")
        st.caption(f"{len(parsed)} usable pairs · Full split validation is optional and stricter than quick-fit.")
        c1, c2, c3 = st.columns(3)
        if c1.button("Use these pairs for discovery", type="primary"):
            st.session_state["pairs"] = parsed
            st.success("Ready in the Find vectors tab")
        c2.download_button("Download JSONL", data=pair_to_jsonl(parsed),
                           file_name="emotion_pairs_custom.jsonl", mime="application/x-ndjson")
        if c3.button("Save dataset locally"):
            path = CUSTOM_DATA / unique_name("emotion_pairs_custom", "jsonl")
            path.write_bytes(pair_to_jsonl(parsed))
            st.success(f"Saved {path.relative_to(ROOT)}")
        with st.expander("Check full train / validation / test suitability"):
            try:
                from mm_emotion.dataset import validate_pairs
                st.json(validate_pairs(parsed))
                st.success("Valid 4-emotion held-out dataset")
            except Exception as exc:
                st.info(f"Not yet a full 3-split benchmark: {exc}")
    except Exception as exc:
        st.warning(f"Fix pairs to continue: {exc}")
    st.divider()
    st.subheader("Ready-made full evaluation dataset")
    st.caption("Existing 48-pair train/validation/test dataset (all four emotions) remains available for CLI benchmark runs.")
    st.download_button("Download baseline 48-pair dataset", DATA.read_bytes(), file_name=DATA.name)

with tab_find:
    st.subheader("Discover emotion-related directions across layers")
    obj = backend()
    if obj is None:
        st.info("First load a real model in the sidebar. Input pairs can be prepared without a GPU.")
    else:
        n_layers = len(obj.layers)
        default_layers = [n for n in [5, 11, 17, 23] if n < n_layers] or [n_layers // 2]
        selected_layers = st.multiselect("Decoder blocks to scan (0-based)", list(range(n_layers)),
                                         default=default_layers, key="scan_layers")
        method = st.selectbox("Vector extraction", ["mean_diff", "pairwise_pca"],
                              help="Mean difference is a better starting point. PCA is exploratory.")
        label = st.text_input("Vector name / hypothesis", "joy")
        st.caption("Each pair performs two actual model forward passes per layer scan. We capture only the last input-token decoder block outputs.")
        if st.button("Capture pairs & discover vectors", type="primary"):
            try:
                pairs = parse_pairs(json.dumps(st.session_state.get("pairs", [])), "json")
                with st.spinner(f"Capturing {2*len(pairs)} actual model forward passes"):
                    dirs, meta = make_quick_vector(obj, pairs, selected_layers, method, label=label)
                path = save_quick_vector(VECTORS, unique_name(label, "").rstrip("."), dirs, meta)
                st.session_state["active_vector"] = str(path)
                st.session_state["last_fit"] = meta
                st.success(f"Saved vector: {path.relative_to(ROOT)}")
            except Exception as exc:
                st.error(f"Discovery failed: {exc}")
        if st.session_state.get("last_fit"):
            meta = st.session_state["last_fit"]
            st.warning("These scores are on the SAME training examples used to derive the vector. They are not held-out evidence.")
            view = pd.DataFrame(meta.get("layer_scores", []))
            if not view.empty:
                st.dataframe(view.sort_values("positive_fraction", ascending=False), hide_index=True, use_container_width=True)
                st.bar_chart(view.set_index("layer_0based")["mean_margin"])
    st.divider()
    st.subheader("Browse existing saved vectors")
    candidates = sorted((ROOT / "runs").glob("**/vectors/*.npz")) + sorted(VECTORS.glob("*.npz"))
    candidates = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if candidates:
        active = active_vector()
        index = candidates.index(active) if active in candidates else 0
        chosen = st.selectbox("Vector file", candidates, index=index,
                              format_func=lambda p: str(p.relative_to(ROOT)))
        if st.button("Select vector"):
            st.session_state["active_vector"] = str(chosen)
            st.success(f"Selected {chosen.relative_to(ROOT)}")
    else:
        st.info("No real vectors saved yet; capture pairs above or run the full CLI experiment.")
    if active_vector():
        st.caption("Active vector:")
        st.code(str(active_vector().relative_to(ROOT)))
        try:
            dirs, lays, meta = load_vector(active_vector())
            st.json({k: meta.get(k) for k in ("label", "emotion", "model_id", "precision", "method", "vector_method", "evaluation", "n_pairs", "origin")})
            st.caption(f"Stored directions: {dirs.shape[0]} layers × {dirs.shape[1]} hidden dimensions · layers {lays}")
        except Exception as exc:
            st.error(f"Vector file error: {exc}")

with tab_steer:
    st.subheader("Isolated A/B steering laboratory")
    obj = backend()
    chosen_path = active_vector()
    if obj is None or chosen_path is None:
        st.info("Load a model in the sidebar, then select or discover a vector in tab 2.")
    else:
        transfer = st.checkbox("Explicit FP16 → NF4 or NF4 → FP16 vector transfer (same model ID only)", value=False,
                               help="Cross-precision transfer is a separate experiment, not an automatic compatibility claim")
        try:
            dirs, layers, meta = load_vector(
                chosen_path, model_id=obj.model_id,
                precision=None if transfer else obj.precision,
                num_layers=len(obj.layers), hidden_size=obj.model.config.hidden_size)
            if transfer and meta.get("precision") != obj.precision:
                st.warning(f"Cross-precision transfer: {meta.get('precision')} vector → {obj.precision} model. Interpret separately.")
            st.success(f"Vector ready: {chosen_path.name} · {len(layers)} available layers")
            suggested = meta.get("layer_0based") or layers[0]
            layer = st.selectbox("Intervention decoder block (0-based)", layers,
                                 index=layers.index(suggested) if suggested in layers else 0)
            sign = st.radio("Direction", ["Positive emotion direction (+)", "Reversed direction (-)"], horizontal=True)
            direction = dirs[layers.index(layer)] * (-1 if sign.startswith("Reversed") else 1)
            c1, c2 = st.columns(2)
            with c1:
                strength = st.slider("Steering strength (vector units)", -20.0, 20.0, 3.0, step=0.25)
                coefficients = st.text_input("Dose-response sweep (includes baseline 0)", "-6,-3,0,3,6",
                                             help="Enter comma-separated strengths; zero control is always included")
                scope = st.selectbox("Where to modify decoder output", ["last_token", "prefill_only", "decode_only", "all_tokens"],
                                     help="last_token is the default; prefill/decode scopes isolate the phase; all_tokens is a stronger intervention")
            with c2:
                max_new = st.slider("Maximum generated tokens", 8, 256, 64, step=8)
                temp = st.slider("Temperature (0 = deterministic)", 0.0, 1.5, 0.0, step=0.1)
                top_p = st.slider("Top-p", 0.1, 1.0, 1.0, step=0.05)
                seed = st.number_input("Sampling seed", min_value=0, max_value=2_147_483_647, value=42)
            prompt = st.text_area("Test input (one prompt, or one prompt per line for a small batch)",
                                  "Explain how to approach an upcoming exam in two sentences.", height=120)
            mode = st.radio("Experiment", ["Single A/B (0 vs strength)", "Dose-response sweep", "Probe activation projection"],
                            horizontal=True)
            st.caption("Vector changes only the selected layer's activations; no model weights are fine-tuned or written. Hooks are removed after each generation.")
            if mode == "Probe activation projection":
                neutral = st.text_area("Reference neutral prompt", "I am preparing for an upcoming exam.")
                if st.button("Measure projection of contrast", type="primary"):
                    try:
                        with st.spinner("Capturing two real activations"):
                            ap = obj.capture(prompt, layers)
                            an = obj.capture(neutral, layers)
                        margins = np.einsum("lh,lh->l", ap-an, dirs)
                        df = pd.DataFrame({"layer_0based": layers, "relative_projection": margins.tolist()})
                        st.dataframe(df, hide_index=True)
                        st.bar_chart(df.set_index("layer_0based"))
                        st.caption("Projection is uncalibrated and depends on input and prompt formatting; this is not an emotion probability.")
                    except Exception as exc:
                        st.error(f"Probe failed: {exc}")
            elif st.button("Run real baseline + intervention", type="primary"):
                try:
                    prompts = validate_test_prompts(prompt, max_prompts=10)
                    coefs = [0.0, float(strength)] if mode.startswith("Single") else sweep_grid(coefficients)
                    coefs = list(dict.fromkeys(coefs))
                    if len(prompts) * len(coefs) > 40:
                        raise ValueError("Too many generations (>40). Reduce prompts or coefficients")
                    results = []
                    bar = st.progress(0)
                    with st.spinner("Running the same loaded model for baseline and steering"):
                        for i, (p, coef) in enumerate((p, c) for p in prompts for c in coefs):
                            r = obj.generate(p, layer, direction, coef, max_new_tokens=max_new,
                                             temperature=temp, top_p=top_p, seed=int(seed), intervention_scope=scope)
                            results.append({"prompt": p, "coefficient": coef, "layer_0based": layer, **r})
                            bar.progress((i+1)/(len(prompts)*len(coefs)))
                    settings = {"model_id": obj.model_id, "precision": obj.precision, "backend": "hf",
                                "vector_file": str(chosen_path.relative_to(ROOT)),
                                "vector_sha256": hashlib.sha256(chosen_path.read_bytes()).hexdigest(),
                                "vector_source_model": meta.get("model_id"), "vector_source_precision": meta.get("precision"),
                                "explicit_cross_precision_transfer": bool(transfer and meta.get("precision") != obj.precision),
                                "layer_0based": layer, "scope": scope, "direction_sign": -1 if sign.startswith("Reversed") else 1,
                                "coefficients": coefs, "seed": int(seed), "temperature": temp, "top_p": top_p,
                                "max_new_tokens": max_new, "vector_origin": meta.get("origin"),
                                "vector_evaluation": meta.get("evaluation", "train_val_test_pipeline")}
                    target = SESSIONS / unique_name("steering", "json")
                    serialize_results(target, results, settings)
                    st.session_state["last_results"] = {"results": results, "settings": settings,
                                                        "path": str(target.relative_to(ROOT))}
                    st.success(f"Saved genuine generations: {target.relative_to(ROOT)}")
                except Exception as exc:
                    st.error(f"Steering failed: {exc}")
            last = st.session_state.get("last_results")
            if last:
                st.subheader("Baseline versus intervention")
                results = last["results"]
                baseline = next((r for r in results if r["coefficient"] == 0), None)
                changed = next((r for r in results if r["coefficient"] != 0), None)
                if baseline:
                    left, right = st.columns(2)
                    left.markdown("**Unsteered baseline (coefficient 0)**")
                    left.write(baseline["text"] or "(empty output)")
                    right.markdown(f"**Steered (coefficient {changed['coefficient'] if changed else '—'})**")
                    right.write((changed or baseline)["text"] or "(empty output)")
                data = pd.DataFrame(results)
                st.dataframe(data, use_container_width=True, hide_index=True)
                if "new_tokens" in data and "coefficient" in data:
                    st.caption("Diagnostic: generated token count by steering coefficient (NOT an emotion score)")
                    st.line_chart(data.groupby("coefficient", as_index=True)["new_tokens"].mean())
                st.download_button("Export current A/B result as JSON", json.dumps(last, indent=2),
                                   file_name="vector_steering_result.json", mime="application/json")
                st.caption(f"Recorded in {last['path']} · Observable behavioral differences require independent labeling / blinded evaluations.")
        except Exception as exc:
            st.error(f"Cannot apply selected vector to this model: {exc}")

with tab_history:
    st.subheader("Search, compare, and export saved steering sessions")
    files = sorted(SESSIONS.glob("*.json"), reverse=True)
    if not files:
        st.info("No real steering sessions have been saved. Run an experiment in tab 3.")
    else:
        entries = []
        for file in files[:300]:
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                if item.get("schema") != "mm-emotion.steering-session.v1":
                    continue
                settings = item["settings"]
                entries.append({"file": file.name, "time": item["created_utc"],
                                "model": settings["model_id"], "precision": settings["precision"],
                                "layer": settings["layer_0based"], "scope": settings["scope"],
                                "outputs": len(item["results"])})
            except Exception:
                continue
        if entries:
            table = pd.DataFrame(entries)
            search = st.text_input("Filter by model / precision / run filename")
            if search:
                table = table[table.apply(lambda r: search.lower() in " ".join(map(str, r)).lower(), axis=1)]
            st.dataframe(table, hide_index=True, use_container_width=True)
            if not table.empty:
                selected = st.selectbox("Open session", table["file"].tolist())
                record = json.loads((SESSIONS / selected).read_text(encoding="utf-8"))
                st.json(record["settings"])
                st.dataframe(pd.DataFrame(record["results"]), hide_index=True, use_container_width=True)
                st.download_button("Download selected session", (SESSIONS / selected).read_bytes(),
                                   file_name=selected, mime="application/json")
                st.download_button("Export all visible runs (CSV)", table.to_csv(index=False),
                                   file_name="vector_sessions_index.csv", mime="text/csv")
    st.divider()
    st.markdown("**For rigorous research:** use the CLI benchmark and validation/test splits instead of selecting the best layer from training scores. The read-only Model Mountain integration only accepts traceable research evidence; it does not change the existing security gateway.")
