"""CMPE 187 Deliverable #2A classification-contract implementation.

All prompt examples are new author-authored *templates*, not copied from the PDF.
Inverted inequalities in the PDF are preserved verbatim as source data; adjusted
thresholds are PROPOSALS and must be explicitly approved before grading.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

EMOTIONS = ("happiness", "sadness", "anger")
ABSTRACTIONS = ("objective", "stated", "enacted")
DICTIONS = ("standard", "expressive")
INTENSITIES = {("objective", "standard"): "neutral", ("objective", "expressive"): "neutral/moderate",
               ("stated", "standard"): "moderate", ("stated", "expressive"): "moderate",
               ("enacted", "standard"): "moderate/high", ("enacted", "expressive"): "high"}
# PDF page 5-7: flipped inequality signs on most interval rows. Corrected values are unapproved.
BOUNDS = {("objective", "standard"): (-1.0, 0.2), ("objective", "expressive"): (0.2, 0.5),
          ("stated", "standard"): (0.5, 0.6), ("stated", "expressive"): (0.6, 0.7),
          ("enacted", "standard"): (0.7, 0.8), ("enacted", "expressive"): (0.7, 1.0)}
RAW_RANGES = ["cos(theta) <= 0.2", "0.2 >= cos(theta) >= 0.5", "0.5 >= cos(theta) >= 0.6",
              "0.6 >= cos(theta) >= 0.7", "0.7 >= cos(theta) >= 0.8", "0.7 >= cos(theta) >= 1.0"]
WORDS = {
 "happiness": {"objective": "A student received good results on an exam.",
               "stated": "The student says they feel happy about their results.",
               "enacted": "The student smiles, cheers, and celebrates the results with friends."},
 "sadness": {"objective": "A student received disappointing results on an exam.",
             "stated": "The student says they feel sad about their results.",
             "enacted": "The student tears up, lowers their head, and quietly withdraws after the results."},
 "anger": {"objective": "A student received an unfair grade after their work was misplaced.",
           "stated": "The student says they are angry about the unfair grade.",
           "enacted": "The student clenches their fists, raises their voice, and confronts the mistake."},
}
NEUTRAL = "A student received an update regarding their academic work."


def starter_cases() -> list[dict]:
    rows = []
    for emotion in EMOTIONS:
        for abstraction in ABSTRACTIONS:
            for diction in DICTIONS:
                suffix = " Please respond in a neutral explanatory sentence." if diction == "standard" else " Wow! Describe the situation vividly!"
                lo, hi = BOUNDS[abstraction, diction]
                rows.append({"case_id": f"{emotion[:3]}-{abstraction[:3]}-{diction[:3]}",
                             "emotion": emotion, "expected_emotion": emotion,
                             "intensity": INTENSITIES[abstraction, diction], "abstraction": abstraction,
                             "diction": diction, "prompt": WORDS[emotion][abstraction] + suffix,
                             "neutral_reference": NEUTRAL + suffix,
                             "source_range_text": RAW_RANGES[list(BOUNDS).index((abstraction, diction))],
                             "proposed_min": lo, "proposed_max": hi,
                             "threshold_status": "proposed_not_approved",
                             "source": "CMPE 187 Group Deliverable #2A, pp. 2-7; prompt_text=authored_starter"})
    validate_cases(rows)
    return rows


def training_pairs() -> list[dict]:
    topics = ("a scheduled project meeting", "a completed science fair presentation", "the return of a library book", "a completed team assignment")
    triggers = {"happiness": "pleased and delighted", "sadness": "sad and discouraged", "anger": "angry and frustrated"}
    out = []
    for e in EMOTIONS:
        for i, topic in enumerate(topics):
            # Training and 18-row evaluation use different situations and phrasing.
            out.append({"pair_id": f"train-{e}-{i}", "emotion": e, "scenario_id": f"train-{topic}",
                        "neutral_prompt": f"Consider {topic}. A person describes it without a strong emotion. Answer briefly.",
                        "emotion_prompt": f"Consider {topic}. A person feels {triggers[e]} about it. Answer briefly."})
    return out


def validate_cases(cases: list[dict], *, require_full_matrix: bool = False) -> dict:
    ids = set(); keys = set()
    for row in cases:
        for k in ("case_id", "emotion", "intensity", "abstraction", "diction", "prompt", "neutral_reference"):
            if not isinstance(row.get(k), str) or not row[k].strip():
                raise ValueError(f"Case missing {k}: {row}")
        if row.get("expected_emotion",row["emotion"]) not in EMOTIONS:
            raise ValueError(f"Invalid expected emotion for {row['case_id']}")
        if row["case_id"] in ids:
            raise ValueError(f"Duplicate case_id {row['case_id']}")
        ids.add(row["case_id"])
        if row["emotion"] not in EMOTIONS or row["abstraction"] not in ABSTRACTIONS or row["diction"] not in DICTIONS:
            raise ValueError(f"Unknown case classification: {row['case_id']}")
        if row["prompt"] == row["neutral_reference"]:
            raise ValueError(f"Identical prompt/reference for {row['case_id']}")
        keys.add((row["emotion"], row["abstraction"], row["diction"]))
        for k in ("proposed_min", "proposed_max"):
            if k in row and row[k] != "" and not (-1.0 <= float(row[k]) <= 1.0):
                raise ValueError(f"Out-of-range threshold for {row['case_id']}")
    if require_full_matrix and (len(keys) != 18 or len(cases) != 18):
        raise ValueError(f"Full matrix must contain 18 unique emotion x abstraction x diction cases; found {len(keys)}")
    return {"cases": len(cases), "unique_classifications": len(keys), "full_matrix": len(keys) == len(cases) == 18}


def read_cases(path: str | Path, *, require_full_matrix=False) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(rows, dict): rows = rows["cases"]
    elif p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
    else:
        raise ValueError("Case input must be CSV, JSON or JSONL")
    validate_cases(rows, require_full_matrix=require_full_matrix)
    return rows


def write_cases(path: str | Path, rows: list[dict]):
    validate_cases(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json": path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    elif path.suffix == ".jsonl": path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
    elif path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    else: raise ValueError("Unsupported output")


def cosine(a, b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom>1e-10 else None


def grade_case(case, similarities: dict, *, approve_proposed_thresholds=False) -> dict:
    """Do not silently grade proposed thresholds or unscorable zero-norm vectors."""
    expected=case["expected_emotion"]
    finite={k:v for k,v in similarities.items() if v is not None and math.isfinite(v)}
    prediction=max(finite,key=finite.get) if finite else None
    score=finite.get(expected)
    lo=float(case.get("proposed_min",-1.0)); hi=float(case.get("proposed_max",1.0))
    if not finite or score is None: status="unscorable_vector"
    elif not approve_proposed_thresholds: status="ungraded_provisional_thresholds"
    elif prediction != expected or not (lo<=score<=hi): status="fail"
    else: status="pass"
    return {"predicted_emotion":prediction, "expected_cosine":score,
            "emotion_match": prediction==expected if prediction else None,
            "in_proposed_range":lo<=score<=hi if score is not None else None,
            "status":status, "threshold_approved_for_run":approve_proposed_thresholds,
            "all_similarity_scores":finite}


def evaluate_model(backend, cases, train_pairs, layer:int, *, approve_proposed_thresholds=False,
                   generations=False, max_new_tokens=48, seed=42) -> list[dict]:
    """Actual HF model activations only. Same-model, held-out vector training.
    Test labels/prompt text must NOT be reused in train_pairs; enforced by string overlap.
    """
    validate_cases(cases)
    if layer not in range(len(backend.layers)): raise ValueError("Invalid zero-based layer")
    eval_text={x[k].strip() for x in cases for k in ("prompt","neutral_reference")}
    train_text={x[k].strip() for x in train_pairs for k in ("emotion_prompt","neutral_prompt")}
    if eval_text & train_text: raise ValueError("Training/test prompt leakage")
    from .vectors import normalized_difference
    vectors={}
    for e in EMOTIONS:
        epairs=[p for p in train_pairs if p["emotion"]==e]
        if len(epairs)<2: raise ValueError(f"Need >=2 training pairs for {e}")
        pos=np.stack([backend.capture(p["emotion_prompt"],[layer]) for p in epairs])
        neu=np.stack([backend.capture(p["neutral_prompt"],[layer]) for p in epairs])
        vec=normalized_difference(pos,neu)[0]
        if not np.isfinite(vec).all() or float(np.linalg.norm(vec))<1e-10:
            raise ValueError(f"Zero/nonfinite baseline emotion vector: {e}")
        vectors[e]=vec
    output=[]
    for case in cases:
        import time
        tic=time.perf_counter()
        p=backend.capture(case["prompt"],[layer])[0]
        n=backend.capture(case["neutral_reference"],[layer])[0]
        diff=p-n
        similarities={e:cosine(diff,v) for e,v in vectors.items()}
        scored=grade_case(case,similarities,approve_proposed_thresholds=approve_proposed_thresholds)
        record={**case,"model_id":backend.model_id,"precision":backend.precision,"layer_0based":layer,
                "origin":"real_model_inference", "capture":"last_input_token_decoder_block_output",
                "backend":"hf","seed":seed,"coefficient":0.,"exception":None,
                "duration_seconds":time.perf_counter()-tic,
                "activation_delta_norm":float(np.linalg.norm(diff)),**scored}
        if generations:
            record["direct_inference"]=backend.generate(case["prompt"],max_new_tokens=max_new_tokens,seed=seed)
        output.append(record)
    return output


def summary(rows: list[dict]) -> dict:
    counts=Counter(r.get("status","not_run") for r in rows)
    scores=[r["expected_cosine"] for r in rows if isinstance(r.get("expected_cosine"),(float,int))]
    labels=defaultdict(Counter)
    for r in rows:
        for k in ("model_id","emotion","abstraction","diction","intensity"):
            labels[k+":"+str(r.get(k,"?"))][r.get("status","not_run")]+=1
    return {"n_cases":len(rows),"status_counts":dict(counts),
            "scorable_cases":len(scores),"mean_expected_cosine":float(np.mean(scores)) if scores else None,
            "per_classification":{k:dict(v) for k,v in sorted(labels.items())},
            "caution":"Provisional thresholds are not calibrated accuracy criteria; neutral case class is conceptually ambiguous."}


def save_results(path, cases, records, config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    payload={"schema":"mm-emotion.cmpe187.v1","origin":"real_model_inference" if records else "not_run",
             "created_utc":datetime.now(timezone.utc).isoformat(),
             "cases_sha256":hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
             "settings":config,"results":records,"statistics":summary(records)}
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return payload
