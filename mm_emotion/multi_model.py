"""Direct inference and controlled cross-model experiments with one GPU.

Sequential model loading prevents accidental VRAM aggregation on 11/16 GiB GPUs.
Each model has independent vectors, layers and baseline generations. Never
silently transfer a vector from a different model architecture/checkpoint.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .course_tests import read_cases, training_pairs, evaluate_model, save_results, summary
from .workbench import load_vector

REGISTRY = {
    "qwen05": {"model_id":"Qwen/Qwen2.5-0.5B-Instruct","precision":"fp16","description":"Fastest CUDA smoke test"},
    "qwen15": {"model_id":"Qwen/Qwen2.5-1.5B-Instruct","precision":"fp32","description":"FP32: FP16 capture overflow observed on Kaggle T4"},
    "qwen3": {"model_id":"Qwen/Qwen2.5-3B-Instruct","precision":"nf4","description":"4-bit candidate"},
    "qwen7": {"model_id":"Qwen/Qwen2.5-7B-Instruct","precision":"nf4","description":"11/16GB experimental"},
    "smol135": {"model_id":"HuggingFaceTB/SmolLM2-135M-Instruct","precision":"fp32","description":"Small second-family comparison checkpoint"},
    "llama1b": {"model_id":"meta-llama/Llama-3.2-1B-Instruct","precision":"fp16","description":"License-gated Hugging Face access"},
    "gemma1b": {"model_id":"google/gemma-3-1b-it","precision":"fp16","description":"Architecture-specific compatibility must be checked"},
}


def parse_models(spec: str, custom_path: str | None = None) -> list[dict]:
    reg = dict(REGISTRY)
    if custom_path:
        obj=json.loads(Path(custom_path).read_text(encoding="utf8"))
        for key,item in obj.items():
            if not isinstance(item,dict) or "model_id" not in item:raise ValueError(f"Invalid registry entry {key}")
            reg[key]=item
    names=[v.strip() for v in spec.split(",") if v.strip()]
    if not names or len(names)>8 or len(names)!=len(set(names)):
        raise ValueError("Choose 1-8 unique model aliases")
    out=[]
    for name in names:
        if name not in reg:raise ValueError(f"Unknown model {name}. Available: {list(reg)}")
        item=dict(reg[name]);item["alias"]=name
        if item.get("precision") not in ("fp16","nf4","fp32"):
            raise ValueError("Only HF fp16/nf4/fp32 supported; MIA has separate backend")
        out.append(item)
    return out


def read_prompts(path: str | Path) -> list[dict]:
    p=Path(path)
    if p.suffix.lower()==".txt":
        vals=[s.strip() for s in p.read_text(encoding="utf8").splitlines() if s.strip()]
        rows=[{"id":f"prompt-{i+1:03d}","prompt":v} for i,v in enumerate(vals)]
    elif p.suffix.lower()==".jsonl":
        rows=[json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s.strip()]
    elif p.suffix.lower()==".json":
        rows=json.loads(p.read_text(encoding="utf8"))
        if isinstance(rows,dict):rows=rows["prompts"]
    elif p.suffix.lower()==".csv":
        with p.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    else:raise ValueError("Prompts file must be txt/json/jsonl/csv")
    if not isinstance(rows,list) or not (1<=len(rows)<=250):raise ValueError("Expected 1-250 prompts")
    normalized=[]
    for i,r in enumerate(rows):
        if not isinstance(r,dict) or not isinstance(r.get("prompt"),str) or not r["prompt"].strip():
            raise ValueError(f"Missing prompt at row {i+1}")
        if len(r["prompt"])>12000:raise ValueError(f"Prompt {i+1} too long")
        normalized.append({"id":str(r.get("id") or f"prompt-{i+1:03d}"),"prompt":r["prompt"].strip()})
    return normalized


def release_model(backend):
    if backend is None:return
    try: backend.model.cpu()
    except Exception: pass
    del backend
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:pass


def run_models(models:list[dict], *, prompts:list[dict]|None=None, cases:list[dict]|None=None,
               mode="infer", layer=5, max_new_tokens=64, temperature=0., top_p=1., seed=42,
               approve_proposed_thresholds=False, continue_on_error=False,
               vector_paths:dict|None=None, coefficients:list[float]|None=None,
               seeds:list[int]|None=None, token=None):
    """Run models one after another; original model never remains in memory.

    Mode infer: baseline inference for prompts.
    Mode matrix: fit per-model distinct heldout train pairs, then evaluate 18 cases.
    Mode sweep: one vector per model alias loaded from its native checkpoint,
                coefficients with mandatory zero baseline.
    """
    if mode not in ("infer","matrix","sweep"):raise ValueError("Unknown mode")
    if mode in ("infer","sweep") and not prompts:raise ValueError("Prompts required")
    if mode=="matrix" and not cases:raise ValueError("Cases required")
    if mode=="sweep" and not vector_paths:raise ValueError("Native per-model vector path required")
    seeds = list(dict.fromkeys(seeds if seeds is not None else [seed]))
    if not seeds or len(seeds)>20 or any(type(s) is not int or not 0<=s<=2147483647 for s in seeds):
        raise ValueError("Choose 1-20 integer seeds in [0,2147483647]")
    if mode=="matrix" and len(seeds)>1:
        raise ValueError("Matrix captures are deterministic; use one seed per matrix run")
    from .artifacts import runtime_metadata
    runtime=runtime_metadata()
    output=[];errors=[];manifest=[]
    for m in models:
        backend=None
        context={"prompt":None,"prompt_id":None,"coefficient":None,"seed":seed}
        tic=time.perf_counter()
        try:
            from .hf_backend import HFBackend
            auth = {"token":token} if token else {}
            backend=HFBackend(m["model_id"],precision=m["precision"],max_length=int(m.get("max_length",512)),**auth)
            if mode=="matrix":
                rec=evaluate_model(backend,cases,training_pairs(),layer,
                                   approve_proposed_thresholds=approve_proposed_thresholds,
                                   generations=bool(m.get("generations",False)),max_new_tokens=max_new_tokens,seed=seed)
                output.extend({**r,"model_alias":m["alias"]} for r in rec)
            elif mode=="infer":
                for r, run_seed in ((r,s) for r in prompts for s in seeds):
                    context={"prompt":r['prompt'],"prompt_id":r['id'],"coefficient":0.,"seed":run_seed}
                    res=backend.generate(r["prompt"], max_new_tokens=max_new_tokens,
                                         temperature=temperature,top_p=top_p,seed=run_seed)
                    output.append({"origin":"real_model_generation","model_alias":m["alias"],"model_id":m["model_id"],
                                   "precision":m["precision"],"id":r["id"],"prompt":r["prompt"],
                                   "backend":"hf","seed":run_seed,"exception":None,
                                   "layer_0based":None,"coefficient":0.,"output":res})
            else:
                path=vector_paths.get(m["alias"])
                if not path:raise ValueError(f"Missing native vector for {m['alias']}")
                dirs,layers,meta=load_vector(path,model_id=m["model_id"],precision=m["precision"],
                                             num_layers=len(backend.layers),hidden_size=backend.model.config.hidden_size)
                if layer not in layers:raise ValueError(f"Selected layer {layer} absent in vector for {m['alias']}")
                direction=dirs[layers.index(layer)]
                grid=list(dict.fromkeys([0.]+(coefficients or [-3.,0.,3.])))
                if any(not np.isfinite(c) or abs(c)>30 for c in grid):raise ValueError("Bad coefficients")
                for r, run_seed in ((r,s) for r in prompts for s in seeds):
                    for coeff in grid:
                        context={"prompt":r['prompt'],"prompt_id":r['id'],"coefficient":coeff,"seed":run_seed}
                        res=backend.generate(r["prompt"],layer,direction,coefficient=coeff,
                                             max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,seed=run_seed)
                        output.append({"origin":"real_model_generation","model_alias":m["alias"],"model_id":m["model_id"],
                                       "precision":m["precision"],"id":r["id"],"prompt":r["prompt"],
                                       "vector_path":str(path),"vector_hash":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                       "backend":"hf","seed":run_seed,"exception":None,
                                       "vector_metadata_hash":hashlib.sha256(Path(path).with_suffix('.json').read_bytes()).hexdigest(),
                                       "layer_0based":layer,"coefficient":coeff,"output":res})
            manifest.append({"alias":m["alias"],"model_id":m["model_id"],"precision":m["precision"],
                             "device":str(backend.device) if hasattr(backend,'device') else 'unspecified',
                             "decoder_layers":len(backend.layers),
                             "resolved_revision":getattr(getattr(backend.model,'config',None),'_commit_hash',None),
                             "status":"completed","seconds":round(time.perf_counter()-tic,3)})
        except Exception as exc:
            err={"model_alias":m["alias"],"model_id":m["model_id"],"precision":m["precision"],
                 "backend":"hf","layer_0based":layer,"seeds":seeds,
                 "seconds":round(time.perf_counter()-tic,3),"error":repr(exc),**context}
            errors.append(err);manifest.append({**err,"status":"failed"})
            if not continue_on_error:raise RuntimeError(f"Model {m['alias']} failed: {exc}") from exc
        finally:
            if backend is not None:
                try: backend.model.cpu()
                except Exception: pass
                del backend
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():torch.cuda.empty_cache()
            except ImportError:pass
    return {"schema":"mm-emotion.multi-model.v1","origin":"real_model_inference" if output else "no_successful_inference",
            "created_utc":datetime.now(timezone.utc).isoformat(),"mode":mode,"runtime":runtime,"models":manifest,
            "configuration":{"layer_0based":layer,"max_new_tokens":max_new_tokens,"temperature":temperature,"top_p":top_p,
                             "seed":seed,"seeds":seeds,"provisional_thresholds_approved_for_run":approve_proposed_thresholds},
            "results":output,"failures":errors,"stats":summary(output) if mode=="matrix" else {"outputs":len(output)}}


def export_flat_csv(run_path: str | Path, payload: dict) -> Path:
    """Flatten recorded model evidence for spreadsheet/manual rubric review."""
    p=Path(run_path).with_suffix('.csv')
    if p.exists(): raise FileExistsError(f'Refuse to overwrite result CSV: {p}')
    values=payload.get('results',[])
    if not values:return p
    keys=sorted({k for record in values for k in record})
    with p.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for record in values:
            writer.writerow({k:json.dumps(record[k],ensure_ascii=False) if isinstance(record.get(k),(dict,list)) else record.get(k,'') for k in keys})
    return p


def main(argv=None):
    parser=argparse.ArgumentParser(description="Sequential multi-model direct inference / rubric testing / steering sweeps")
    parser.add_argument("--models",default="qwen05,qwen15",help="Comma-separated aliases from model registry")
    parser.add_argument("--registry",default=None,help="Optional custom JSON aliases and checkpoints")
    parser.add_argument("--mode",choices=["infer","matrix","sweep"],default="infer")
    parser.add_argument("--prompts",default="course/direct_inference_prompts.txt")
    parser.add_argument("--cases",default="course/18_test_cases.csv")
    parser.add_argument("--out",required=True)
    parser.add_argument("--layer",type=int,default=5)
    parser.add_argument("--max-new-tokens",type=int,default=64)
    parser.add_argument("--temperature",type=float,default=0.)
    parser.add_argument("--top-p",type=float,default=1.)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--seeds",default=None,help="Repeated sampling seeds, e.g. 42,43,44 (infer/sweep)")
    parser.add_argument("--accept-provisional-thresholds",action="store_true",help="Grading is opt-in because PDF bounds conflict")
    parser.add_argument("--continue-on-error",action="store_true")
    parser.add_argument("--native-vectors",default=None,help="JSON map model alias -> path to vector NPZ, for sweep")
    parser.add_argument("--coefficients",default="-3,0,3")
    args=parser.parse_args(argv)
    path=Path(args.out)
    if path.exists():raise SystemExit(f"Refuse to overwrite: {path}")
    if args.mode=="matrix" and not args.accept_provisional_thresholds:
        print("NOTE: Metrics collected but thresholds UNGRADED until instructor review; use --accept-provisional-thresholds only after approval.")
    result=run_models(parse_models(args.models,args.registry),mode=args.mode,layer=args.layer,
                      prompts=read_prompts(args.prompts) if args.mode!="matrix" else None,
                      cases=read_cases(args.cases) if args.mode=="matrix" else None,
                      max_new_tokens=args.max_new_tokens,temperature=args.temperature,top_p=args.top_p,seed=args.seed,
                      approve_proposed_thresholds=args.accept_provisional_thresholds,
                      continue_on_error=args.continue_on_error,
                      vector_paths=json.loads(Path(args.native_vectors).read_text()) if args.native_vectors else None,
                      coefficients=[float(x.strip()) for x in args.coefficients.split(",")],
                      seeds=[int(s.strip()) for s in args.seeds.split(',')] if args.seeds else None)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    csv_path=export_flat_csv(path,result)
    print(json.dumps({"csv":str(csv_path) if csv_path.exists() else None,"out":str(path),"models":result["models"],"n":len(result["results"]),"errors":len(result["failures"])},indent=2))

if __name__=="__main__":main()
