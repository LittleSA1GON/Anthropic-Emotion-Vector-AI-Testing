"""Execute an explicit UI request on a Kaggle worker; no simulated model outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def execute(config, output, *, backend_factory=None):
    from .kaggle_runner import validate_request
    from .workbench import make_quick_vector, save_quick_vector
    from .artifacts import runtime_metadata
    from .course_tests import evaluate_model, starter_cases, training_pairs, summary
    config=validate_request(config)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    result={'schema':'mm-emotion.remote-experiment.v1','origin':'real_model_inference',
            'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
            'configuration':config,'runtime':runtime_metadata(),'models':[],
            'results':[],'failures':[],'course_results':[]}
    path=root/'experiment.json'
    def persist():path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    persist()
    try:
        if backend_factory is None:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('Kaggle CUDA GPU unavailable')
            from .hf_backend import HFBackend
            backend_factory=HFBackend
        for model in config['models']:
            backend=None;started=time.perf_counter()
            context={'stage':'model_load','prompt':None,'coefficient':None,'layer_0based':None,'seed':None}
            try:
                backend=backend_factory(model['model_id'],precision=model['precision'],max_length=config['max_length'])
                context['stage']='vector_discovery'
                captures={}
                directions,meta=make_quick_vector(backend,config['pairs'],config['layers'],
                    config['method'],label=config['emotion'],capture_sink=captures)
                vector=save_quick_vector(root/'vectors',model['alias'],directions,meta)
                (root/'captures').mkdir(exist_ok=True)
                np.savez_compressed(root/'captures'/(model['alias']+'.npz'),
                    positive=captures['positive'],neutral=captures['neutral'],directions=directions,
                    layers=np.array(config['layers']),
                    reference=backend.capture(config['neutral_reference'],config['layers']))
                vector_hash=hashlib.sha256(vector.read_bytes()).hexdigest()
                for prompt in config['prompts']:
                    for layer in config['intervention_layers']:
                        direction=directions[config['layers'].index(layer)]
                        for seed in config['seeds']:
                            for coefficient in config['coefficients']:
                                context={'stage':'generation','prompt':prompt,'coefficient':coefficient,
                                         'layer_0based':layer,'seed':seed}
                                generated=backend.generate(prompt,layer,direction,coefficient,
                                    max_new_tokens=config['max_new_tokens'],temperature=config['temperature'],
                                    top_p=config['top_p'],seed=seed,intervention_scope=config['scope'],capture_layer=layer)
                                if any(block._forward_hooks for block in backend.layers):
                                    raise RuntimeError('An activation hook remained after generation')
                                result['results'].append({'origin':'real_model_generation',
                                    'model_alias':model['alias'],'model_id':model['model_id'],
                                    'precision':model['precision'],'backend':'hf','prompt':prompt,
                                    'layer_0based':layer,'coefficient':coefficient,'seed':seed,
                                    'vector_path':vector.relative_to(root).as_posix(),'vector_sha256':vector_hash,
                                    'exception':None,**generated})
                                persist()
                if config['include_matrix']:
                    context={'stage':'course_matrix','prompt':None,'coefficient':0.,'layer_0based':config['intervention_layers'][0],'seed':config['seeds'][0]}
                    rows=evaluate_model(backend,starter_cases(),training_pairs(),config['intervention_layers'][0],seed=config['seeds'][0])
                    result['course_results'].extend([{**r,'model_alias':model['alias']} for r in rows])
                result['models'].append({**model,'status':'completed','seconds':time.perf_counter()-started,
                    'resolved_revision':getattr(backend.model.config,'_commit_hash',None),
                    'vector_path':vector.relative_to(root).as_posix(),'vector_sha256':vector_hash,
                    'layer_scores':meta['layer_scores'],
                    'peak_vram_bytes':int(backend.torch.cuda.max_memory_allocated()) if hasattr(backend,'torch') else None})
            except Exception as exc:
                error={**model,**context,'exception':repr(exc),'seconds':time.perf_counter()-started}
                result['failures'].append(error);result['models'].append({**error,'status':'failed'})
            finally:
                if backend is not None:backend.close()
                persist()
        result['status']='completed' if not result['failures'] else ('partial' if result['results'] else 'failed')
        if result['course_results']:
            course={'schema':'mm-emotion.multi-model.v1','mode':'matrix','results':result['course_results'],
                    'models':result['models'],'stats':summary(result['course_results'])}
            (root/'course_matrix.json').write_text(json.dumps(course,indent=2),encoding='utf8')
    except Exception as exc:
        result['status']='failed';result['failures'].append({'exception':repr(exc)})
    finally:persist()
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf8'))
    path=execute(config,args.out)
    print('Experiment evidence:',path)
    if json.loads(path.read_text(encoding='utf8'))['status']=='failed':raise SystemExit(1)
