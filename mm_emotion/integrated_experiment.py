"""Multi-emotion discovery, neutral subtraction, and native-vector steering."""
from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .experiment_inputs import validate_integrated_request,emotion_colors


def score_emotions(delta,directions,labels):
    delta=np.asarray(delta,dtype=np.float64);directions=np.asarray(directions,dtype=np.float64)
    norm=float(np.linalg.norm(delta));scores={}
    for label,direction in zip(labels,directions):
        denominator=norm*float(np.linalg.norm(direction))
        scores[label]=float(np.clip(delta@direction/denominator,-1,1)) if denominator>1e-10 else None
    ranked=sorted(((k,v) for k,v in scores.items() if v is not None),key=lambda item:item[1],reverse=True)
    best=ranked[0][0] if ranked and ranked[0][1]>0 else None
    return {'scores':scores,'closest_emotion':best,'delta_norm':norm,
            'margin':ranked[0][1]-ranked[1][1] if len(ranked)>1 else None,
            'interpretation':'Exploratory cosine alignment with trained directions; not an emotion probability',
            'thresholds_approved':False}


def execute_integrated(config,output,*,backend_factory=None):
    from .vectors import normalized_difference,layer_metrics
    from .artifacts import runtime_metadata
    config=validate_integrated_request(config)
    rows=config['csv_rows'];train=[r for r in rows if r['row_type']=='train']
    tests=[r for r in rows if r['row_type']=='test']
    labels=list(dict.fromkeys(r['emotion'] for r in train));colors=emotion_colors(labels)
    if config['inference_prompt']:
        tests=[{'case_id':'inference','prompt':config['inference_prompt'],'neutral_prompt':config['inference_neutral'],'emotion':''}]
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    path=root/'experiment.json'
    result={'schema':'mm-emotion.remote-experiment.v1','experiment_version':2,'origin':'real_model_inference',
            'status':'running','configuration':config,'created_utc':datetime.now(timezone.utc).isoformat(),
            'runtime':runtime_metadata(),'emotions':labels,'emotion_colors':colors,
            'models':[],'results':[],'classifications':[],'failures':[],'course_results':[]}
    def persist():path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    persist()
    training_hash=hashlib.sha256(json.dumps(train,sort_keys=True).encode()).hexdigest()
    try:
        if backend_factory is None:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('Kaggle CUDA GPU unavailable')
            from .hf_backend import HFBackend
            backend_factory=HFBackend
        for model in config['models']:
            backend=None;started=time.perf_counter();context={'stage':'load_model'}
            try:
                kwargs={'precision':model['precision'],'max_length':512}
                if model.get('revision'):kwargs['revision']=model['revision']
                backend=backend_factory(model['model_id'],**kwargs)
                layer=config['layer'] if config['layer'] is not None else len(backend.layers)//2
                if layer not in range(len(backend.layers)):raise ValueError(f'Model has {len(backend.layers)} layers; requested {layer}')
                revision=getattr(backend.model.config,'_commit_hash',None) or model.get('revision')
                reuse=config['reuse_vectors'].get(model['alias'])
                context={'stage':'learn_vectors','layer_0based':layer}
                if reuse:
                    vector_source=Path(reuse['npz']);metadata=json.loads(Path(reuse['metadata']).read_text(encoding='utf8'))
                    for key,value in [('model_id',model['model_id']),('precision',model['precision']),('training_sha256',training_hash),('revision',revision)]:
                        if metadata.get(key)!=value:raise ValueError(f'Native vector mismatch: {key}')
                    if hashlib.sha256(vector_source.read_bytes()).hexdigest()!=metadata['vector_sha256']:raise ValueError('Native vector checksum mismatch')
                    with np.load(vector_source,allow_pickle=False) as data:
                        directions=data['directions'];layers=data['layers'].astype(int).tolist()
                        positive=data['train_positive'];neutral=data['train_neutral'];train_labels=data['train_labels'].astype(str).tolist()
                        if data['emotions'].astype(str).tolist()!=labels:raise ValueError('Native emotion labels differ')
                    if layer not in layers:raise ValueError('Steering must use the layer where these vectors were learned')
                else:
                    layers=sorted(set([max(0,len(backend.layers)//4),layer,min(len(backend.layers)-1,3*len(backend.layers)//4)]))
                    positive=np.stack([backend.capture(r['prompt'],layers) for r in train])
                    neutral=np.stack([backend.capture(r['neutral_prompt'],layers) for r in train])
                    train_labels=[r['emotion'] for r in train]
                    directions=np.stack([normalized_difference(positive[np.array(train_labels)==label],neutral[np.array(train_labels)==label]) for label in labels])
                if not np.isfinite(directions).all() or np.any(np.linalg.norm(directions,axis=-1)<1e-8):raise ValueError('A learned emotion direction is zero or nonfinite')
                position=layers.index(layer);native=directions[:,position,:]
                (root/'vectors').mkdir(exist_ok=True);vector=root/'vectors'/(model['alias']+'.npz')
                np.savez_compressed(vector,directions=directions,layers=np.asarray(layers),emotions=np.asarray(labels),
                    train_positive=positive,train_neutral=neutral,train_labels=np.asarray(train_labels))
                vector_hash=hashlib.sha256(vector.read_bytes()).hexdigest()
                metadata={'schema':'mm-emotion.multi-emotion-vector.v2','origin':'real_model_inference','model_id':model['model_id'],
                    'precision':model['precision'],'revision':revision,'emotions':labels,'layers_0based':layers,
                    'selected_layer':layer,'training_sha256':training_hash,'vector_sha256':vector_hash,
                    'evaluation':'training_only_vectors; separate CSV test prompts'}
                vector.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
                raw=[];references=[];case_ids=[]
                for case_index,case in enumerate(tests):
                    context={'stage':'neutral_subtraction','case_id':case['case_id'],'prompt':case['prompt'],'layer_0based':layer}
                    activation=backend.capture(case['prompt'],layers);reference=backend.capture(case['neutral_prompt'],layers)
                    raw.append(activation);references.append(reference);case_ids.append(case['case_id'])
                    classification=score_emotions(activation[position]-reference[position],native,labels)
                    result['classifications'].append({**case,**classification,'model_alias':model['alias'],
                        'model_id':model['model_id'],'precision':model['precision'],'layer_0based':layer,
                        'neutral_reference':case['neutral_prompt'],'origin':'real_model_inference'})
                    zero={label:0. for label in labels};conditions=[('baseline',zero)]
                    if case_index==0 and config['analysis_sweeps'] and not config['inference_prompt']:
                        conditions += [(f'{label} {coefficient:+g}',{**zero,label:coefficient}) for label in labels for coefficient in (-3.,3.)]
                    if config['inference_prompt'] and any(config['steering_weights'].values()):
                        conditions.append(('steered',config['steering_weights']))
                    for condition,weights in conditions:
                        direction=sum((weights[label]*native[i] for i,label in enumerate(labels)),np.zeros(native.shape[1],dtype=np.float32))
                        coefficient=0. if condition=='baseline' else 1.
                        context={'stage':'generate','case_id':case['case_id'],'prompt':case['prompt'],'layer_0based':layer,
                                 'condition':condition,'steering_weights':weights,'seed':config['seed']}
                        generated=backend.generate(case['prompt'],layer,direction,coefficient,max_new_tokens=config['max_new_tokens'],
                            temperature=config['temperature'],seed=config['seed'],capture_layer=layer)
                        measured=np.asarray(generated['observed_prefill_activation'])
                        matched=score_emotions(measured-reference[position],native,labels)
                        result['results'].append({'origin':'real_model_generation','model_alias':model['alias'],'model_id':model['model_id'],
                            'precision':model['precision'],'revision':revision,'backend':'hf','case_id':case['case_id'],'prompt':case['prompt'],
                            'neutral_reference':case['neutral_prompt'],'layer_0based':layer,'condition':condition,'weights':weights,
                            'coefficient':coefficient,'vector_path':vector.relative_to(root).as_posix(),'vector_sha256':vector_hash,
                            'emotion_scores':matched['scores'],'closest_emotion':matched['closest_emotion'],'exception':None,**generated})
                        if any(block._forward_hooks for block in backend.layers):raise RuntimeError('A steering hook was not removed')
                        persist()
                (root/'captures').mkdir(exist_ok=True)
                np.savez_compressed(root/'captures'/(model['alias']+'.npz'),positive=positive,neutral=neutral,train_labels=np.asarray(train_labels),
                    raw=np.stack(raw),reference=np.stack(references),delta=np.stack(raw)-np.stack(references),
                    case_ids=np.asarray(case_ids),directions=directions,layers=np.asarray(layers),emotions=np.asarray(labels))
                result['models'].append({**model,'status':'completed','seconds':time.perf_counter()-started,'resolved_revision':revision,
                    'selected_layer':layer,'scanned_layers':layers,'vector_path':vector.relative_to(root).as_posix(),
                    'vector_sha256':vector_hash,'vector_reused':bool(reuse),
                    'peak_vram_bytes':int(backend.torch.cuda.max_memory_allocated()) if hasattr(backend,'torch') else None})
            except Exception as exc:
                result['failures'].append({**model,**context,'exception':repr(exc)})
                result['models'].append({**model,'status':'failed','seconds':time.perf_counter()-started})
            finally:
                if backend is not None:backend.close()
                persist()
        result['status']='completed' if not result['failures'] else ('partial' if result['results'] else 'failed')
    except Exception as exc:result['status']='failed';result['failures'].append({'exception':repr(exc)})
    finally:persist()
    return path
