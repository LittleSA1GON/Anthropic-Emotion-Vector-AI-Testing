"""Bounded weight inspection and exact, model-native activation arithmetic."""
from pathlib import Path
import numpy as np


def describe_backend(backend):
    config=backend.model.config
    text=getattr(config,'text_config',config)
    count=None
    if hasattr(backend.model,'num_parameters'):count=int(backend.model.num_parameters())
    def value(*names):
        return next((getattr(text,n) for n in names if getattr(text,n,None) is not None),None)
    return {'model_id':backend.model_id,'format':'Hugging Face / PyTorch',
            'model_type':value('model_type'),'layers':len(backend.layers),
            'hidden_dim':value('hidden_size','n_embd','d_model','word_embed_proj_dim'),
            'attention_heads':value('num_attention_heads','n_head'),
            'intermediate_dim':value('intermediate_size','ffn_dim'),
            'parameter_count':count,'parameter_count_method':'transformers.num_parameters' if count else None,
            'precision':backend.precision,'device':str(backend.device),
            'capabilities':{'generation':True,'activation_capture':True,'steering':True,
                            'weight_inspection':backend.precision!='nf4'},
            'components':[{'layer':i,'attention':next((type(getattr(block,n)).__name__ for n in ['self_attn','attn','attention'] if hasattr(block,n)),None),
                           'mlp':next((type(getattr(block,n)).__name__ for n in ['mlp','feed_forward','fc1'] if hasattr(block,n)),None)}
                          for i,block in enumerate(backend.layers)]}


def inspect_weights(backend, layers, *, sample_limit=2048):
    """Uniform samples of at most two parameter tensors per requested layer.

    NF4 packed storage is not a floating-point model weight and is not displayed.
    This never dequantizes an entire checkpoint for presentation.
    """
    if not 1<=sample_limit<=4096:raise ValueError('Weight sample budget must be 1-4096')
    if backend.precision=='nf4':
        return {'available':False,'reason':'NF4 floating-point weights are unavailable without dequantization; packed storage is not shown as weights','tensors':[]}
    tensors=[];torch=backend.torch
    for layer in layers:
        if layer not in range(len(backend.layers)):raise ValueError('Invalid layer')
        used=set()
        for name,param in backend.layers[layer].named_parameters():
            if param.ndim<2 or not param.is_floating_point() or not param.is_contiguous():continue
            kind='attention' if any(n in name for n in ['attn','attention']) else 'mlp' if any(n in name for n in ['mlp','feed_forward','fc1','fc2']) else 'other'
            if kind in used:continue
            flat=param.detach().view(-1);n=int(flat.numel());count=min(n,sample_limit)
            indices=torch.linspace(0,n-1,count,device=flat.device,dtype=torch.long)
            sampled=flat[indices].float().cpu().numpy().astype(np.float64)
            if not np.isfinite(sampled).all():raise ValueError('Nonfinite weight sample')
            tensors.append({'layer':layer,'name':name,'component':kind,'shape':list(param.shape),'dtype':str(param.dtype),
                'tensor_elements':n,'sample_count':count,'sampling':'uniform indices across the flattened tensor',
                'sample_mean':float(sampled.mean()),'sample_std':float(sampled.std()),
                'sample_min':float(sampled.min()),'sample_max':float(sampled.max()),
                'sample_values':sampled[:64].tolist(),'sample_indices':indices[:64].cpu().tolist()})
            used.add(kind)
            if len(used)>=2:break
    return {'available':bool(tensors),'reason':None if tensors else 'No supported floating-point parameter matrices were exposed','tensors':tensors}


def layer_data(result_path, alias, layer):
    root=Path(result_path).parent.resolve();path=(root/'captures'/(alias+'.npz')).resolve()
    if not path.is_relative_to(root):raise ValueError('Invalid capture path')
    with np.load(path,allow_pickle=False) as data:
        layers=data['layers'].astype(int).tolist();index=layers.index(layer)
        return {'layers':layers,'positive':data['positive'][:,index].astype(np.float64),
                'neutral':data['neutral'][:,index].astype(np.float64),'labels':data['train_labels'].astype(str).tolist(),
                'emotions':data['emotions'].astype(str).tolist(),'directions':data['directions'][:,index].astype(np.float64),
                'raw':data['raw'][:,index].astype(np.float64),'reference':data['reference'][:,index].astype(np.float64),
                'case_ids':data['case_ids'].astype(str).tolist()}


def extraction_values(data, emotion):
    mask=np.array(data['labels'])==emotion
    if mask.sum()<2:raise ValueError('Need at least two matched pairs')
    positive=data['positive'][mask];neutral=data['neutral'][mask]
    if positive.shape!=neutral.shape or positive.ndim!=2:raise ValueError('Expected [pairs, hidden_dim]')
    emotion_mean=positive.mean(axis=0);neutral_mean=neutral.mean(axis=0);raw=emotion_mean-neutral_mean
    norm=float(np.linalg.norm(raw))
    if not np.isfinite(raw).all() or norm<1e-12:raise ValueError('Invalid or zero training contrast')
    direction=raw/norm;native=data['directions'][data['emotions'].index(emotion)]
    return {'emotion_mean':emotion_mean,'neutral_mean':neutral_mean,'raw_direction':raw,'unit_direction':direction,
            'raw_norm':norm,'pairs':int(mask.sum()),'hidden_dim':int(raw.size),
            'native_matches':bool(np.allclose(native,direction,rtol=1e-5,atol=1e-6))}


def steering_values(baseline, observed, directions, weights, labels):
    """Compare a recorded hook output with its requested sum of native vectors."""
    baseline=np.asarray(baseline,dtype=np.float64);observed=np.asarray(observed,dtype=np.float64)
    directions=np.asarray(directions,dtype=np.float64)
    if directions.shape!=(len(labels),baseline.size) or observed.shape!=baseline.shape:raise ValueError('Steering dimensions differ')
    requested=np.array([weights.get(label,0) for label in labels])@directions
    actual=observed-baseline
    return {'requested_delta':requested,'observed_delta':actual,'residual':actual-requested,
            'residual_norm':float(np.linalg.norm(actual-requested)),
            'note':'Recorded first-prefill hook outputs; floating-point rounding can leave a residual.'}
