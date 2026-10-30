"""Model links and a single CSV contract for the integrated experiment lab."""
from __future__ import annotations
import csv
import hashlib
import io
import re
from urllib.parse import urlparse,unquote

COLORS={'happiness':'#b57b12','joy':'#b57b12','sadness':'#426bb2','anger':'#b74d43',
        'calm':'#32877d','anxiety':'#8264a5'}
PALETTE=['#b57b12','#426bb2','#b74d43','#32877d','#8264a5','#738744']
CSV_COLUMNS=['row_type','case_id','emotion','prompt','neutral_prompt','abstraction','diction','intensity',
             'expected_emotion','source_range_text','proposed_min','proposed_max','threshold_status']


def emotion_colors(labels):
    result={}
    for label in labels:
        preferred=COLORS.get(label)
        result[label]=preferred if preferred and preferred not in result.values() else next((c for c in PALETTE if c not in result.values()),PALETTE[len(result)%len(PALETTE)])
    return result


def parse_model_link(value):
    value=value.strip();revision=None
    if '://' in value:
        parsed=urlparse(value)
        if parsed.scheme!='https' or parsed.hostname not in {'huggingface.co','www.huggingface.co'} or parsed.username:
            raise ValueError('Use a Hugging Face model link, such as https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct')
        parts=[unquote(p) for p in parsed.path.split('/') if p]
        if len(parts)<2 or parts[0] in {'datasets','spaces','models','api'}:raise ValueError('This link is not a model checkpoint')
        if len(parts)>2:
            if len(parts)!=4 or parts[2]!='tree':raise ValueError('Link to the model page or a specific /tree/revision')
            revision=parts[3]
        value='/'.join(parts[:2])
    if not re.fullmatch(r'[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+',value):raise ValueError('Enter organization/model or a Hugging Face model link')
    if revision and not re.fullmatch(r'[a-zA-Z0-9._-]+',revision):raise ValueError('Unsupported model revision')
    return value,revision


def probe_model(value,*,get=None):
    """Read official Hub metadata; preflight is a candidate check, not GPU evidence."""
    if get is None:
        import requests
        get=requests.get
    model_id,revision=parse_model_link(value)
    metadata_url='https://huggingface.co/api/models/'+model_id
    if revision:metadata_url+='/revision/'+revision
    response=get(metadata_url,timeout=25)
    if response.status_code in (401,403):raise ValueError('Model metadata requires access. Start with a public checkpoint or configure model access.')
    response.raise_for_status();info=response.json()
    revision=info.get('sha')
    if not isinstance(revision,str) or not re.fullmatch(r'[a-fA-F0-9]{40}',revision):
        raise ValueError('The Hub did not return an immutable checkpoint revision. No GPU job was started.')
    config_response=get('https://huggingface.co/'+model_id+'/resolve/'+revision+'/config.json',timeout=25)
    config_response.raise_for_status();config=config_response.json()
    text=config.get('text_config',config)
    kind=text.get('model_type',config.get('model_type'))
    supported={'qwen2','qwen3','llama','mistral','gemma','gemma2','gemma3_text','phi3','phi','gpt2','opt','falcon','smollm3','olmo2'}
    layers=text.get('num_hidden_layers',text.get('n_layer'))
    if config.get('is_encoder_decoder') or kind not in supported or not isinstance(layers,int) or not 1<=layers<=200:
        raise ValueError('This architecture is not supported by the current decoder-hook backend. No GPU job was started.')
    size=info.get('safetensors',{}).get('total')
    precision='fp32' if model_id=='Qwen/Qwen2.5-1.5B-Instruct' else ('nf4' if size and size>2_000_000_000 else 'fp16')
    return {'alias':'m'+hashlib.sha256(model_id.encode()).hexdigest()[:10],'model_id':model_id,
            'revision':revision,'precision':precision,'layers':layers,'architecture':kind,
            'parameters':size,'gated':bool(info.get('gated')),'preflight':'candidate'}


def csv_template():
    from .course_tests import training_pairs,starter_cases
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for i,row in enumerate(training_pairs(),1):
        writer.writerow({'row_type':'train','case_id':f'train-{i:03d}','emotion':row['emotion'],
                         'prompt':row['emotion_prompt'],'neutral_prompt':row['neutral_prompt']})
    for case in starter_cases():
        writer.writerow({'row_type':'test',**{k:case.get(k,'') for k in CSV_COLUMNS if k not in {'row_type','neutral_prompt'}},
                         'neutral_prompt':case['neutral_reference']})
    return stream.getvalue().encode('utf8')


def _prompt_key(prompt):
    # Compare formatting variants without rewriting the user's recorded input.
    return ' '.join(prompt.split())


def parse_experiment_csv(content):
    if isinstance(content,bytes):content=content.decode('utf-8-sig')
    if len(content)>2_000_000:raise ValueError('Use a CSV smaller than 2 MB')
    reader=csv.DictReader(io.StringIO(content))
    required={'row_type','case_id','emotion','prompt','neutral_prompt'}
    if not reader.fieldnames or not required<=set(reader.fieldnames):
        raise ValueError('CSV needs row_type, case_id, emotion, prompt and neutral_prompt. Download the supplied format.')
    rows=[]
    for i,row in enumerate(reader,2):
        clean={k:str(row.get(k) or '').strip() for k in CSV_COLUMNS}
        if not any(clean.values()):continue
        if clean['row_type'] not in {'train','test'}:raise ValueError(f'Line {i}: row_type must be train or test')
        if not clean['case_id'] or not clean['prompt'] or not clean['neutral_prompt']:raise ValueError(f'Line {i}: case_id, prompt and neutral_prompt are required')
        if len(clean['prompt'])>12000 or len(clean['neutral_prompt'])>12000:raise ValueError(f'Line {i}: prompt exceeds 12,000 characters')
        if clean['row_type']=='train' and (not re.fullmatch(r'[\w -]{1,40}',clean['emotion']) or _prompt_key(clean['prompt'])==_prompt_key(clean['neutral_prompt'])):
            raise ValueError(f'Line {i}: training rows need an emotion label and distinct emotional/neutral prompts')
        rows.append(clean)
        if len(rows)>200:raise ValueError('Use at most 200 CSV rows per job')
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('CSV case_id values must be unique')
    train=[r for r in rows if r['row_type']=='train'];tests=[r for r in rows if r['row_type']=='test']
    labels=list(dict.fromkeys(r['emotion'] for r in train))
    if not 1<=len(labels)<=6:raise ValueError('Provide one to six emotion labels in training rows')
    if any(len({(_prompt_key(r['prompt']),_prompt_key(r['neutral_prompt'])) for r in train if r['emotion']==label})<2 for label in labels):
        raise ValueError('Each emotion needs at least two distinct training pairs')
    if not 1<=len(tests)<=50:raise ValueError('Provide one to fifty test rows')
    training_text={_prompt_key(r[k]) for r in train for k in ('prompt','neutral_prompt')}
    if training_text.intersection(_prompt_key(r[k]) for r in tests for k in ('prompt','neutral_prompt')):
        raise ValueError('Training and test prompts must be separate; do not evaluate vectors on their own training prompts')
    return {'rows':rows,'training':train,'tests':tests,'emotions':labels}


def validate_integrated_request(request):
    allowed={'schema','models','csv_rows','layer','max_new_tokens','temperature','seed','steering_weights',
             'inference_prompt','inference_neutral','reuse_vectors','analysis_sweeps'}
    if set(request)-allowed:raise ValueError('Unexpected experiment fields; credentials are not part of a request')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=CSV_COLUMNS);writer.writeheader()
    for row in request.get('csv_rows',[]):writer.writerow({k:row.get(k,'') for k in CSV_COLUMNS})
    dataset=parse_experiment_csv(stream.getvalue())
    models=request.get('models',[])
    if not 1<=len(models)<=3:raise ValueError('Add one to three models')
    clean_models=[]
    for model in models:
        model_id,_=parse_model_link(model['model_id']);alias=model['alias']
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}',alias) or model.get('precision') not in {'fp16','fp32','nf4'}:raise ValueError('Invalid model settings')
        revision=model.get('revision')
        if revision is not None and not re.fullmatch(r'[a-zA-Z0-9._-]{1,100}',revision):raise ValueError('Invalid revision')
        clean_models.append({'alias':alias,'model_id':model_id,'precision':model['precision'],'revision':revision})
    if len({m['alias'] for m in clean_models})!=len(models):raise ValueError('Duplicate model aliases')
    layer=request.get('layer')
    if layer is not None and (type(layer) is not int or not 0<=layer<200):raise ValueError('Invalid layer')
    tokens=request.get('max_new_tokens',32);seed=request.get('seed',42);temperature=request.get('temperature',0.)
    if type(tokens) is not int or not 1<=tokens<=128:raise ValueError('Use 1-128 output tokens')
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError('Invalid seed')
    if not isinstance(temperature,(float,int)) or not 0<=temperature<=2:raise ValueError('Invalid temperature')
    weights=request.get('steering_weights',{})
    if set(weights)-set(dataset['emotions']) or any(not isinstance(v,(float,int)) or not -10<=v<=10 for v in weights.values()):raise ValueError('Each emotion strength must be between -10 and +10')
    if sum(abs(v) for v in weights.values())>30:raise ValueError('Combined absolute steering strength must not exceed 30')
    prompt=request.get('inference_prompt','').strip();neutral=request.get('inference_neutral','').strip()
    if prompt and (not neutral or len(prompt)>12000 or len(neutral)>12000):raise ValueError('Inference needs a prompt and neutral reference, each under 12,000 characters')
    train={_prompt_key(r[k]) for r in dataset['training'] for k in ('prompt','neutral_prompt')}
    if prompt and train.intersection([_prompt_key(prompt),_prompt_key(neutral)]):raise ValueError('Inference prompts must be separate from training')
    reuse=request.get('reuse_vectors',{})
    if set(reuse)-{m['alias'] for m in models}:raise ValueError('Unknown native vector attachment')
    for alias,entry in reuse.items():
        if set(entry)!={'npz','metadata'} or any(not re.fullmatch(r'inputs/[a-zA-Z0-9_-]+\.(npz|json)',v) for v in entry.values()):raise ValueError('Invalid vector attachment paths')
    return {'schema':'mm-emotion.integrated-request.v2','models':clean_models,'csv_rows':dataset['rows'],
            'layer':layer,'max_new_tokens':tokens,'temperature':float(temperature),'seed':seed,
            'steering_weights':{label:float(weights.get(label,0)) for label in dataset['emotions']},
            'inference_prompt':prompt,'inference_neutral':neutral,'reuse_vectors':reuse,
            'analysis_sweeps':bool(request.get('analysis_sweeps',True))}
