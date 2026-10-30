"""Private Kaggle GPU jobs for the local experiment UI.

Credentials are passed only through the CLI subprocess environment, never into
notebooks, request JSON, status files, or results. Existing local Kaggle login
is used when a session token is not supplied.
"""
from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime,timezone
import numpy as np
from .packaging import source_files
from .workbench import parse_pairs,sweep_grid,validate_test_prompts


def validate_request(request):
    if isinstance(request,dict) and request.get('schema')=='mm-emotion.integrated-request.v2':
        from .experiment_inputs import validate_integrated_request
        return validate_integrated_request(request)
    allowed={'models','pairs','prompts','neutral_reference','layers','intervention_layers',
             'coefficients','seeds','emotion','method','scope','max_new_tokens','max_length',
             'temperature','top_p','include_matrix'}
    if not isinstance(request,dict) or set(request)-allowed:
        raise ValueError('Unexpected experiment fields; credentials do not belong in requests')
    config=dict(request)
    models=config.get('models',[])
    if not 1<=len(models)<=3:raise ValueError('Choose one to three checkpoints')
    aliases=[];clean_models=[]
    for item in models:
        if set(item)-{'alias','model_id','precision'}:raise ValueError('Unexpected model fields')
        alias=item.get('alias','')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}',alias):raise ValueError('Invalid model alias')
        if not re.fullmatch(r'[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+',item.get('model_id','')):
            raise ValueError('Use a Hugging Face organization/checkpoint ID')
        if item.get('precision') not in {'fp16','fp32','nf4'}:raise ValueError('Unsupported precision')
        aliases.append(alias);clean_models.append(dict(item))
    if len(set(aliases))!=len(aliases):raise ValueError('Model aliases must be unique')
    config['models']=clean_models
    pairs=parse_pairs(json.dumps(config.get('pairs',[])),'json',max_rows=100)
    if len(pairs)<2:raise ValueError('At least two contrast pairs are required')
    # Transmit the prompts required for this experiment, excluding arbitrary extra columns.
    config['pairs']=[{k:p[k] for k in ('neutral_prompt','emotion_prompt')} for p in pairs]
    if any(len(p[k])>12000 for p in config['pairs'] for k in p):raise ValueError('Training prompt too long')
    prompts=config.get('prompts',[])
    if not isinstance(prompts,list) or any(not isinstance(p,str) for p in prompts):raise ValueError('Prompts must be text')
    config['prompts']=validate_test_prompts('\n'.join(prompts),max_prompts=10)
    reference=config.get('neutral_reference','').strip()
    if not reference or len(reference)>12000:raise ValueError('Enter a neutral reference prompt')
    config['neutral_reference']=reference
    train={p[k].strip() for p in config['pairs'] for k in p}
    if train.intersection(config['prompts']+[reference]):raise ValueError('Test/reference prompts must be separate from training pairs')
    for name in ('layers','intervention_layers'):
        vals=config.get(name,[])
        if not vals or len(vals)>12 or len(set(vals))!=len(vals) or any(type(v) is not int or not 0<=v<200 for v in vals):
            raise ValueError('Choose one to twelve unique zero-based layers')
    if not set(config['intervention_layers'])<=set(config['layers']):raise ValueError('Intervention layers must be scanned layers')
    config['coefficients']=sweep_grid(','.join(map(str,config.get('coefficients',[-3,0,3]))))
    seeds=config.get('seeds',[42])
    if not seeds or len(seeds)>5 or len(set(seeds))!=len(seeds) or any(type(v) is not int or not 0<=v<=2147483647 for v in seeds):
        raise ValueError('Choose one to five unique integer seeds')
    config['seeds']=seeds
    count=len(models)*len(config['prompts'])*len(config['intervention_layers'])*len(config['coefficients'])*len(seeds)
    if count>120:raise ValueError(f'{count} generations requested; limit is 120 per job')
    config.setdefault('max_new_tokens',48);config.setdefault('max_length',512)
    if type(config['max_new_tokens']) is not int or not 1<=config['max_new_tokens']<=256:raise ValueError('Use 1-256 generated tokens')
    if type(config['max_length']) is not int or not 128<=config['max_length']<=2048:raise ValueError('Input limit must be 128-2048')
    config.setdefault('temperature',0.);config.setdefault('top_p',1.)
    if not np.isfinite(config['temperature']) or not 0<=config['temperature']<=2:raise ValueError('Invalid temperature')
    if not np.isfinite(config['top_p']) or not 0<config['top_p']<=1:raise ValueError('Invalid top-p')
    config.setdefault('method','mean_diff');config.setdefault('scope','last_token');config.setdefault('emotion','custom')
    if config['method'] not in {'mean_diff','pairwise_pca'}:raise ValueError('Invalid vector method')
    if config['scope'] not in {'last_token','prefill_only','decode_only','all_tokens'}:raise ValueError('Invalid scope')
    if not isinstance(config['emotion'],str) or not config['emotion'].strip() or len(config['emotion'])>80:raise ValueError('Provide a short vector label')
    config.setdefault('include_matrix',False)
    if type(config['include_matrix']) is not bool:raise ValueError('Invalid matrix switch')
    return config


def prepare_job(root,username,request,*,job_id=None,attachments=None):
    root=Path(root).resolve()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{3,50}',username):raise ValueError('Enter your Kaggle username')
    config=validate_request(request)
    slug=job_id or ('mm-emotion-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    if not re.fullmatch(r'[a-z0-9-]{8,70}',slug):raise ValueError('Invalid job ID')
    directory=root/'runs'/'kaggle_ui'/slug
    upload=directory/'upload';upload.mkdir(parents=True,exist_ok=False)
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(source_files(root)):
            archive.writestr(p.relative_to(root).as_posix(),p.read_bytes())
        for name,source in (attachments or {}).items():
            if not re.fullmatch(r'inputs/[a-zA-Z0-9_-]+\.(npz|json)',name):raise ValueError('Invalid attachment filename')
            source=Path(source)
            if not source.is_file() or source.stat().st_size>50_000_000:raise ValueError('Invalid native-vector attachment')
            archive.writestr(name,source.read_bytes())
    payload=base64.b64encode(memory.getvalue()).decode('ascii')
    code="""import base64,io,zipfile,os,sys,subprocess,json
from pathlib import Path
project=Path('/kaggle/working/emotion_lab')
project.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    for name in archive.namelist():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe source path')
    archive.extractall(project)
os.chdir(project)
sys.path.insert(0,str(project))
subprocess.run([sys.executable,'-m','pip','install','-r','requirements-hf.txt'],check=True)
from mm_emotion.credentials import load_kaggle_model_secret
load_kaggle_model_secret()
from mm_emotion.remote_experiment import execute
path=execute(REQUEST,Path('/kaggle/working/results'))
record=json.loads(path.read_text())
print('EXPERIMENT_STATUS',record['status'],'GENERATIONS',len(record['results']))
if record['status']=='failed':raise RuntimeError(record['failures'])
""".replace('PAYLOAD',repr(payload)).replace('REQUEST',repr(config))
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'id':'experiment','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    (upload/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
    ref=username.lower()+'/'+slug
    metadata={'id':ref,'title':slug.replace('-',' '),'code_file':'experiment.ipynb',
              'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
              'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
    (upload/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    job={'schema':'mm-emotion.kaggle-job.v1','ref':ref,'url':'https://www.kaggle.com/code/'+ref,
         'directory':str(directory),'status':'prepared','configuration':config,
         'created_utc':datetime.now(timezone.utc).isoformat()}
    save_job(job)
    return job


def save_job(job):
    path=Path(job['directory'])/'job.json'
    path.write_text(json.dumps(job,indent=2,ensure_ascii=False),encoding='utf8')


def run_cli(arguments,*,token=None,timeout=120):
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    if token and token.strip():env['KAGGLE_API_TOKEN']=token.strip()
    proc=subprocess.run([sys.executable,'-m','kaggle',*arguments],env=env,capture_output=True,
                        text=True,encoding='utf8',errors='replace',timeout=timeout)
    output=(proc.stdout+'\n'+proc.stderr).strip()
    if token:output=output.replace(token,'[redacted]')
    output=re.sub(r'KGAT_[a-zA-Z0-9_-]+','[redacted]',output)
    if proc.returncode:raise RuntimeError(output[-5000:] or 'Kaggle CLI request failed')
    return output


def submit_job(job,*,token=None):
    if job['status']!='prepared':raise ValueError('This job has already been submitted; create a new request')
    text=run_cli(['kernels','push','-p',str(Path(job['directory'])/'upload'),'--accelerator','NvidiaTeslaT4'],token=token)
    if 'successfully pushed' not in text.lower():raise RuntimeError(text)
    # Kaggle may derive a different slug from the title; its returned URL is authoritative.
    created=re.search(r'https://www\.kaggle\.com/code/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)',text)
    if created:
        actual=created.group(1)
        if actual.split('/')[0].lower()!=job['ref'].split('/')[0].lower():
            raise RuntimeError('Kaggle returned a different account; check the username and API token together')
        job['ref']=actual;job['url']=created.group(0)
    job['status']='submitted';job['submission_message']=text;save_job(job)
    return job


def refresh_job(job,*,token=None):
    text=run_cli(['kernels','status',job['ref']],token=token,timeout=45)
    match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',text)
    state=match.group(1).lower() if match else 'unknown'
    job['status']=state;job['status_message']=text;save_job(job)
    return job


def download_job(job,*,token=None):
    destination=Path(job['directory'])/'outputs';destination.mkdir(exist_ok=True)
    run_cli(['kernels','output',job['ref'],'-p',str(destination),'--file-pattern',r'(^|/)results/','--quiet'],token=token,timeout=180)
    candidates=list(destination.rglob('experiment.json'))
    if len(candidates)!=1:raise RuntimeError('Experiment output is not available yet; refresh the job status and retry')
    path=candidates[0]
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('schema')!='mm-emotion.remote-experiment.v1':raise ValueError('Unexpected downloaded result schema')
    if any(r.get('origin')!='real_model_generation' for r in data.get('results',[])):raise ValueError('Invalid generation evidence')
    job['result_path']=str(path);job['status']=data['status'];save_job(job)
    return data


def result_zip(result_path):
    root=Path(result_path).parent;memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(root).as_posix())
    return memory.getvalue()
