import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mm_emotion.kaggle_runner import validate_request,prepare_job,submit_job,refresh_job,run_cli
from mm_emotion.remote_experiment import execute


def request():
    return {'models':[{'alias':'unit','model_id':'test/checkpoint','precision':'fp32'}],
        'pairs':[{'neutral_prompt':'plain A','emotion_prompt':'happy A'},{'neutral_prompt':'plain B','emotion_prompt':'happy B'}],
        'prompts':['Explain exam preparation.'],'neutral_reference':'An exam is scheduled.',
        'layers':[0,1],'intervention_layers':[1],'coefficients':[-3,3],'seeds':[42]}


def test_request_bounds_baseline_and_disjoint_data():
    config=validate_request(request());assert config['coefficients']==[0,-3,3]
    for change in [{'api_key':'secret'},{'intervention_layers':[4]},{'seeds':[42,42]},
                   {'prompts':['plain A']},{'coefficients':[float('nan')]},
                   {'pairs':[request()['pairs'][0]]}]:
        with pytest.raises(ValueError):validate_request({**request(),**change})
    big={**request(),'prompts':['test '+str(i) for i in range(10)],'seeds':[1,2,3,4,5]}
    with pytest.raises(ValueError,match='limit is 120'):validate_request(big)


def test_private_notebook_and_no_credentials_in_payload(tmp_path,monkeypatch):
    (tmp_path/'mm_emotion').mkdir();(tmp_path/'mm_emotion'/'x.py').write_text('public source')
    (tmp_path/'.env').write_text('DO_NOT_EMBED_THIS')
    monkeypatch.setenv('KAGGLE_API_TOKEN','SESSION_ONLY_TOKEN')
    job=prepare_job(tmp_path,'researcher',request(),job_id='mm-emotion-unit-job')
    upload=Path(job['directory'])/'upload'
    metadata=json.loads((upload/'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['id'].startswith('researcher/')
    serialized=''.join(p.read_text() for p in Path(job['directory']).rglob('*.json'))+(upload/'experiment.ipynb').read_text()
    assert 'SESSION_ONLY_TOKEN' not in serialized and 'DO_NOT_EMBED_THIS' not in serialized
    monkeypatch.setattr('mm_emotion.kaggle_runner.run_cli',lambda *a,**kw:'Kernel version 1 successfully pushed https://www.kaggle.com/code/researcher/server-assigned-slug')
    assert submit_job(job,token='PRIVATE')['status']=='submitted'
    assert job['ref']=='researcher/server-assigned-slug'
    with pytest.raises(ValueError):submit_job(job)
    monkeypatch.setattr('mm_emotion.kaggle_runner.run_cli',lambda *a,**kw:'has status "KernelWorkerStatus.COMPLETE"')
    assert refresh_job(job)['status']=='complete'


def test_cli_token_uses_subprocess_env_and_redacts_errors(monkeypatch):
    def fake_run(cmd,**kwargs):
        assert kwargs['env']['KAGGLE_API_TOKEN']=='SENSITIVE'
        assert 'SENSITIVE' not in cmd
        return SimpleNamespace(returncode=1,stdout='failure SENSITIVE',stderr='')
    monkeypatch.setattr('mm_emotion.kaggle_runner.subprocess.run',fake_run)
    with pytest.raises(RuntimeError,match=r'failure \[redacted\]'):run_cli(['kernels','list'],token='SENSITIVE')


class TinyBackend:
    def __init__(self,model_id,precision,max_length):
        self.model_id=model_id;self.precision=precision
        self.layers=[SimpleNamespace(_forward_hooks={}) for _ in range(2)]
        self.model=SimpleNamespace(config=SimpleNamespace(_commit_hash='unit-test-only'))
    def capture(self,prompt,layers):
        return np.array([[1.+('happy' in prompt),len(prompt)*.01,l+1.,.5] for l in layers],dtype=np.float32)
    def generate(self,prompt,layer,direction,coefficient,**kwargs):
        return {'text':'unit-test text','seed':kwargs['seed'],'new_tokens':3,'duration_seconds':.01,
                'scope':kwargs['intervention_scope'],'observed_prefill_activation':(self.capture(prompt,[layer])[0]+coefficient*direction).tolist(),
                'observed_prefill_layer_0based':layer,'activation_origin':'unit-test-only'}
    def close(self):pass


def test_remote_worker_persists_measured_contract_and_pca(tmp_path):
    # Explicit fake factory exercises orchestration only; never used as GPU evidence.
    path=execute(request(),tmp_path/'result',backend_factory=TinyBackend)
    result=json.loads(path.read_text())
    assert result['status']=='completed' and len(result['results'])==3
    assert [r['coefficient'] for r in result['results']]==[0,-3,3]
    from mm_emotion.activation_view import activation_points
    anchors,points,variance=activation_points(result,path,'unit',1,'Explain exam preparation.')
    assert len(points)==3 and len(anchors)==5 and sum(variance)<=1.00001
    assert all(np.isfinite(p['projection']) for p in points)
    class Failing(TinyBackend):
        def generate(self,*args,**kwargs):raise RuntimeError('generation failure')
    failed=json.loads(execute(request(),tmp_path/'failure',backend_factory=Failing).read_text())
    assert failed['status']=='failed' and failed['failures'][0]['prompt']=='Explain exam preparation.'
