import json
import numpy as np
import pytest
from mm_emotion.multi_model import parse_models,read_prompts,run_models
from mm_emotion.course_tests import starter_cases

class FakeHF:
    currently=0; max_concurrent=0
    def __init__(self,model_id,precision,max_length):
        FakeHF.currently+=1;FakeHF.max_concurrent=max(FakeHF.currently,FakeHF.max_concurrent)
        self.model_id=model_id;self.precision=precision;self.layers=list(range(10))
        self.model=type('Model',(),{'cpu':lambda _:None, 'config':type('Cfg',(),{'hidden_size':3})()})()
    def generate(self,prompt,*args,**kwargs):return {'text':prompt,'seed':kwargs.get('seed')}
    def capture(self,prompt,layers):return np.ones((len(layers),3),dtype=np.float32)
    def __del__(self):FakeHF.currently-=1

def test_aliases_and_prompt_loaders(tmp_path):
    assert [x['alias'] for x in parse_models('qwen05,qwen15')]==['qwen05','qwen15']
    with pytest.raises(ValueError):parse_models('qwen05,qwen05')
    with pytest.raises(ValueError):parse_models('unknown')
    p=tmp_path/'prompts.txt';p.write_text('one\ntwo\n',encoding='utf8')
    assert len(read_prompts(p))==2

def test_sequential_direct_inference(monkeypatch):
    import mm_emotion.hf_backend as module
    monkeypatch.setattr(module,'HFBackend',FakeHF)
    FakeHF.currently=0;FakeHF.max_concurrent=0
    models=parse_models('qwen05,qwen15')
    out=run_models(models,prompts=[{'id':'1','prompt':'hello'}],mode='infer',seed=42)
    assert len(out['results'])==2
    assert all(r['coefficient']==0 for r in out['results'])
    assert all(r['origin']=='real_model_generation' for r in out['results'])
    assert FakeHF.max_concurrent<=1

def test_matrix_reports_ungraded_without_threshold_review(monkeypatch):
    import mm_emotion.hf_backend as module
    from mm_emotion.course_tests import evaluate_model as evaluator
    monkeypatch.setattr(module,'HFBackend',FakeHF)
    # Mock nonzero unit vectors only for multi-model orchestration test.
    def fake_evaluator(backend,cases,train_pairs,layer,**kwargs):
        return [dict(c, model_id=backend.model_id, precision=backend.precision,
                     expected_cosine=.5,status='ungraded_provisional_thresholds',origin='real_model_inference') for c in cases]
    import mm_emotion.multi_model as mm
    monkeypatch.setattr(mm,'evaluate_model',fake_evaluator)
    out=run_models(parse_models('qwen05'),cases=starter_cases(),mode='matrix')
    assert len(out['results'])==18
    assert out['stats']['status_counts']['ungraded_provisional_thresholds']==18
