import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mm_emotion.experiment_inputs import parse_model_link,probe_model,csv_template,parse_experiment_csv,validate_integrated_request,emotion_colors
from mm_emotion.integrated_experiment import execute_integrated,score_emotions


def config():
    rows=[]
    for emotion,word in [('happiness','happy'),('anger','angry')]:
        for i in range(2):rows.append({'row_type':'train','case_id':f'{emotion}-{i}','emotion':emotion,'prompt':f'{word} training situation {i}','neutral_prompt':f'plain situation {emotion} {i}'})
    rows += [{'row_type':'test','case_id':'test-happy','emotion':'happiness','prompt':'I am happy about exam results','neutral_prompt':'The exam results are available'},
             {'row_type':'test','case_id':'test-angry','emotion':'anger','prompt':'I am angry about a mistake','neutral_prompt':'A mistake was reported'}]
    return {'schema':'mm-emotion.integrated-request.v2','models':[{'alias':'unit','model_id':'test/model','precision':'fp32'}],
            'csv_rows':rows,'layer':1,'max_new_tokens':8,'analysis_sweeps':True}


def test_model_links_and_preflight_use_official_hub():
    assert parse_model_link('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main')==('Qwen/Qwen2.5-0.5B-Instruct','main')
    for link in ['https://example.com/a/b','https://huggingface.co/spaces/a','https://huggingface.co/a/b/resolve/main/model.bin']:
        with pytest.raises(ValueError):parse_model_link(link)
    urls=[]
    def get(url,**kwargs):
        urls.append(url)
        data={'sha':'a'*40,'safetensors':{'total':500_000_000}} if '/api/models/' in url else {'model_type':'qwen2','num_hidden_layers':24}
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:data)
    model=probe_model('Qwen/Qwen2.5-0.5B-Instruct',get=get)
    assert model['revision']=='a'*40 and model['layers']==24 and model['preflight']=='candidate'
    assert all(u.startswith('https://huggingface.co/') for u in urls)


def test_branch_preflight_pins_metadata_and_config_to_same_commit():
    urls=[]
    def get(url,**kwargs):
        urls.append(url)
        data={'sha':'b'*40} if '/api/models/' in url else {'model_type':'qwen2','num_hidden_layers':24}
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:data)
    model=probe_model('https://huggingface.co/test/model/tree/release',get=get)
    assert urls==['https://huggingface.co/api/models/test/model/revision/release',
                  'https://huggingface.co/test/model/resolve/'+'b'*40+'/config.json']
    assert model['revision']=='b'*40
    def missing_sha(url,**kwargs):
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:{})
    with pytest.raises(ValueError,match='immutable'):probe_model('test/model',get=missing_sha)


def test_csv_template_preserves_original_18_classifications():
    dataset=parse_experiment_csv(csv_template())
    assert len(dataset['training'])==12 and len(dataset['tests'])==18
    assert set(dataset['emotions'])=={'happiness','sadness','anger'}
    assert all(r['threshold_status']=='proposed_not_approved' for r in dataset['tests'])
    duplicate=csv_template().decode().replace('hap-obj-exp','hap-obj-sta')
    with pytest.raises(ValueError,match='unique'):parse_experiment_csv(duplicate)
    bad=config();bad['csv_rows'][-1]['prompt']=bad['csv_rows'][0]['prompt']
    with pytest.raises(ValueError,match='separate'):validate_integrated_request(bad)
    with pytest.raises(ValueError):validate_integrated_request({**config(),'api_key':'secret'})


def test_duplicate_pairs_and_formatting_leakage_are_rejected():
    bad=config()
    for key in ['prompt','neutral_prompt']:bad['csv_rows'][1][key]=bad['csv_rows'][0][key].replace(' ','  ')
    with pytest.raises(ValueError,match='distinct'):validate_integrated_request(bad)
    bad=config();bad['csv_rows'][-1]['prompt']=bad['csv_rows'][0]['prompt'].replace(' ','  ')
    with pytest.raises(ValueError,match='separate'):validate_integrated_request(bad)
    bad=config();bad.update(inference_prompt=bad['csv_rows'][0]['prompt'].replace(' ','\t'),inference_neutral='A separate neutral reference')
    with pytest.raises(ValueError,match='separate'):validate_integrated_request(bad)


class MultiBackend:
    def __init__(self,model_id,precision,max_length,revision=None):
        self.model_id=model_id;self.precision=precision;self.layers=[SimpleNamespace(_forward_hooks={}) for _ in range(4)]
        self.model=SimpleNamespace(config=SimpleNamespace(_commit_hash='unit-rev'))
    def capture(self,prompt,layers):
        return np.array([[3*('happy' in prompt),3*('angry' in prompt),len(prompt)*.001,1+l*.01] for l in layers],dtype=np.float32)
    def generate(self,prompt,layer,direction,coefficient,**kwargs):
        return {'text':'unit-test only','seed':kwargs['seed'],'new_tokens':2,'duration_seconds':.1,'scope':'last_token',
                'observed_prefill_activation':(self.capture(prompt,[layer])[0]+coefficient*direction).tolist(),
                'observed_prefill_layer_0based':layer,'activation_origin':'unit-test-only'}
    def close(self):pass


def test_multi_emotion_detection_and_reused_mixed_steering(tmp_path,monkeypatch):
    request=config();path=execute_integrated(request,tmp_path/'discovery',backend_factory=MultiBackend)
    result=json.loads(path.read_text())
    assert result['status']=='completed' and len(result['classifications'])==2
    assert result['classifications'][0]['closest_emotion']=='happiness'
    assert result['classifications'][1]['closest_emotion']=='anger'
    assert len(result['results'])==6
    stages={e['state'] for e in result['lifecycle_events']}
    assert {'model_loading','activation_capture','activation_averaging','vector_subtraction','emotion_steering','comparison','completed'}<=stages
    assert 'gpu_ready' not in stages and all(e['mode']=='test' for e in result['lifecycle_events'])
    assert result['models'][0]['weight_summaries']['available'] is False
    assert all(r['thresholds_approved'] is False for r in result['classifications'])
    (tmp_path/'inputs').mkdir()
    import shutil
    for suffix in ['.npz','.json']:shutil.copy2(tmp_path/'discovery/vectors'/('unit'+suffix),tmp_path/'inputs'/('unit'+suffix))
    request.update(inference_prompt='Explain this happy new outcome',inference_neutral='A new outcome is reported',
                   steering_weights={'happiness':2,'anger':-1},reuse_vectors={'unit':{'npz':'inputs/unit.npz','metadata':'inputs/unit.json'}})
    monkeypatch.chdir(tmp_path)
    steered=json.loads(execute_integrated(request,tmp_path/'inference',backend_factory=MultiBackend).read_text())
    assert steered['status']=='completed' and steered['models'][0]['vector_reused'] is True
    assert [r['condition'] for r in steered['results']]==['baseline','steered']
    assert steered['results'][1]['weights']=={'happiness':2.,'anger':-1.}
    request['models'][0]['precision']='fp16'
    failed=json.loads(execute_integrated(request,tmp_path/'wrong',backend_factory=MultiBackend).read_text())
    assert failed['status']=='failed' and 'precision' in failed['failures'][0]['exception']
    assert failed['lifecycle_events'][-1]['state']=='error'


def test_zero_delta_remains_unscorable():
    score=score_emotions(np.zeros(3),np.eye(3),['a','b','c'])
    assert score['closest_emotion'] is None and all(v is None for v in score['scores'].values())
    assert len(set(emotion_colors(['happiness','joy','sadness','anger']).values()))==4
