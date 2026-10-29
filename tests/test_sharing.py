import io
import zipfile
import pytest
from mm_emotion.credentials import model_token, load_kaggle_model_secret
from mm_emotion.packaging import source_files


def test_credentials_are_optional_and_explicit_wins(monkeypatch):
    monkeypatch.delenv('HF_TOKEN',raising=False)
    assert model_token() is None
    monkeypatch.setenv('HF_TOKEN',' env-token ')
    assert model_token()=='env-token'
    assert model_token(' session-token ')=='session-token'
    assert load_kaggle_model_secret() is True


def test_release_excludes_local_credentials_and_environments(tmp_path):
    for name in ['README.md','.env','.env.example','kaggle.json','.venv/lib/token.py',
                 'mm_emotion/core.py','docs/kaggle.json','docs/.env.local',
                 '.streamlit/secrets.toml','runs/private.json','dist/old.zip']:
        p=tmp_path/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text('sentinel')
    paths={p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}
    assert paths=={'README.md','.env.example','mm_emotion/core.py'}


def test_repeated_seeds_and_failure_metadata(monkeypatch):
    from mm_emotion.multi_model import run_models,parse_models
    from test_multi_model import FakeHF
    import mm_emotion.hf_backend as hf
    monkeypatch.setattr(hf,'HFBackend',FakeHF)
    result=run_models(parse_models('qwen05'),prompts=[{'id':'p','prompt':'hello'}],seeds=[42,43])
    assert [r['seed'] for r in result['results']]==[42,43]
    assert all(r['backend']=='hf' and r['exception'] is None for r in result['results'])
    def fail(*args,**kwargs):raise RuntimeError('load failure')
    monkeypatch.setattr(hf,'HFBackend',fail)
    failed=run_models(parse_models('qwen05'),prompts=[{'id':'p','prompt':'hello'}],continue_on_error=True)
    assert failed['origin']=='no_successful_inference'
    assert failed['failures'][0]['precision']=='fp16'


def test_sweep_native_model_and_zero_control(monkeypatch,tmp_path):
    import numpy as np
    from test_multi_model import FakeHF
    from mm_emotion.multi_model import run_models,parse_models
    from mm_emotion.workbench import save_quick_vector
    import mm_emotion.hf_backend as hf
    monkeypatch.setattr(hf,'HFBackend',FakeHF)
    model=parse_models('qwen05')
    vec=save_quick_vector(tmp_path,'unit',np.ones((1,3),dtype=np.float32),
        {'model_id':model[0]['model_id'],'precision':'fp16','origin':'real_model_inference','layers_0based':[5]})
    out=run_models(model,prompts=[{'id':'p','prompt':'held out'}],mode='sweep',
                   vector_paths={'qwen05':str(vec)},coefficients=[-3,3],seeds=[42,43])
    assert [r['coefficient'] for r in out['results']]==[0,-3,3,0,-3,3]
    wrong=[{**model[0],'model_id':'different/model'}]
    with pytest.raises(RuntimeError,match='Vector model ID differs'):
        run_models(wrong,prompts=[{'id':'p','prompt':'held out'}],mode='sweep',vector_paths={'qwen05':str(vec)})
    vec.write_bytes(vec.read_bytes()+b'tampered')
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        run_models(model,prompts=[{'id':'p','prompt':'held out'}],mode='sweep',vector_paths={'qwen05':str(vec)})


def test_blind_export_hides_conditions_and_never_overwrites(tmp_path):
    from mm_emotion.blinding import export_blinded
    payload={'results':[{'origin':'real_model_generation','prompt':'test','output':{'text':'response'},
                         'coefficient':c,'model_id':'test/model'} for c in [0,3,-3]]}
    target=export_blinded(payload,tmp_path/'blind')
    text=(target/'annotations.csv').read_text()
    assert 'coefficient' not in text and 'test/model' not in text
    import json
    assert len(json.loads((target/'condition_key_PRIVATE.json').read_text()))==3
    with pytest.raises(FileExistsError):export_blinded(payload,target)
    with pytest.raises(ValueError):export_blinded({'results':[]},tmp_path/'empty')
