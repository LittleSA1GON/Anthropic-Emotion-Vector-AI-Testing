import json
from types import SimpleNamespace
import numpy as np
import pytest
from mm_emotion.lifecycle import EventJournal,compute_snapshot,read_events,confirm_cuda,STATES
from mm_emotion.inspection import extraction_values,steering_values,inspect_weights,describe_backend


def test_states_reject_invented_gpu_and_queue_signals(tmp_path):
    journal=EventJournal(tmp_path/'events.jsonl')
    with pytest.raises(ValueError):journal.emit('gpu_ready','selected GPU',source='client')
    with pytest.raises(ValueError):journal.emit('gpu_queued','assumed queue',source='client')
    with pytest.raises(ValueError):journal.emit('model_loading','not ready')
    journal.emit('gpu_ready','CUDA operation completed',cuda_verified=True,name='test device')
    for state in ['model_loading','weights_inspection','activation_capture','activation_averaging','vector_subtraction','emotion_steering','comparison','completed']:
        journal.emit(state,'confirmed '+state)
    assert journal.state=='completed' and len(read_events(journal.path))==9
    with journal.path.open('a') as f:f.write('{incomplete')
    assert len(read_events(journal.path))==9
    assert all(e['state'] in STATES for e in journal.events)


def test_remote_status_does_not_imply_gpu_allocation_and_disconnect_recovers():
    assert compute_snapshot()['state']=='idle'
    assert compute_snapshot({'status':'running'})['gpu'] is None
    assert compute_snapshot({'status':'running'})['state']=='gpu_requested'
    assert compute_snapshot({'status':'queued'})['state']=='idle'
    job={'status':'queued','status_source':'kaggle_status'}
    assert compute_snapshot(job)['state']=='gpu_queued'
    job['connection_error']='network timeout'
    assert compute_snapshot(job)['last_confirmed_state']=='gpu_queued'
    assert compute_snapshot(job)['state']=='disconnected'
    job.pop('connection_error')
    assert compute_snapshot(job)['state']=='gpu_queued'
    result={'status':'completed','runtime':{'gpu_names':['archived T4']},'models':[]}
    snapshot=compute_snapshot(job,result)
    assert snapshot['mode']=='recorded' and snapshot['gpu']['name']=='archived T4'
    assert compute_snapshot({'status':'error'})['state']=='error'


def test_gpu_probe_requires_successful_tensor_operation():
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    assert confirm_cuda(torch)['cuda_verified'] is False
    torch.cuda.is_available=lambda:True
    assert confirm_cuda(torch)['cuda_verified'] is False


def test_extraction_and_steering_exact_arrays():
    p=np.array([[3.,2.,1.],[5.,4.,3.],[0.,0.,0.]])
    n=np.array([[1.,1.,1.],[1.,1.,1.],[0.,0.,0.]])
    raw=p[:2].mean(0)-n[:2].mean(0);unit=raw/np.linalg.norm(raw)
    data={'positive':p,'neutral':n,'labels':['happy','happy','calm'],'emotions':['happy'], 'directions':unit[None]}
    extracted=extraction_values(data,'happy')
    assert extracted['native_matches'] and extracted['hidden_dim']==3 and extracted['pairs']==2
    assert np.allclose(extracted['raw_direction'],[3.,2.,1.])
    base=np.array([10.,20.,30.]);observed=base-2*unit
    steered=steering_values(base,observed,unit[None],{'happy':-2},['happy'])
    assert steered['residual_norm']<1e-12
    zero=steering_values(base,base,unit[None],{'happy':0},['happy'])
    assert np.array_equal(zero['observed_delta'],np.zeros(3))


def test_weight_summaries_are_real_bounded_and_do_not_mutate_parameters():
    import torch
    block=torch.nn.Module();block.self_attn=torch.nn.Linear(6,6,bias=False);block.mlp=torch.nn.Linear(6,12,bias=False)
    with torch.no_grad():block.self_attn.weight.copy_(torch.arange(36).reshape(6,6))
    before=block.self_attn.weight.clone()
    backend=SimpleNamespace(layers=[block],precision='fp32',torch=torch,model_id='test/model',device='cpu',
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=6,num_attention_heads=2,model_type='test')))
    samples=inspect_weights(backend,[0],sample_limit=8)
    assert samples['available'] and len(samples['tensors'])==2
    row=samples['tensors'][0]
    assert row['sample_count']==8 and row['sample_values']==[0.,5.,10.,15.,20.,25.,30.,35.]
    assert torch.equal(before,block.self_attn.weight)
    architecture=describe_backend(backend)
    assert architecture['layers']==1 and architecture['hidden_dim']==6
    backend.precision='nf4'
    assert inspect_weights(backend,[0])['available'] is False
