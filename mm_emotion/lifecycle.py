"""Evidence-backed experiment events. No scheduler or GPU allocation estimates."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

STATES = ('idle','gpu_requested','gpu_queued','gpu_unavailable','gpu_ready','model_loading',
          'weights_inspection','activation_capture','activation_averaging','vector_subtraction',
          'emotion_steering','comparison','completed','error','disconnected')
RUNTIME_STATES = set(STATES[3:14]) - {'error','completed'}
LABELS = {s:s.replace('_',' ').capitalize() for s in STATES}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_message(value):
    value = re.sub(r'KGAT_[\w-]+|hf_[\w-]{15,}', '[redacted]', str(value))
    return value[:1200]


class EventJournal:
    """Append-only journal shared by worker instrumentation and recorded replay."""
    def __init__(self, path, *, mode='live'):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.mode=mode;self.events=read_events(self.path);self.state=self.events[-1]['state'] if self.events else 'idle'
        age=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(self.events[0]['timestamp_utc'])).total_seconds()) if self.events else 0
        self.started=time.monotonic()-age

    def emit(self, state, message, *, source='runtime', **details):
        if state not in STATES:raise ValueError('Unknown lifecycle state')
        if source not in {'runtime','client','kaggle_status','connection','demonstration'}:
            raise ValueError('Unknown event source')
        if self.mode not in {'demo','test'}:
            if state=='gpu_ready' and (source!='runtime' or details.get('cuda_verified') is not True):
                raise ValueError('GPU readiness needs a successful runtime CUDA operation')
            if state=='gpu_queued' and (source!='kaggle_status' or details.get('provider_status')!='queued'):
                raise ValueError('Queue state must come from the provider')
            if state in RUNTIME_STATES and source!='runtime':
                raise ValueError('This lifecycle stage requires runtime evidence')
            if state in {'model_loading','weights_inspection','activation_capture','activation_averaging',
                         'vector_subtraction','emotion_steering','comparison'} and not any(e['state']=='gpu_ready' for e in self.events):
                raise ValueError('Model stages require prior verified GPU readiness')
        # Serialize before changing state; no partial event if a value is invalid.
        event={'schema':'mm-emotion.lifecycle-event.v1','sequence':len(self.events),
               'timestamp_utc':utc_now(),'elapsed_seconds':round(time.monotonic()-self.started,4),
               'state':state,'source':source,'mode':self.mode,'message':clean_message(message),
               'details':details}
        line=json.dumps(event,ensure_ascii=False,allow_nan=False)
        needs_newline=self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n')
        with self.path.open('a',encoding='utf8') as stream:
            stream.write(('\n' if needs_newline else '')+line+'\n');stream.flush()
        self.events.append(event);self.state=state
        return event


def read_events(path):
    """Ignore a final incomplete line while a local worker is still writing."""
    events=[]
    if not Path(path).is_file():return events
    for line in Path(path).read_text(encoding='utf8').splitlines():
        try:
            event=json.loads(line)
            if event.get('schema')=='mm-emotion.lifecycle-event.v1' and event.get('state') in STATES:events.append(event)
        except ValueError:continue
    return events


def record_job_event(job,state,message,*,source='client',**details):
    journal=EventJournal(Path(job['directory'])/'client-events.jsonl')
    if journal.events and journal.events[-1]['state']==state:return
    journal.emit(state,message,source=source,**details)


def compute_snapshot(job=None, result=None):
    """One source of truth for badges, the scene, monitoring, and replay mode."""
    snapshot={'state':'idle','mode':'live','connection':'Not checked','gpu':None,'architecture':None,
              'message':'No connected runtime. Choose a model and submit an experiment.',
              'events':[],'last_confirmed_state':None}
    if not job:return snapshot
    status=job.get('status','prepared')
    snapshot['connection']='Kaggle status API' if job.get('status_checked_utc') else 'Not checked'
    if status=='submitted':snapshot.update(state='gpu_requested',message='Notebook submitted. GPU allocation is not observable yet.')
    elif status=='queued' and (job.get('status_source')=='kaggle_status' or 'KernelWorkerStatus.QUEUED' in job.get('status_message','')):
        snapshot.update(state='gpu_queued',message='Kaggle reports QUEUED. No GPU has been verified by this client.')
    elif status=='running':snapshot.update(state='gpu_requested',message='Kaggle reports RUNNING. Runtime GPU and model stages are not available until outputs are published.')
    elif status in {'error','failed','cancelled','canceled'}:
        if job.get('status_source')=='submission':
            snapshot.update(state='error',message='Notebook submission failed; no runtime was confirmed. '+clean_message(job.get('submission_message','')))
        else:snapshot.update(state='error',message='Kaggle reports '+status.upper()+'. Open job details for its failure or quota message.')
    elif status in {'complete','completed','partial'}:
        snapshot.update(state='completed',message='Job finished. Fetch its outputs to inspect recorded runtime measurements.')
    if result:
        events=result.get('lifecycle_events',[])
        snapshot.update(mode='recorded',events=events,connection='Recorded output; not a connected GPU session')
        if events:
            snapshot.update(state=events[-1]['state'],message=events[-1]['message'])
            ready=[e for e in events if e['state']=='gpu_ready' and e.get('details',{}).get('cuda_verified')]
            if ready:snapshot['gpu']=ready[-1]['details']
        else:
            snapshot['state']='error' if result.get('status')=='failed' else 'completed'
            snapshot['message']='Recorded experiment. Detailed lifecycle events were not collected by this version.'
            names=result.get('runtime',{}).get('gpu_names',[])
            if names:snapshot['gpu']={'name':names[0],'verification':'reported by archived runtime; no live device connection'}
        complete=[m for m in result.get('models',[]) if m.get('status')=='completed']
        if complete:snapshot['architecture']=complete[0].get('architecture')
    if job.get('connection_error'):
        snapshot.update(last_confirmed_state=snapshot['state'],state='disconnected',connection='Status request failed',
                        message=clean_message(job['connection_error']))
    return snapshot


def confirm_cuda(torch):
    """Verify a real CUDA operation, not just an accelerator selection flag."""
    if not torch.cuda.is_available():return {'cuda_verified':False,'reason':'CUDA is not available in this runtime'}
    try:
        probe=torch.ones(1,device='cuda');value=float((probe+1).item());del probe
        if value!=2.:raise RuntimeError('CUDA operation returned an unexpected value')
        device=torch.cuda.current_device();props=torch.cuda.get_device_properties(device)
        free,total=torch.cuda.mem_get_info(device)
        return {'cuda_verified':True,'name':props.name,'total_memory_bytes':int(total),
                'free_memory_bytes_at_probe':int(free),'device_index':int(device),
                'verification':'torch CUDA tensor operation','observed_utc':utc_now()}
    except Exception as exc:return {'cuda_verified':False,'reason':clean_message(exc)}
