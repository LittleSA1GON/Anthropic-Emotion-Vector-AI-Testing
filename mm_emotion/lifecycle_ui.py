"""Reactive Streamlit monitoring and a CPU-only, Blender-authored scene."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from .lifecycle import compute_snapshot,LABELS,STATES
from .kaggle_runner import refresh_job,download_job,save_job

ROOT=Path(__file__).resolve().parents[1]
_scene=components.declare_component('model_mountain_neural_scene',path=str(ROOT/'assets/neural'))


@st.cache_data(show_spinner=False)
def asset_template():
    return json.loads((ROOT/'assets/neural/neural_scene.json').read_text(encoding='utf8'))


def render_scene(snapshot,*,key,selected_layer=None,inspectable_layers=None):
    return _scene(snapshot=snapshot,asset=asset_template(),selected_layer=selected_layer,
                  inspectable_layers=inspectable_layers or [],default=None,key=key)


def render_status(snapshot,job=None):
    mode={'live':'Live status','recorded':'Recorded data','demo':'Demonstration','test':'Test backend'}.get(snapshot['mode'],snapshot['mode'])
    cols=st.columns([2,2,2]);cols[0].caption(mode);cols[0].write('**'+LABELS[snapshot['state']]+'**')
    cols[1].caption('Connection');cols[1].write(snapshot['connection'])
    cols[2].caption('GPU evidence')
    gpu=snapshot.get('gpu')
    if gpu:
        cols[2].write(gpu.get('name','Device reported'))
        if gpu.get('total_memory_bytes'):cols[2].caption(f"{gpu['total_memory_bytes']/2**30:.1f} GiB total · measured at runtime")
    else:cols[2].write('Not verified')
    st.caption(snapshot['message'])
    if job and job.get('created_utc') and snapshot['mode']=='live':
        created=datetime.fromisoformat(job['created_utc']);elapsed=max(0,(datetime.now(timezone.utc)-created).total_seconds())
        st.caption(f'Elapsed since submission: {int(elapsed//60)}m {int(elapsed%60)}s. This is not a queue or completion estimate.')


def idle_workspace():
    with st.expander('Compute & visualization',expanded=True):
        demo=st.toggle('Demonstration mode — synthetic, no GPU',False,key='lab_demo')
        snapshot=compute_snapshot()
        if st.session_state.get('verified_kaggle_username')==st.session_state.get('account_username') and st.session_state.get('account_username'):
            snapshot['connection']='Credential checked; no connected runtime'
        if demo:
            stage=st.selectbox('Demonstration stage',STATES,index=STATES.index('vector_subtraction'),format_func=lambda s:LABELS[s])
            snapshot.update(mode='demo',state=stage,connection='No Kaggle session',message='Illustrative scene only. No model is loaded and no experimental results are generated.',
                architecture={'layers':12,'hidden_dim':64})
        render_status(snapshot)
        render_scene(snapshot,key='idle_neural_scene')
        st.caption('GPU assignment is externally controlled. Submitting a notebook requests compute; accelerator selection alone does not verify a device.')
        st.link_button('Kaggle accelerator documentation','https://www.kaggle.com/docs/notebooks')


def monitor(job,inspector):
    """Only this fragment polls; native setup controls and browser page stay mounted."""
    active=job['status'] not in {'completed','complete','failed','error','partial','cancelled','canceled'}
    @st.fragment(run_every=8 if active else None)
    def panel():
        current=json.loads((Path(job['directory'])/'job.json').read_text(encoding='utf8'))
        terminal=current['status'] in {'completed','failed','partial'} and current.get('result_path')
        a,b=st.columns([3,1])
        if terminal:a.caption('Recorded output loaded. Periodic polling has stopped.');automatic=False
        else:automatic=a.toggle('Check remote status automatically',value=active,key=job['ref']+'_auto')
        requested=b.button('Retry / fetch',key=job['ref']+'_refresh')
        if requested or (automatic and not terminal and time.time()-st.session_state.get(job['ref']+'_last_poll',0)>=7):
            st.session_state[job['ref']+'_last_poll']=time.time()
            try:
                refresh_job(current,token=st.session_state.get('account_token'))
                if current['status'] in {'complete','error'}:
                    try:download_job(current,token=st.session_state.get('account_token'))
                    except Exception as exc:
                        current['output_message']=str(exc);save_job(current)
            except Exception:pass # refresh_job records the connection failure without erasing status.
        result=None
        if current.get('result_path'):
            try:result=json.loads(Path(current['result_path']).read_text(encoding='utf8'))
            except (ValueError,OSError) as exc:st.error('The downloaded recording could not be read: '+str(exc))
        snapshot=compute_snapshot(current,result)
        render_status(snapshot,current)
        if not result:
            render_scene(snapshot,key=job['ref']+'_waiting_scene')
            st.info('This API connection exposes notebook status, not Kaggle’s scheduler or a live model telemetry stream. Detailed runtime stages are recorded inside the job and become replayable when Kaggle publishes its outputs.')
            if current.get('output_message'):st.caption(current['output_message'])
            st.link_button('Open private notebook and accelerator settings',current['url'])
        else:
            inspector(current,result,snapshot)
        # Stop the periodic fragment after one terminal result transition. This
        # is a Streamlit render, not a browser-page reload or a fabricated stage.
        if active and terminal and not st.session_state.get(job['ref']+'_settled'):
            st.session_state[job['ref']+'_settled']=True;st.rerun()
    panel()


def replay_controls(result,snapshot,alias):
    events=[e for e in result.get('lifecycle_events',[]) if e.get('mode')!='test' and e.get('details',{}).get('model_alias',alias)==alias]
    if events:
        view=st.selectbox('Recorded experiment timeline',list(range(len(events))),index=len(events)-1,
            format_func=lambda i:f"{events[i]['elapsed_seconds']:.1f}s · {LABELS[events[i]['state']]} · {events[i]['message']}",key='timeline_'+alias)
        event=events[view];snapshot={**snapshot,'state':event['state'],'message':event['message']}
        with st.expander('Exact recorded events'):
            st.dataframe(pd.DataFrame([{k:e.get(k) for k in ['sequence','timestamp_utc','elapsed_seconds','state','source','message']} for e in events]),hide_index=True,width='stretch')
    return snapshot
