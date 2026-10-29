"""Local UI for remote Kaggle GPU experiments using each researcher's account."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from mm_emotion.dataset import load_pairs
from mm_emotion.multi_model import REGISTRY
from mm_emotion.workbench import parse_pairs
from mm_emotion.kaggle_runner import prepare_job,submit_job,refresh_job,download_job,result_zip,run_cli,validate_request
from mm_emotion.ui_theme import apply_lab_theme

st.set_page_config(page_title='Kaggle Emotion Experiment Lab',page_icon='🔬',layout='wide')
apply_lab_theme()
ROOT=Path(__file__).resolve().parents[1]
st.markdown('<p class="lab-eyebrow">MODEL MOUNTAIN / EXPERIMENT WORKBENCH</p>',unsafe_allow_html=True)
st.title('Emotion vector lab')
st.write('Discover and steer model-specific activation directions.')
st.caption('Your Kaggle account runs each experiment as a private GPU batch. Model outputs and measured activations return here.')

profile=ROOT/'runs'/'kaggle_ui'/'profile.json'
default_username=json.loads(profile.read_text(encoding='utf8')).get('username','') if profile.exists() else ''
with st.sidebar:
    st.header('Compute account')
    username=st.text_input('Kaggle username',value=default_username,key='kaggle_username')
    token=st.text_input('Kaggle API token',type='password',key='kaggle_api_token',
        help='Optional if you already configured the Kaggle CLI. This field stays in the current UI session and is never saved in experiment files.')
    st.caption('Use your own Kaggle token and username. A blank token uses your existing local Kaggle login.')
    if st.button('Check Kaggle credentials'):
        try:
            run_cli(['kernels','list','--mine','--page-size','1','--csv'],token=token,timeout=45)
            st.success('Kaggle accepted the current credential.')
        except Exception as exc:st.error(str(exc))
    st.divider()
    st.caption('Public Qwen and SmolLM checkpoints need no Hugging Face key. Gated weights require model access and an attached Kaggle HF_TOKEN secret; tokens are never embedded in the job source.')

setup,history=st.tabs(['New experiment','Results'])
with setup:
    st.subheader('01 / Model and training pairs')
    names=st.multiselect('Run these checkpoints sequentially',list(REGISTRY),default=['qwen05'],
        format_func=lambda n:f"{n} · {REGISTRY[n]['model_id']} · {REGISTRY[n]['precision']}")
    left,right=st.columns(2)
    with left:
        emotion=st.selectbox('Example emotion / vector hypothesis',['joy','calm','anger','anxiety'])
        label=st.text_input('Name your vector',emotion)
        method=st.selectbox('Vector extraction',['mean_diff','pairwise_pca'])
    with right:
        layers_text=st.text_input('Scan decoder layers (zero-based)','5,11,17')
        interventions_text=st.text_input('Intervene on these scanned layers, one at a time','5')
        scope=st.selectbox('Intervention scope',['last_token','prefill_only','decode_only','all_tokens'])
    rows=[r for r in load_pairs(ROOT/'data/emotion_pairs_v1.jsonl') if r['emotion']==emotion and r['split']=='train']
    table=pd.DataFrame([{k:r[k] for k in ('neutral_prompt','emotion_prompt')} for r in rows])
    upload=st.file_uploader('Import your own contrast pairs (CSV / JSON / JSONL)',type=['csv','json','jsonl'])
    if upload:
        try:table=pd.DataFrame(parse_pairs(upload.getvalue(),upload.name.rsplit('.',1)[-1]))[['neutral_prompt','emotion_prompt']]
        except Exception as exc:st.error(str(exc))
    edited=st.data_editor(table,num_rows='dynamic',hide_index=True,width='stretch',key='remote_pairs_'+emotion)
    st.caption('At least two neutral/emotional contrast pairs. These fit exploratory vectors; the training scores are not held-out accuracy.')
    st.subheader('02 / Prompt and intervention')
    prompts=st.text_area('Test prompts (one per line; separate from vector-training prompts)',
        'Explain how to approach an upcoming exam in two sentences.',height=95)
    reference=st.text_input('Neutral reference for activation projection','I am preparing for an upcoming exam.')
    a,b,c=st.columns(3)
    coefficients=a.text_input('Strength sweep (zero baseline is always included)','-3,0,3')
    seeds=b.text_input('Repeat with seeds','42')
    tokens=c.slider('Generated tokens',8,128,48,8)
    temperature=st.slider('Temperature (0 uses deterministic decoding)',0.,1.5,0.,.1)
    include_matrix=st.checkbox('Also execute the original 18-case course matrix per model (thresholds remain ungraded)',False)
    st.caption('Run sends the selected training pairs, test prompts, settings, and public lab source to a private notebook in your Kaggle account and uses your GPU quota. API tokens stay outside the notebook.')
    if st.button('Run on my Kaggle T4',type='primary',disabled=not names):
        try:
            pairs=edited.fillna('').to_dict('records')
            pairs=[p for p in pairs if p.get('neutral_prompt') or p.get('emotion_prompt')]
            config=validate_request({'models':[{'alias':n,'model_id':REGISTRY[n]['model_id'],'precision':REGISTRY[n]['precision']} for n in names],
                'pairs':pairs,'emotion':label,'method':method,'layers':[int(x) for x in layers_text.split(',') if x.strip()],
                'intervention_layers':[int(x) for x in interventions_text.split(',') if x.strip()],
                'prompts':[p for p in prompts.splitlines() if p.strip()],'neutral_reference':reference,
                'coefficients':[float(x) for x in coefficients.split(',') if x.strip()],
                'seeds':[int(x) for x in seeds.split(',') if x.strip()],
                'max_new_tokens':tokens,'temperature':temperature,'scope':scope,'include_matrix':include_matrix})
            with st.spinner('Uploading your private experiment through the Kaggle API...'):
                job=prepare_job(ROOT,username.strip(),config)
                submit_job(job,token=token)
            st.session_state['remote_job_ref']=job['ref']
            st.success('Submitted. Open Results to monitor and retrieve the run.')
            st.link_button('Open this private Kaggle run',job['url'])
        except Exception as exc:st.error(f'Could not submit: {exc}')

with history:
    jobs=[]
    for path in sorted((ROOT/'runs'/'kaggle_ui').glob('*/job.json'),reverse=True):
        try:
            job=json.loads(path.read_text(encoding='utf8'))
            if job['ref'].split('/')[0]==username.strip().lower():jobs.append(job)
        except (ValueError,KeyError):pass
    if not jobs:
        st.write('No experiments yet.')
        st.caption('Submit your first run from New experiment. Its queue status, generated text, and activation plots will appear here.')
    else:
        refs=[j['ref'] for j in jobs];active=st.session_state.get('remote_job_ref')
        selected=st.selectbox('Experiment job',refs,index=refs.index(active) if active in refs else 0)
        job=next(j for j in jobs if j['ref']==selected)
        st.write('Status: **'+job['status']+'**')
        st.link_button('Kaggle run details',job['url'])
        a,b=st.columns(2)
        if a.button('Refresh status'):
            try:
                refresh_job(job,token=token);st.rerun()
            except Exception as exc:st.error(str(exc))
        if b.button('Retrieve results and vectors'):
            try:
                with st.spinner('Downloading recorded model outputs and activation captures...'):download_job(job,token=token)
                st.rerun()
            except Exception as exc:st.error(str(exc))
        result_path=Path(job['result_path']) if job.get('result_path') else None
        if result_path and result_path.exists():
            result=json.loads(result_path.read_text(encoding='utf8'))
            model_rows=[{'Checkpoint':m['model_id'],'Precision':m['precision'],'Status':m['status'],
                         'Seconds':round(m.get('seconds',0),1),
                         'Peak GPU GiB':round(m['peak_vram_bytes']/2**30,2) if m.get('peak_vram_bytes') is not None else None}
                        for m in result['models']]
            st.dataframe(pd.DataFrame(model_rows),hide_index=True,width='stretch')
            if result['failures']:st.error('Some operations failed. Their exceptions are retained in the evidence.');st.json(result['failures'])
            if result['results']:
                results=result['results']
                model=st.selectbox('Inspect checkpoint',list(dict.fromkeys(r['model_alias'] for r in results)))
                filtered=[r for r in results if r['model_alias']==model]
                layer=st.selectbox('Inspect intervention layer',sorted({r['layer_0based'] for r in filtered}))
                prompt=st.selectbox('Inspect test prompt',list(dict.fromkeys(r['prompt'] for r in filtered)))
                seed=st.selectbox('Inspect seed',sorted({r['seed'] for r in filtered}))
                chosen=[r for r in filtered if r['layer_0based']==layer and r['prompt']==prompt and r['seed']==seed]
                st.subheader('Generated text under the same checkpoint and seed')
                for r in chosen:
                    with st.expander('Unsteered baseline (0)' if r['coefficient']==0 else f"Coefficient {r['coefficient']:+g}",expanded=True):
                        st.write(r['text']);st.caption(f"{r['duration_seconds']:.2f} seconds · {r['new_tokens']} tokens · {r['scope']}")
                from mm_emotion.activation_view import activation_points,activation_animation
                try:
                    points,observed,variance=activation_points(result,result_path,model,layer,prompt)
                    observed=[p for p in observed if p['seed']==seed]
                    st.subheader('Measured activation geometry')
                    st.plotly_chart(activation_animation(points,observed,variance),width='stretch')
                    st.caption('Play animates the real recorded prefill activations across coefficients. PCA axes are fitted on training/reference captures only. A 3D projection is not an emotion probability. With decode_only steering, the prefill markers correctly stay unchanged.')
                    st.line_chart(pd.DataFrame(observed).pivot(index='coefficient',columns='seed',values='projection'),width='stretch')
                    st.caption('Projection of the actual measured activation relative to the neutral reference, onto this model’s native vector.')
                    st.dataframe(pd.DataFrame(observed)[['coefficient','added_projection','orthogonal_delta_norm']],hide_index=True,width='stretch')
                    st.caption('Isolation check: added_projection compares the measured state with its zero control. orthogonal_delta_norm measures change outside the selected vector; FP16 rounding can make it nonzero.')
                except Exception as exc:st.warning(f'Activation plot unavailable: {exc}')
                flat=[{k:v for k,v in r.items() if k!='observed_prefill_activation'} for r in results]
                st.download_button('Download generation table (CSV)',pd.DataFrame(flat).to_csv(index=False),'kaggle_generations.csv','text/csv')
            if result.get('course_results'):
                st.subheader('Original course matrix (ungraded)')
                st.dataframe(pd.DataFrame(result['course_results']),hide_index=True,width='stretch')
            st.download_button('Download full experiment, captures and native vectors (ZIP)',result_zip(result_path),
                               'kaggle_emotion_experiment.zip','application/zip')
            st.download_button('Download raw JSON evidence',result_path.read_bytes(),'experiment.json','application/json')
