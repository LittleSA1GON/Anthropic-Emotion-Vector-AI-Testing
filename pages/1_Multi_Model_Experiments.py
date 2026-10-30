"""Streamlit multi-model research panel; all GPU calls require explicit click."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from mm_emotion.ui_theme import apply_lab_theme
apply_lab_theme()
from mm_emotion.multi_model import REGISTRY, parse_models, run_models
from mm_emotion.course_tests import starter_cases, validate_cases
from mm_emotion.credentials import credential_panel
hf_token = credential_panel()

ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/'runs'/'multi_model';OUTPUT.mkdir(parents=True,exist_ok=True)
st.title('Multi-model direct inference & experiments')
st.caption('Runs checkpoints sequentially; zero-control for every steering sweep. No model weights loaded until Run is clicked.')

names=st.multiselect('Model checkpoints',list(REGISTRY),default=['qwen05','qwen15'],format_func=lambda n:f'{n}: {REGISTRY[n]["model_id"]} ({REGISTRY[n]["precision"]})')
with st.expander('Add custom Hugging Face checkpoint'):
    custom_alias=st.text_input('Alias','custom1')
    custom_model=st.text_input('Model ID','')
    custom_precision=st.selectbox('Precision',['fp16','nf4','fp32'])
    custom_max_length=st.number_input('Max input tokens',128,4096,512,128)
mode=st.radio('Experiment type',['Direct inference','18-case CMPE matrix','Native vector strength sweep'],horizontal=True)
seed=st.number_input('Seed',0,2147483647,42)
seed_list=st.text_input('Repeated seeds (infer / sweep; matrix uses one)',str(seed))
layer=st.number_input('Decoder layer (zero-based)',0,200,5)
max_new_tokens=st.slider('Max generated tokens',8,256,64,8)
col1,col2=st.columns(2)
with col1:temperature=st.slider('Temperature',0.,1.5,0.,.1)
with col2:top_p=st.slider('Top-p',0.05,1.,1.,.05)

models=[{**REGISTRY[n],'alias':n} for n in names]
if custom_model.strip():
    if custom_alias in names:st.error('Custom alias conflicts with a selected model')
    else:models.append({'alias':custom_alias,'model_id':custom_model.strip(),'precision':custom_precision,'max_length':int(custom_max_length)})

if mode in ('Direct inference','Native vector strength sweep'):
    prompt_text=st.text_area('Test prompts (one per line)',height=180,
        value='A student has exciting news. Explain how to respond.\nA student is angry about an unfair grade. Give one helpful sentence.')
    uploaded=st.file_uploader('Optional prompt file (.txt / .csv / .json / .jsonl)',type=['txt','csv','json','jsonl'])
    if uploaded:
        try:
            import tempfile
            from mm_emotion.multi_model import read_prompts
            tmp=OUTPUT/('import_'+hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]+'.'+uploaded.name.split('.')[-1]);tmp.write_bytes(uploaded.getvalue())
            prompts=read_prompts(tmp)
            st.caption(f'Imported {len(prompts)} prompts')
        except Exception as exc:
            st.error(str(exc));prompts=[]
    else:
        prompts=[{'id':f'prompt-{i+1}','prompt':p.strip()} for i,p in enumerate(prompt_text.splitlines()) if p.strip()]
    if mode=='Native vector strength sweep':
        st.warning('Only saved vectors from the exact checkpoint and precision are accepted. Use Find vectors in the main workbench first.')
        vector_paths={}
        vecs=sorted((ROOT/'runs').rglob('*.npz'))
        for m in models:
            candidate=st.selectbox(f'Native vector file for {m["alias"]}', ['Select a vector']+[str(p.relative_to(ROOT)) for p in vecs],key='v_'+m['alias'])
            if candidate!='Select a vector':vector_paths[m['alias']]=str(ROOT/candidate)
        coeff=st.text_input('Coefficient sweep','-6,-3,0,3,6')
    else:vector_paths={};coeff='0'
else:
    st.markdown('### Rubric-aligned 18-case matrix')
    cases=starter_cases()
    edited=st.data_editor(pd.DataFrame(cases),num_rows='fixed',use_container_width=True,
        column_config={'prompt':st.column_config.TextColumn('Test prompt',width='large'),
                       'neutral_reference':st.column_config.TextColumn('Matched neutral reference',width='large')})
    cases=edited.to_dict('records')
    st.caption('3 emotions × 3 abstraction categories × 2 diction categories. Baseline vectors trained on separate prompt contexts.')
    accept=st.checkbox('Instructor has reviewed and approved the proposed ascending cosine ranges',value=False)
    st.warning('The original PDF contains reversed inequalities. Without explicit approval, the test runner records similarity but leaves cases ungraded.')
    prompts=[];vector_paths={};coeff='0'

if st.button('Run selected models sequentially',type='primary',disabled=not models):
    try:
        previous=st.session_state.pop('backend',None)
        st.session_state.pop('model_key',None)
        if previous is not None:
            previous.close()
            del previous
        if mode=='18-case CMPE matrix':validate_cases(cases,require_full_matrix=True)
        elif not prompts:raise ValueError('Provide at least one prompt')
        with st.spinner(f'Loading and evaluating {len(models)} model(s) in sequence (downloads may be large)...'):
            out=run_models(models,mode='matrix' if mode=='18-case CMPE matrix' else 'sweep' if mode=='Native vector strength sweep' else 'infer',
                           cases=cases if mode=='18-case CMPE matrix' else None,
                           prompts=prompts if mode!='18-case CMPE matrix' else None,
                           layer=int(layer),max_new_tokens=int(max_new_tokens),temperature=float(temperature),top_p=float(top_p),seed=int(seed),
                           approve_proposed_thresholds=accept if mode=='18-case CMPE matrix' else False,
                           vector_paths=vector_paths,coefficients=[float(x.strip()) for x in coeff.split(',')],continue_on_error=True,token=hf_token,
                           seeds=[int(x.strip()) for x in seed_list.split(',')])
        path=OUTPUT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'_experiment.json')
        path.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        st.session_state['multi_last']=out
        st.success('Experiment recorded: '+str(path.relative_to(ROOT)))
    except Exception as exc:st.error(f'Experiment stopped: {exc}')

out=st.session_state.get('multi_last')
if out:
    st.subheader('Run evidence')
    st.dataframe(pd.DataFrame(out['models']),use_container_width=True)
    if out.get('failures'):st.error('Some models failed; inspect errors in the exported evidence.');st.json(out['failures'])
    st.dataframe(pd.DataFrame(out['results']),use_container_width=True)
    st.download_button('Download full JSON evidence',json.dumps(out,indent=2,ensure_ascii=False).encode('utf8'),'multi_model_experiment.json')
    if out['results']:
        st.download_button('Download CSV results',pd.DataFrame(out['results']).to_csv(index=False).encode('utf8'),'multi_model_results.csv')
