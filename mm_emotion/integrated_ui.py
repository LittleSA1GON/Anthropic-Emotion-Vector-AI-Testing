"""The integrated model-link / CSV / Kaggle experiment workspace."""
from pathlib import Path
import copy
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
from .experiment_inputs import probe_model,csv_template,parse_experiment_csv,emotion_colors
from .kaggle_runner import prepare_job,submit_job,refresh_job,download_job,result_zip,run_cli
from .ui_theme import apply_lab_theme

ROOT=Path(__file__).resolve().parents[1]


def _submit(config,attachments=None):
    with st.spinner('Submitting a private Kaggle GPU job...'):
        job=prepare_job(ROOT,st.session_state['account_username'].strip(),config,attachments=attachments)
        submit_job(job,token=st.session_state.get('account_token'))
    st.session_state['integrated_active_job']=job['ref']
    st.session_state['next_lab_view']='Results & steering';st.rerun()


def _jobs():
    jobs=[];owner=st.session_state.get('account_username','').strip().lower()
    for p in sorted((ROOT/'runs/kaggle_ui').glob('*/job.json'),reverse=True):
        try:
            job=json.loads(p.read_text(encoding='utf8'))
            if job['ref'].split('/')[0]==owner and job['configuration'].get('schema')=='mm-emotion.integrated-request.v2':jobs.append(job)
        except (ValueError,KeyError):pass
    return jobs


def _setup():
    st.subheader('1. Add a model')
    url=st.text_input('Hugging Face model link',value='https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
                      help='Paste a downloadable language-model checkpoint. Text API endpoints do not expose the decoder activations used here.')
    if st.button('Check and add model'):
        try:
            with st.spinner('Checking the model architecture and revision...'):model=probe_model(url)
            models=st.session_state.setdefault('lab_models',[])
            st.session_state['lab_models']=[m for m in models if m['model_id']!=model['model_id']]+[model]
            st.success('Architecture check passed. Kaggle will verify that the weights load and decoder hooks work.')
        except Exception as exc:st.error(f'Model could not be added: {exc}')
    models=st.session_state.get('lab_models',[])
    for model in list(models):
        text,precision,remove=st.columns([6,2,1])
        text.write(model['model_id']);text.caption(f"{model['layers']} decoder layers · {model['architecture']} · revision {str(model['revision'])[:10]}")
        model['precision']=precision.selectbox('Precision',['fp16','fp32','nf4'],index=['fp16','fp32','nf4'].index(model['precision']),key='precision_'+model['alias'])
        if remove.button('Remove',key='remove_'+model['alias']):
            st.session_state['lab_models']=[m for m in models if m['alias']!=model['alias']];st.rerun()
        if model.get('gated'):st.caption('This checkpoint requires model access and an attached Kaggle HF_TOKEN secret.')
    st.subheader('2. Upload the test CSV')
    example=st.selectbox('Built-in example',['Anthropic research pilot','CMPE 187 course matrix'],key='lab_example')
    if example=='Anthropic research pilot':
        example_bytes=(ROOT/'data/templates/anthropic_emotion_pilot.csv').read_bytes()
        example_filename='anthropic_emotion_pilot.csv'
        st.caption('Original story prompts inspired by [Anthropic’s emotion-concepts paper](https://arxiv.org/html/2604.07729v1): happy, calm and desperate. This is a small adaptation using the lab’s last-token contrasts, not a replication of its token-averaged and denoised method.')
    else:
        example_bytes=csv_template();example_filename='emotion_experiment_template.csv'
    a,b=st.columns([3,1])
    upload=a.file_uploader('One CSV for training pairs and test prompts',type=['csv'])
    b.download_button('Download example CSV',example_bytes,example_filename,'text/csv')
    st.caption('train rows learn one vector per emotion; test rows compare new prompts with matched neutral references.')
    dataset=None
    try:
        dataset=parse_experiment_csv(upload.getvalue() if upload else example_bytes)
        if not upload:st.caption('Using '+example+'. Run it directly, or upload your own CSV to replace it.')
        else:st.caption('Using your uploaded CSV. The built-in example is available to download separately.')
        st.write(f"{len(dataset['training'])} training pairs · {len(dataset['tests'])} tests · "+', '.join(dataset['emotions']))
        with st.expander('Preview CSV'):
            st.dataframe(pd.DataFrame(dataset['rows'])[['row_type','case_id','emotion','prompt','neutral_prompt']],hide_index=True,width='stretch')
    except Exception as exc:st.error(str(exc))
    st.subheader('3. Connect your Kaggle account')
    a,b=st.columns(2)
    username=a.text_input('Kaggle username',value=st.session_state.get('account_username',''),key='_lab_username')
    token=b.text_input('Kaggle API token',type='password',value=st.session_state.get('account_token',''),key='_lab_token',
        help='Kept only in this UI session. Leave blank to use your saved local Kaggle login.')
    st.session_state['account_username']=username;st.session_state['account_token']=token
    st.caption('Your account runs private GPU jobs. Tokens are not embedded in notebooks or experiment files.')
    if st.button('Check account'):
        try:
            run_cli(['kernels','list','--mine','--page-size','1','--csv'],token=token,timeout=45)
            st.session_state['verified_kaggle_username']=username
            st.success('Kaggle accepted this credential.')
        except Exception as exc:st.error(str(exc))
    layer=None;tokens=32;seed=42;temperature=0.;sweeps=True
    with st.expander('Experiment settings'):
        if st.checkbox('Choose a specific decoder layer'):
            layer=int(st.number_input('Layer (zero-based)',0,199,5))
        else:st.caption('Automatic: scan early, middle and late layers; intervene at the middle layer of each model.')
        a,b,c=st.columns(3)
        tokens=a.slider('Output tokens',8,128,32,8)
        seed=int(b.number_input('Seed',0,2147483647,42))
        temperature=c.slider('Temperature',0.,1.5,0.,.1)
        sweeps=st.checkbox('Record ±3 isolation examples for every emotion on the first test',True)
    if models and dataset:
        generations=len(models)*(len(dataset['tests'])+(2*len(dataset['emotions']) if sweeps else 0))
        st.caption(f'This run records {generations} generations. Models load sequentially. Test CSV rows remain ungraded against unapproved course thresholds.')
    st.caption('Run sends the selected CSV rows, model settings and lab source to your Kaggle account and uses its GPU quota.')
    if st.session_state.get('lab_demo'):st.caption('Turn off demonstration mode to submit a real Kaggle experiment.')
    if st.button('Run experiment',type='primary',disabled=not models or not dataset or not username.strip() or st.session_state.get('lab_demo',False)):
        try:
            _submit({'schema':'mm-emotion.integrated-request.v2',
                'models':[{k:m[k] for k in ['alias','model_id','revision','precision']} for m in models],
                'csv_rows':dataset['rows'],'layer':layer,'max_new_tokens':tokens,'seed':seed,
                'temperature':temperature,'analysis_sweeps':sweeps})
        except Exception as exc:st.error(f'Experiment could not be submitted: {exc}')


def _results():
    jobs=_jobs()
    if not jobs:
        st.subheader('No experiment yet')
        st.write('Add a model, upload your CSV, and run the experiment from Setup.')
        return
    refs=[j['ref'] for j in jobs];active=st.session_state.get('integrated_active_job')
    ref=st.selectbox('Experiment',refs,index=refs.index(active) if active in refs else 0,
        format_func=lambda r:next(j['configuration']['models'][0]['model_id']+' · '+j['created_utc'][:16]+' UTC' for j in jobs if j['ref']==r))
    job=next(j for j in jobs if j['ref']==ref)
    from .lifecycle_ui import monitor
    monitor(job,_inspect_result)


def _inspect_result(job,result,snapshot):
    path=Path(job['result_path'])
    if result['failures']:
        st.error('Some models could not complete this experiment.')
        with st.expander('Compatibility and error details'):st.json(result['failures'])
    complete=[m for m in result['models'] if m['status']=='completed']
    if not complete:
        from .lifecycle_ui import render_scene
        render_scene(snapshot,key=job['ref']+'_failed_scene')
        st.info('No complete model measurements are available. The error recording is preserved; return to Setup to submit a corrected experiment.')
        st.link_button('Open private job diagnostics',job['url'])
        return
    aliases=[m['alias'] for m in complete]
    alias=st.selectbox('Model',aliases,format_func=lambda a:next(m['model_id']+' · '+m['precision'] for m in complete if m['alias']==a))
    model=next(m for m in complete if m['alias']==alias)
    from .lifecycle_ui import render_scene,replay_controls
    snapshot={**snapshot,'architecture':model.get('architecture')}
    snapshot=replay_controls(result,snapshot,alias)
    layer_key='inspect_'+alias+'_'+str(path)+'_layer'
    scene_key=job['ref']+'_'+alias+'_scene'
    packet=st.session_state.get(scene_key)
    if isinstance(packet,dict) and packet.get('nonce')!=st.session_state.get(layer_key+'_nonce') and packet.get('layer') in model['scanned_layers']:
        st.session_state[layer_key]=packet['layer'];st.session_state[layer_key+'_nonce']=packet['nonce']
    scene_modes=['Recorded lifecycle','Extract direction']
    if any(r['model_alias']==alias and r['condition']!='baseline' for r in result['results']):scene_modes.append('Steering')
    scene_mode=st.radio('Recorded scene focus',scene_modes,index=1,horizontal=True,key=job['ref']+'_'+alias+'_focus')
    from .inspection import layer_data,extraction_values
    inspected=st.session_state.get(layer_key,model['selected_layer'])
    if inspected not in model['scanned_layers']:
        inspected=model['selected_layer'];st.session_state[layer_key]=inspected
    emotion=st.session_state.get('inspect_'+alias+'_'+str(path)+'_emotion',result['emotions'][0])
    inspected_data=layer_data(path,alias,inspected)
    computed=extraction_values(inspected_data,emotion)
    largest=np.argsort(abs(computed['raw_direction']))[-6:].tolist()
    snapshot['extraction']={'emotion':emotion,'layer':inspected,'pairs':computed['pairs'],'hidden_dim':computed['hidden_dim'],
        'norm':computed['raw_norm'],'dimensions':[{'index':i,'emotion_mean':float(computed['emotion_mean'][i]),
        'neutral_mean':float(computed['neutral_mean'][i]),'difference':float(computed['raw_direction'][i])} for i in largest]}
    if scene_mode!='Recorded lifecycle':
        snapshot={**snapshot,'state':'vector_subtraction' if scene_mode=='Extract direction' else 'emotion_steering',
            'message':'Recorded-data illustration: '+('matched means → subtraction → unit normalization' if scene_mode=='Extract direction' else 'native vector addition; inspect actual requested versus observed deltas below')}
    packet=render_scene(snapshot,key=scene_key,
        selected_layer=inspected,inspectable_layers=model['scanned_layers'])
    if isinstance(packet,dict) and packet.get('nonce')!=st.session_state.get(layer_key+'_nonce') and packet.get('layer') in model['scanned_layers']:
        st.session_state[layer_key]=packet['layer'];st.session_state[layer_key+'_nonce']=packet['nonce']
        st.rerun() # Publish the new component selection before rendering dependent charts.
    cases=[r for r in result['classifications'] if r['model_alias']==alias]
    case_id=st.selectbox('Test case',[c['case_id'] for c in cases],format_func=lambda i:next(c['case_id']+' · '+c['prompt'][:85] for c in cases if c['case_id']==i))
    case=next(c for c in cases if c['case_id']==case_id)
    st.caption(f"Inspecting layer {inspected} · run intervention layer {model['selected_layer']} · revision {str(model.get('resolved_revision'))[:12]} · vectors {'reused' if model.get('vector_reused') else 'learned from CSV training rows'}")
    readouts=[r for r in result['results'] if r['model_alias']==alias and r.get('case_id')==case_id and r.get('observed_prefill_layer_0based')==inspected]
    readout_names=['Input minus neutral']+[r['condition'] for r in readouts]
    default_readout=len(readout_names)-1 if result['configuration'].get('inference_prompt') else 0
    selected_readout=st.selectbox('Activation readout',readout_names,index=default_readout,key=job['ref']+'_'+alias+'_readout_'+str(inspected)+'_'+case_id)
    measured=next((r for r in readouts if r['condition']==selected_readout),None)
    from .integrated_experiment import score_emotions
    case_index=inspected_data['case_ids'].index(case_id)
    input_match=score_emotions(inspected_data['raw'][case_index]-inspected_data['reference'][case_index],inspected_data['directions'],inspected_data['emotions'])
    scores=measured['emotion_scores'] if measured else input_match['scores']
    closest=measured['closest_emotion'] if measured else input_match['closest_emotion']
    st.subheader('Emotion alignment · layer '+str(inspected))
    if not readouts:st.caption('Generation snapshots were not collected at this layer. Select the run intervention layer to inspect actual steering readouts.')
    st.write('Closest emotion vector: **'+str(closest or 'No positive match')+'**')
    st.caption('Each gauge shows cosine alignment of the neutral-subtracted activation with a learned vector. Scores range from −1 to +1; they are not probabilities or calibrated emotion labels.')
    from .experiment_figures import emotion_gauge,experiment_animation,experiment_animation_2d,subtraction_chart
    labels=result['emotions'];colors=emotion_colors(labels)
    for offset in range(0,len(labels),3):
        columns=st.columns(min(3,len(labels)-offset))
        for col,label in zip(columns,labels[offset:offset+3]):col.plotly_chart(emotion_gauge(label,scores[label],colors[label]),width='stretch',config={'displayModeBar':False})
    geometry,text,table=st.tabs(['Experiment playback','Generated text','All test results'])
    with geometry:
        from .scientific_ui import render_inspection
        render_inspection(result,path,alias)
        st.subheader('Recorded test-case playback · layer '+str(inspected))
        st.caption('Replay: input → matched neutral → subtraction → vector match → measured steering states.')
        projection=st.radio('Test-case projection',['2D PCA (no WebGL)','3D PCA'],horizontal=True,key=job['ref']+'_'+alias+'_projection')
        animation=experiment_animation_2d if projection.startswith('2D') else experiment_animation
        st.plotly_chart(animation(result,path,alias,case_id,inspected),width='stretch')
        st.caption('All activation points come from model captures. Difference vectors are computed from those captures. Direction arrows are scaled for visibility; PCA is a reduced-dimensional view.')
        with st.expander('See the subtraction by hidden dimension'):
            st.latex(r'\Delta h_\ell = h_\ell(\mathrm{input}) - h_\ell(\mathrm{neutral})')
            st.plotly_chart(subtraction_chart(result,path,alias,case_id,inspected),width='stretch')
    with text:
        for row in result['results']:
            if row['model_alias']==alias and row.get('case_id')==case_id:
                st.markdown('**'+row['condition']+'**');st.write(row['text'])
                st.caption('Measured activation match: '+str(row.get('closest_emotion') or 'No positive match')+f" · {row['duration_seconds']:.2f}s")
    with table:
        flat=[{'Model':r['model_alias'],'Case':r['case_id'],'Expected':r.get('expected_emotion') or r.get('emotion',''),
               'Closest vector':r['closest_emotion'],**{label:r['scores'].get(label) for label in labels}} for r in result['classifications']]
        st.dataframe(pd.DataFrame(flat),hide_index=True,width='stretch')
        st.download_button('Download results CSV',pd.DataFrame(flat).to_csv(index=False),'emotion_results.csv','text/csv')
        st.caption('The original classifications and source range strings remain in the downloaded evidence. No unapproved thresholds are graded.')
    st.subheader('Inference with emotion steering')
    st.caption('Set how much of each native vector to add or subtract. The next private Kaggle job reuses these exact vectors and checkpoint revision, and compares your settings with zero steering.')
    weights={};prefix=job['ref']+'_'+alias
    steering_layer=st.selectbox('Steering layer (native vectors available)',model['scanned_layers'],
        index=model['scanned_layers'].index(model['selected_layer']),key=prefix+'_steering_layer')
    for offset in range(0,len(labels),3):
        columns=st.columns(min(3,len(labels)-offset))
        for col,label in zip(columns,labels[offset:offset+3]):
            import html
            col.markdown(f'<span style="display:inline-flex;align-items:center;gap:.5rem;font-weight:600">'
                         f'<span aria-hidden="true" style="width:.6rem;height:.6rem;border-radius:50%;background:{colors[label]}"></span>'
                         f'{html.escape(label)}</span>',unsafe_allow_html=True)
            previous=float(result['configuration'].get('steering_weights',{}).get(label,0))
            weights[label]=col.slider('Steer '+label,-10.,10.,previous,.25,key=prefix+'_weight_'+label)
    prompt=st.text_area('Inference prompt',case['prompt'],key=prefix+'_infer_prompt')
    neutral=st.text_input('Matched neutral reference',case['neutral_reference'],key=prefix+'_infer_neutral')
    if st.button('Generate with these emotion settings',type='primary'):
        try:
            config=copy.deepcopy(result['configuration'])
            config.update(models=[{k:model.get(k) for k in ['alias','model_id','precision']}],layer=steering_layer,
                inference_prompt=prompt,inference_neutral=neutral,steering_weights=weights,analysis_sweeps=False,
                reuse_vectors={alias:{'npz':'inputs/'+alias+'.npz','metadata':'inputs/'+alias+'.json'}})
            config['models'][0]['revision']=model['resolved_revision']
            vector=path.parent/model['vector_path']
            _submit(config,{'inputs/'+alias+'.npz':vector,'inputs/'+alias+'.json':vector.with_suffix('.json')})
        except Exception as exc:st.error(f'Inference could not be submitted: {exc}')
    a,b=st.columns(2)
    a.download_button('Download experiment and vectors',result_zip(path),'emotion_experiment.zip','application/zip')
    b.download_button('Download raw evidence',path.read_bytes(),'experiment.json','application/json')


def main():
    st.set_page_config(page_title='Model Mountain · Emotion lab',page_icon='🔬',layout='wide')
    apply_lab_theme()
    profile=ROOT/'runs/kaggle_ui/profile.json'
    if 'account_username' not in st.session_state:
        st.session_state['account_username']=json.loads(profile.read_text()).get('username','') if profile.exists() else os.environ.get('MM_KAGGLE_USERNAME','')
    st.session_state.setdefault('account_token','')
    if 'next_lab_view' in st.session_state:st.session_state['lab_view']=st.session_state.pop('next_lab_view')
    st.title('Emotion lab')
    st.caption('Model link, test CSV, and your Kaggle account. Learn emotion directions, inspect neutral subtraction, and steer inference.')
    with st.container(key='lab_workspace'):
        view=st.radio('Workspace',['Setup','Results & steering'],horizontal=True,label_visibility='collapsed',key='lab_view')
    if view=='Setup':
        controls,visual=st.columns([1,1.4],gap='large')
        with controls:_setup()
        with visual:
            from .lifecycle_ui import idle_workspace
            idle_workspace()
    else:_results()
    with st.expander('Advanced tools'):
        st.markdown('[Local GPU workbench](/Local_Vector_Workbench) · [Legacy batch experiments](/Multi_Model_Experiments) · [Course report builder](/CMPE187_Report_Builder)')
