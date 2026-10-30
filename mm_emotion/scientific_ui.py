"""Native controls and bounded charts for recorded scientific measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from .inspection import layer_data,extraction_values,steering_values
from .ui_theme import INK,INK_SOFT,PANEL,RULE,plot_font

NEUTRAL='#79c9ed';EMOTIONAL='#e8b56b';DIRECTION='#77d5ba'


@st.cache_data(max_entries=18,show_spinner=False)
def cached_layer(path,alias,layer,modified):
    return layer_data(path,alias,layer)


def chart_layout(figure,**kwargs):
    figure.update_layout(height=320,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=plot_font(),margin=dict(l=20,r=15,t=35,b=40),
        xaxis=dict(gridcolor=RULE),yaxis=dict(gridcolor=RULE),legend=dict(orientation='h',y=1.15),**kwargs)
    return figure


def render_inspection(result,result_path,alias):
    import plotly.graph_objects as go
    model=next(m for m in result['models'] if m['alias']==alias and m['status']=='completed')
    layers=model['scanned_layers'];prefix='inspect_'+alias+'_'+str(result_path)
    a,b=st.columns(2)
    st.session_state.setdefault(prefix+'_layer',model['selected_layer'])
    layer=a.selectbox('Inspect captured layer',layers,index=None,key=prefix+'_layer')
    emotion=b.selectbox('Inspect emotion',result['emotions'],key=prefix+'_emotion')
    capture=Path(result_path).parent/'captures'/(alias+'.npz')
    data=cached_layer(str(result_path),alias,layer,capture.stat().st_mtime_ns)
    values=extraction_values(data,emotion)
    extraction,activations,weights,steering=st.tabs(['Extract the direction','Activation data','Architecture & weights','Steering arithmetic'])
    with extraction:
        st.caption(f"Recorded layer {layer} · {values['pairs']} matched pairs · {values['hidden_dim']} hidden dimensions · final prefill token")
        st.latex(r'\bar h_E=\frac{1}{N}\sum_i h_{E,i},\quad\bar h_N=\frac{1}{N}\sum_i h_{N,i}')
        st.latex(r'd=\bar h_E-\bar h_N,\qquad \hat d=d/\|d\|_2')
        st.write(f"Raw contrast norm: **{values['raw_norm']:.6g}** · Saved native direction: **{'matches recomputation' if values['native_matches'] else 'DOES NOT MATCH'}**")
        if not values['native_matches']:st.error('The stored direction does not match its training captures. Inspect provenance before reuse.')
        count=st.select_slider('Largest-changing dimensions',options=[8,16,32],value=16,key=prefix+'_top')
        indices=np.argsort(np.abs(values['raw_direction']))[-count:]
        frame=go.Figure()
        for name,column,color in [('Emotion mean','emotion_mean',EMOTIONAL),('Matched neutral mean','neutral_mean',NEUTRAL),('Element-wise difference','raw_direction',DIRECTION)]:
            frame.add_bar(name=name,x=indices.astype(str),y=values[column][indices],marker_color=color)
        st.plotly_chart(chart_layout(frame,barmode='group',xaxis_title='Hidden dimension',yaxis_title='Activation units'),width='stretch',key=prefix+'_subtraction')
        dimension=int(st.number_input('Inspect an exact hidden dimension',0,values['hidden_dim']-1,0,key=prefix+'_dimension'))
        st.dataframe(pd.DataFrame({'Quantity':['Emotion mean','Neutral mean','Raw difference','Unit direction'],
            'Value':[float(values[k][dimension]) for k in ['emotion_mean','neutral_mean','raw_direction','unit_direction']]}),hide_index=True,width='stretch')
        full=pd.DataFrame({'hidden_dimension':np.arange(values['hidden_dim']),**{k:values[k] for k in ['emotion_mean','neutral_mean','raw_direction','unit_direction']}})
        st.download_button('Download exact extraction values',full.to_csv(index=False),f'{alias}_{emotion}_layer{layer}.csv','text/csv',key=prefix+'_download')
    with activations:
        st.caption('Hidden-state activations, not model parameters or attention probabilities. Charts send only selected dimensions or histogram bins to the browser.')
        mask=np.array(data['labels'])==emotion
        indices=np.argsort(np.abs(values['raw_direction']))[-32:]
        matrix=np.vstack([data['neutral'][mask],data['positive'][mask]])[:,indices]
        # Bound the visible number of samples; exact arrays remain in the evidence ZIP.
        if len(matrix)>32:matrix=matrix[np.linspace(0,len(matrix)-1,32,dtype=int)]
        heatmap=go.Figure(go.Heatmap(z=matrix,x=indices.astype(str),colorscale='RdBu',zmid=0,colorbar=dict(title='h')))
        st.plotly_chart(chart_layout(heatmap,xaxis_title='Selected hidden dimensions',yaxis_title='Neutral pairs, then emotional pairs (sampled if >32)'),width='stretch',key=prefix+'_heatmap')
        case_id=st.selectbox('Compare test activations',data['case_ids'],key=prefix+'_case')
        row=data['case_ids'].index(case_id);histogram=go.Figure()
        for label,array,color in [('Input',data['raw'][row],EMOTIONAL),('Neutral reference',data['reference'][row],NEUTRAL),('Difference',data['raw'][row]-data['reference'][row],DIRECTION)]:
            frequencies,edges=np.histogram(array,bins=32)
            histogram.add_scatter(x=(edges[:-1]+edges[1:])/2,y=frequencies,name=label,mode='lines',line=dict(color=color))
        st.plotly_chart(chart_layout(histogram,xaxis_title='Activation value',yaxis_title='Hidden dimensions per bin'),width='stretch',key=prefix+'_histogram')
        st.caption('Layer choice changes this recorded-data inspection. New inference at another captured layer uses the separate Steering layer control below.')
    with weights:
        architecture=model.get('architecture')
        if architecture:
            keys=['model_id','format','model_type','layers','hidden_dim','attention_heads','intermediate_dim','parameter_count','precision','device']
            st.dataframe(pd.DataFrame({'Property':keys,'Recorded value':[str(architecture.get(k) if architecture.get(k) is not None else 'Not exposed') for k in keys]}),hide_index=True,width='stretch')
            st.write('Backend capabilities');st.json(architecture.get('capabilities',{}),expanded=False)
            if model.get('peak_vram_bytes') is not None:
                st.caption(f"Peak model allocation: {model['peak_vram_bytes']/2**30:.3f} GiB, recorded by PyTorch. This is allocated memory, not free scheduler capacity.")
            st.caption('Parameter matrices are trainable model weights. Attention probability matrices were not collected by this experiment.')
            components=[c for c in architecture.get('components',[]) if c['layer']==layer]
            if components:st.caption('Loaded components: '+str(components[0]))
        else:st.info(f"Architecture metadata was not collected by this archived version. Capture arrays verify hidden dimension {values['hidden_dim']}; total transformer layer count is not inferred from the sampled layers.")
        summary=model.get('weight_summaries',{})
        tensors=[r for r in summary.get('tensors',[]) if r['layer']==layer]
        if not tensors:st.info(summary.get('reason') or 'No weight samples were collected for this recording. Nothing is synthesized.')
        else:
            chosen=st.selectbox('Parameter tensor sample',list(range(len(tensors))),format_func=lambda i:tensors[i]['name'],key=prefix+'_weight')
            tensor=tensors[chosen]
            st.caption(f"Static trainable parameter · shape {tensor['shape']} · {tensor['dtype']} · {tensor['sample_count']} uniformly indexed values from {tensor['tensor_elements']} elements")
            st.dataframe(pd.DataFrame([{k:tensor[k] for k in ['sample_mean','sample_std','sample_min','sample_max']}]),hide_index=True,width='stretch')
            st.dataframe(pd.DataFrame({'Flattened index':tensor['sample_indices'],'Actual parameter value':tensor['sample_values']}),hide_index=True,width='stretch')
            st.caption('Statistics describe the bounded sample, not the entire matrix. Steering does not modify these parameters.')
    with steering:
        rows=[r for r in result['results'] if r['model_alias']==alias and r.get('observed_prefill_layer_0based')==layer]
        changed=[r for r in rows if r['condition']!='baseline']
        if not changed:st.info('No nonzero steering snapshot was recorded at this layer. Select its intervention layer or run a new comparison.')
        else:
            choice=st.selectbox('Recorded intervention',list(range(len(changed))),format_func=lambda i:changed[i]['case_id']+' · '+changed[i]['condition'],key=prefix+'_condition')
            selected=changed[choice]
            baseline=next(r for r in rows if r['condition']=='baseline' and r['case_id']==selected['case_id'] and r['seed']==selected['seed'])
            check=steering_values(baseline['observed_prefill_activation'],selected['observed_prefill_activation'],data['directions'],selected['weights'],data['emotions'])
            st.latex(r"h'_L=h_L+\sum_e\alpha_e\hat d_{L,e}")
            st.caption('The displayed values are actual first-prefill hook captures. Model weights remain fixed.')
            indices=np.argsort(np.abs(check['requested_delta']))[-16:];figure=go.Figure()
            for key,label,color in [('requested_delta','Requested vector addition',DIRECTION),('observed_delta','Measured activation change',EMOTIONAL)]:
                figure.add_bar(x=indices.astype(str),y=check[key][indices],name=label,marker_color=color)
            st.plotly_chart(chart_layout(figure,barmode='group',xaxis_title='Hidden dimension',yaxis_title='Activation change'),width='stretch',key=prefix+'_steer')
            st.write(f"Rounding / computation residual norm: **{check['residual_norm']:.6g}**")
            st.caption(check['note'])
            a,b=st.columns(2);a.markdown('**Zero baseline**');a.write(baseline['text']);b.markdown('**'+selected['condition']+'**');b.write(selected['text'])
