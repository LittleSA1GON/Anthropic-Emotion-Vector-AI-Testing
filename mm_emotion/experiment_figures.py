"""Measured experiment playback and color-consistent emotion instruments."""
import html
from pathlib import Path
import numpy as np
from .experiment_inputs import emotion_colors
from .ui_theme import ACCENT,INK,INK_SOFT,PANEL,RULE,plot_font

# Playback: each stage holds long enough to read its title; scatter3d cannot tween,
# so stage changes cut cleanly instead of faking interpolation.
STAGE_HOLD_MS=1300


def player_controls(frame_names,prefix='Stage',hold=STAGE_HOLD_MS,x=.27):
    """Shared Play / Pause / Restart buttons and a stage scrubber in the lab palette."""
    play={'frame':{'duration':hold,'redraw':True},'transition':{'duration':0},'fromcurrent':True,'mode':'immediate'}
    restart={'frame':{'duration':hold,'redraw':True},'transition':{'duration':0},'mode':'immediate'}
    pause={'frame':{'duration':0,'redraw':False},'transition':{'duration':0},'mode':'immediate'}
    menu=dict(type='buttons',direction='left',x=0,y=-.08,xanchor='left',yanchor='top',showactive=False,
        bgcolor=PANEL,bordercolor=RULE,borderwidth=1,font=plot_font(12),pad=dict(t=0,r=6),
        buttons=[dict(label='Play',method='animate',args=[None,play]),
                 dict(label='Pause',method='animate',args=[[None],pause]),
                 dict(label='Restart',method='animate',args=[list(frame_names),restart])])
    slider=dict(x=x,len=.97-x,y=-.08,yanchor='top',pad=dict(t=8,b=0),
        bgcolor=PANEL,activebgcolor=ACCENT,bordercolor=RULE,borderwidth=1,tickcolor=RULE,ticklen=4,
        font=plot_font(10),currentvalue=dict(prefix=prefix+': ',font=plot_font(12),xanchor='left'),
        transition=dict(duration=0),
        steps=[dict(label=name,method='animate',args=[[name],{'mode':'immediate','frame':{'duration':0,'redraw':True},
            'transition':{'duration':0}}]) for name in frame_names])
    return [menu],[slider]


def emotion_gauge(label,value,color):
    import plotly.graph_objects as go
    fig=go.Figure(go.Indicator(mode='gauge+number' if value is not None else 'gauge',value=round(float(value or 0),2)+0.,
        title={'text':html.escape(label),'font':{'size':15,'color':INK}},
        number={'valueformat':'+.2f','font':{'size':26,'color':INK},'suffix':'' if value is not None else ' Â· N/A'},
        gauge={'axis':{'range':[-1,1],'tickvals':[-1,0,1],'tickfont':{'size':10,'color':INK_SOFT},'tickcolor':RULE},
               'bar':{'color':color,'thickness':.25},'bgcolor':PANEL,'borderwidth':0,
               'threshold':{'value':0,'line':{'color':'#b5bec4','width':1}}}))
    # The needle sweeps in from zero so a new readout is visibly a new measurement.
    fig.update_layout(height=180,margin=dict(l=22,r=22,t=40,b=5),paper_bgcolor='rgba(0,0,0,0)',font=plot_font(),
        transition=dict(duration=650,easing='cubic-out'))
    if value is None:fig.add_annotation(text='Unscorable',x=.5,y=.12,showarrow=False,font=plot_font(13))
    return fig


def case_arrays(result,result_path,alias,case_id,layer=None):
    root=Path(result_path).parent.resolve();path=(root/'captures'/(alias+'.npz')).resolve()
    if not path.is_relative_to(root):raise ValueError('Invalid capture path')
    model=next(m for m in result['models'] if m['alias']==alias and m['status']=='completed')
    with np.load(path,allow_pickle=False) as data:
        position=data['layers'].astype(int).tolist().index(model['selected_layer'] if layer is None else layer)
        row=data['case_ids'].astype(str).tolist().index(case_id)
        return {'raw':data['raw'][row,position],'neutral_reference':data['reference'][row,position],
                'positive':data['positive'][:,position],'neutral':data['neutral'][:,position],
                'train_labels':data['train_labels'].astype(str).tolist(),
                'directions':data['directions'][:,position], 'emotions':data['emotions'].astype(str).tolist()}


def experiment_animation(result,result_path,alias,case_id,layer=None):
    import plotly.graph_objects as go
    arrays=case_arrays(result,result_path,alias,case_id,layer);labels=arrays['emotions'];colors=emotion_colors(labels)
    anchors=np.vstack([arrays['neutral'],arrays['positive']]);center=anchors.mean(axis=0)
    _,singular,axes=np.linalg.svd(anchors-center,full_matrices=False);basis=axes[:3].T
    if basis.shape[1]<3:basis=np.pad(basis,((0,0),(0,3-basis.shape[1])))
    reference=arrays['neutral_reference'];delta=arrays['raw']-reference
    project=lambda values:(np.asarray(values)-reference)@basis
    raw=delta@basis;frames=[];all_points=[raw,np.zeros(3)]
    def scatter(values,name,color,size=4,mode='markers',**kwargs):
        values=np.asarray(values).reshape(-1,3)
        return go.Scatter3d(x=values[:,0],y=values[:,1],z=values[:,2],mode=mode,name=name,
             marker=dict(color=color,size=size),line=dict(color=color,width=5),**kwargs)
    context=[scatter(project(arrays['neutral']),'Neutral training','#b7c1c6',3)]
    for label in labels:
        values=project(arrays['positive'][np.array(arrays['train_labels'])==label]);all_points.extend(values)
        context.append(scatter(values,label+' training',colors[label],3))
    direction_points=arrays['directions']@basis*max(float(np.linalg.norm(delta)),.5)
    all_points.extend(direction_points);all_points.extend(project(anchors))
    from .integrated_experiment import score_emotions
    classification=score_emotions(delta,arrays['directions'],labels)
    best=classification['closest_emotion'] or 'No positive match'
    stages=[('Input','Input activation',0),('Neutral','Matched neutral reference',1),
            ('Subtract','Subtract neutral: Î”h = h(input) âˆ’ h(neutral)',2),('Match','Closest vector: '+best,3)]
    generations=[r for r in result['results'] if r['model_alias']==alias and r.get('case_id')==case_id and (layer is None or r.get('observed_prefill_layer_0based')==layer)]
    stages += [('Steer '+str(i+1),r['condition'],4+i) for i,r in enumerate(generations)]
    total=len(stages)
    for frame_name,title,stage in stages:
        title=f'<span style="color:{INK_SOFT}">{stage+1} / {total}</span>   {html.escape(title)}'
        traces=list(context)
        traces.append(scatter([[0,0,0]],'Neutral reference','#263b47',6,visible=stage>=1))
        traces.append(scatter([raw],'Input relative to neutral','#164e69',7))
        traces.append(scatter([[0,0,0],raw],'Neutral-subtracted difference','#164e69',4,'lines+markers',visible=stage>=2))
        for label,tip in zip(labels,direction_points):
            traces.append(scatter([[0,0,0],tip],label+' direction',colors[label],3,'lines+markers',visible=stage>=3))
        value=None
        if stage>=4:
            row=generations[stage-4];value=project(row['observed_prefill_activation']);all_points.append(value)
        traces.append(scatter([value if value is not None else raw],'Measured generation state','#e7edf4',8,visible=stage>=4))
        frames.append(go.Frame(name=frame_name,data=traces,layout=go.Layout(title=dict(text=title,font=plot_font(15),x=0,xanchor='left',y=.985,yref='container'))))
    ranges=[];all_points=np.asarray(all_points)
    for dimension in range(3):
        low=float(all_points[:,dimension].min());high=float(all_points[:,dimension].max());pad=max((high-low)*.12,.2)
        ranges.append([low-pad,high+pad])
    figure=go.Figure(data=frames[0].data,frames=frames)
    menus,sliders=player_controls([f.name for f in frames])
    axis=lambda title,span:dict(title=dict(text=title,font=plot_font(11)),range=span,gridcolor=RULE,zerolinecolor='#b7c1c6',
                                showbackground=False,tickfont=plot_font(9))
    figure.update_layout(title=dict(text=frames[0].layout.title.text,font=plot_font(15),x=0,xanchor='left',y=.985,yref='container'),
        height=620,margin=dict(l=0,r=0,t=92,b=96),paper_bgcolor='rgba(0,0,0,0)',font=plot_font(),
        scene=dict(bgcolor='rgba(0,0,0,0)',xaxis=axis('PC1',ranges[0]),yaxis=axis('PC2',ranges[1]),zaxis=axis('PC3',ranges[2]),
                   camera=dict(eye=dict(x=1.45,y=1.45,z=.95))),
        legend=dict(orientation='h',font=plot_font(10),y=1.0,yanchor='bottom',x=0,itemclick='toggleothers'),
        hoverlabel=dict(bgcolor=PANEL,bordercolor=RULE,font=plot_font(12)),
        updatemenus=menus,sliders=sliders,uirevision=f'{alias}:{case_id}')
    return figure


def subtraction_chart(result,result_path,alias,case_id,layer=None):
    import plotly.graph_objects as go
    data=case_arrays(result,result_path,alias,case_id,layer);difference=data['raw']-data['neutral_reference']
    indices=np.argsort(np.abs(difference))[-12:]
    figure=go.Figure()
    for label,values,color in [('Input',data['raw'],'#164e69'),('Neutral',data['neutral_reference'],'#b0bcc2'),('Difference',difference,'#32877d')]:
        figure.add_bar(name=label,x=[str(i) for i in indices],y=values[indices],marker_color=color)
    figure.update_layout(height=280,barmode='group',margin=dict(l=20,r=0,t=20,b=30),
        xaxis_title='Hidden dimensions with the largest neutral-subtracted change',yaxis_title='Activation units',
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=plot_font(),bargap=.25,
        xaxis=dict(gridcolor=RULE,tickfont=plot_font(10)),yaxis=dict(gridcolor=RULE,zerolinecolor='#9aa7ae',tickfont=plot_font(10)),
        hoverlabel=dict(bgcolor=PANEL,bordercolor=RULE,font=plot_font(12)),legend=dict(orientation='h',y=1.15,font=plot_font(11)))
    return figure


def experiment_animation_2d(result,result_path,alias,case_id,layer=None):
    """The same training-fitted PCA and recorded frames, rendered without WebGL."""
    import plotly.graph_objects as go
    spatial=experiment_animation(result,result_path,alias,case_id,layer)
    def flat(trace):
        return go.Scatter(x=trace.x,y=trace.y,mode=trace.mode,name=trace.name,visible=trace.visible,
            marker=trace.marker.to_plotly_json(),line=trace.line.to_plotly_json())
    frames=[go.Frame(name=f.name,data=[flat(t) for t in f.data],layout=go.Layout(title=f.layout.title)) for f in spatial.frames]
    figure=go.Figure(data=[flat(t) for t in spatial.data],frames=frames)
    figure.update_layout(title=spatial.layout.title,height=510,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=plot_font(),margin=dict(l=35,r=15,t=100,b=110),
        xaxis=dict(title='PC1 (training-fitted)',range=spatial.layout.scene.xaxis.range,gridcolor=RULE),
        yaxis=dict(title='PC2 (training-fitted)',range=spatial.layout.scene.yaxis.range,gridcolor=RULE),
        legend=dict(orientation='h',y=1.12,font=plot_font(10)),
        updatemenus=spatial.layout.updatemenus,sliders=spatial.layout.sliders,uirevision=f'{alias}:{case_id}:2d')
    return figure
