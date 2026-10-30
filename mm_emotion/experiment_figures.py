"""Measured experiment playback and color-consistent emotion instruments."""
import html
from pathlib import Path
import numpy as np
from .experiment_inputs import emotion_colors


def emotion_gauge(label,value,color):
    import plotly.graph_objects as go
    fig=go.Figure(go.Indicator(mode='gauge+number' if value is not None else 'gauge',value=float(value or 0),
        title={'text':html.escape(label),'font':{'size':17,'color':color}},
        number={'valueformat':'.2f','font':{'size':25,'color':'#192b36'},'suffix':'' if value is not None else ' · N/A'},
        gauge={'axis':{'range':[-1,1],'tickvals':[-1,0,1],'tickfont':{'size':10}},
               'bar':{'color':color,'thickness':.25},'bgcolor':'#eef1f4','borderwidth':0,
               'threshold':{'value':0,'line':{'color':'#b5bec4','width':1}}}))
    fig.update_layout(height=180,margin=dict(l=22,r=22,t=40,b=5),paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Arial'))
    if value is None:fig.add_annotation(text='Unscorable',x=.5,y=.12,showarrow=False,font=dict(size=14))
    return fig


def case_arrays(result,result_path,alias,case_id):
    root=Path(result_path).parent.resolve();path=(root/'captures'/(alias+'.npz')).resolve()
    if not path.is_relative_to(root):raise ValueError('Invalid capture path')
    model=next(m for m in result['models'] if m['alias']==alias and m['status']=='completed')
    with np.load(path,allow_pickle=False) as data:
        position=data['layers'].astype(int).tolist().index(model['selected_layer'])
        row=data['case_ids'].astype(str).tolist().index(case_id)
        return {'raw':data['raw'][row,position],'neutral_reference':data['reference'][row,position],
                'positive':data['positive'][:,position],'neutral':data['neutral'][:,position],
                'train_labels':data['train_labels'].astype(str).tolist(),
                'directions':data['directions'][:,position], 'emotions':data['emotions'].astype(str).tolist()}


def experiment_animation(result,result_path,alias,case_id):
    import plotly.graph_objects as go
    arrays=case_arrays(result,result_path,alias,case_id);labels=arrays['emotions'];colors=emotion_colors(labels)
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
    classification=next(r for r in result['classifications'] if r['model_alias']==alias and r['case_id']==case_id)
    best=classification['closest_emotion'] or 'No positive match'
    stages=[('Input','Input activation',0),('Neutral','Matched neutral reference',1),
            ('Subtract','Subtract neutral: Δh = h(input) − h(neutral)',2),('Match','Closest vector: '+best,3)]
    generations=[r for r in result['results'] if r['model_alias']==alias and r.get('case_id')==case_id]
    stages += [('Steer '+str(i+1),r['condition'],4+i) for i,r in enumerate(generations)]
    for frame_name,title,stage in stages:
        traces=list(context)
        traces.append(scatter([[0,0,0]],'Neutral reference','#263b47',6,visible=stage>=1))
        traces.append(scatter([raw],'Input relative to neutral','#164e69',7))
        traces.append(scatter([[0,0,0],raw],'Neutral-subtracted difference','#164e69',4,'lines+markers',visible=stage>=2))
        for label,tip in zip(labels,direction_points):
            traces.append(scatter([[0,0,0],tip],label+' direction',colors[label],3,'lines+markers',visible=stage>=3))
        value=None
        if stage>=4:
            row=generations[stage-4];value=project(row['observed_prefill_activation']);all_points.append(value)
        traces.append(scatter([value if value is not None else raw],'Measured generation state','#121b21',8,visible=stage>=4))
        frames.append(go.Frame(name=frame_name,data=traces,layout=go.Layout(title=dict(text=title,font=dict(size=14)))))
    ranges=[];all_points=np.asarray(all_points)
    for dimension in range(3):
        low=float(all_points[:,dimension].min());high=float(all_points[:,dimension].max());pad=max((high-low)*.12,.2)
        ranges.append([low-pad,high+pad])
    figure=go.Figure(data=frames[0].data,frames=frames)
    figure.update_layout(title=dict(text='Input activation',font=dict(size=14)),height=590,
        margin=dict(l=0,r=0,t=50,b=95),paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#192b36'),
        scene=dict(bgcolor='rgba(0,0,0,0)',xaxis=dict(title='PC1',range=ranges[0]),
                   yaxis=dict(title='PC2',range=ranges[1]),zaxis=dict(title='PC3',range=ranges[2])),
        legend=dict(orientation='h',font=dict(size=10),y=1.1,yanchor='bottom'),
        updatemenus=[dict(type='buttons',x=0,y=-.1,showactive=False,buttons=[
            dict(label='Play',method='animate',args=[None,{'frame':{'duration':900,'redraw':True},'fromcurrent':True}]),
            dict(label='Pause',method='animate',args=[[None],{'frame':{'duration':0,'redraw':False},'mode':'immediate'}])])],
        sliders=[dict(x=.25,len=.75,y=-.1,currentvalue=dict(prefix='Stage: '),
            steps=[dict(label=f.name,method='animate',args=[[f.name],{'mode':'immediate','frame':{'duration':0,'redraw':True}}]) for f in frames])])
    return figure


def subtraction_chart(result,result_path,alias,case_id):
    import plotly.graph_objects as go
    data=case_arrays(result,result_path,alias,case_id);difference=data['raw']-data['neutral_reference']
    indices=np.argsort(np.abs(difference))[-12:]
    figure=go.Figure()
    for label,values,color in [('Input',data['raw'],'#164e69'),('Neutral',data['neutral_reference'],'#b0bcc2'),('Difference',difference,'#32877d')]:
        figure.add_bar(name=label,x=[str(i) for i in indices],y=values[indices],marker_color=color)
    figure.update_layout(height=280,barmode='group',margin=dict(l=20,r=0,t=20,b=30),
        xaxis_title='Hidden dimensions with the largest neutral-subtracted change',yaxis_title='Activation units',
        paper_bgcolor='rgba(0,0,0,0)',legend=dict(orientation='h',y=1.15))
    return figure
