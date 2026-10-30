"""PCA views of real measured decoder activations, with explicit provenance."""
from pathlib import Path
import numpy as np


def activation_points(result, result_path, alias, layer, prompt):
    root=Path(result_path).parent
    capture=(root/'captures'/(alias+'.npz')).resolve()
    if not capture.is_relative_to(root.resolve()):raise ValueError('Invalid capture path')
    with np.load(capture,allow_pickle=False) as data:
        layers=data['layers'].astype(int).tolist();idx=layers.index(layer)
        negative=data['neutral'][:,idx,:];positive=data['positive'][:,idx,:]
        reference=data['reference'][idx]
        direction=data['directions'][idx]
    generations=[r for r in result['results'] if r['model_alias']==alias and r['layer_0based']==layer and r['prompt']==prompt]
    if not generations:raise ValueError('No recorded generations for this selection')
    measured=np.array([r['observed_prefill_activation'] for r in generations],dtype=np.float64)
    anchors=np.vstack([negative,positive,reference[None,:]])
    # Fit axes on training/reference only, so changing sweep strengths does not refit the view.
    mean=anchors.mean(axis=0);centered=anchors-mean
    _,singular,axes=np.linalg.svd(centered,full_matrices=False)
    basis=axes[:3].T
    if basis.shape[1]<3:basis=np.pad(basis,((0,0),(0,3-basis.shape[1])))
    projected=(np.vstack([anchors,measured])-mean)@basis
    variance=(singular[:3]**2/max(float(np.sum(singular**2)),1e-20)).tolist()
    variance+=(3-len(variance))*[0.]
    n=len(negative);points=[]
    for i,xyz in enumerate(projected[:len(anchors)]):
        points.append({'x':float(xyz[0]),'y':float(xyz[1]),'z':float(xyz[2]),
                       'group':'Neutral training' if i<n else 'Emotion training' if i<2*n else 'Neutral reference',
                       'label':f'Pair {i%n+1}' if i<2*n else 'Reference'})
    observations=[]
    baselines={r['seed']:np.asarray(r['observed_prefill_activation'],dtype=float) for r in generations if r['coefficient']==0}
    for row,xyz,hidden in zip(generations,projected[len(anchors):],measured):
        change=hidden-baselines[row['seed']]
        added_projection=float(change@direction)
        observations.append({'x':float(xyz[0]),'y':float(xyz[1]),'z':float(xyz[2]),
            'coefficient':row['coefficient'],'seed':row['seed'],
            'projection':float((hidden-reference)@direction),'added_projection':added_projection,
            'orthogonal_delta_norm':float(np.linalg.norm(change-added_projection*direction)),
            'text':row['text']})
    return points,observations,variance


def activation_animation(points,observations,variance):
    import pandas as pd
    import plotly.express as px
    frames=[]
    for coefficient in sorted({p['coefficient'] for p in observations}):
        for point in points:frames.append({**point,'coefficient':coefficient,'seed':'training/reference'})
        for point in observations:
            if point['coefficient']==coefficient:
                frames.append({**point,'group':'Measured test activation','label':'Seed '+str(point['seed']),'seed':str(point['seed'])})
    fig=px.scatter_3d(pd.DataFrame(frames),x='x',y='y',z='z',color='group',
        animation_frame='coefficient',hover_data=['label','coefficient'],
        category_orders={'coefficient':sorted({p['coefficient'] for p in observations})},
        color_discrete_map={'Neutral training':'#b7c1c6','Emotion training':'#164e69',
                            'Neutral reference':'#a8661c','Measured test activation':'#b4233c'})
    from .ui_theme import RULE,PANEL,plot_font
    fig.update_traces(marker=dict(line=dict(width=0)))
    arrays=np.array([[r[c] for c in ('x','y','z')] for r in points+observations])
    ranges=[]
    for dim in range(3):
        lo=float(arrays[:,dim].min());hi=float(arrays[:,dim].max());pad=max((hi-lo)*.15,.25)
        ranges.append([lo-pad,hi+pad])
    axis=lambda title,span:dict(title=dict(text=title,font=plot_font(11)),range=span,gridcolor=RULE,
                                zerolinecolor='#b7c1c6',showbackground=False,tickfont=plot_font(9))
    fig.update_layout(height=650,margin=dict(l=0,r=0,t=85,b=100),uirevision='activation-view',
        paper_bgcolor='rgba(0,0,0,0)',font=plot_font(),hoverlabel=dict(bgcolor=PANEL,bordercolor=RULE,font=plot_font(12)),
        scene=dict(xaxis=axis(f'PC1 ({variance[0]:.0%})',ranges[0]),yaxis=axis(f'PC2 ({variance[1]:.0%})',ranges[1]),
                   zaxis=axis(f'PC3 ({variance[2]:.0%})',ranges[2]),bgcolor='rgba(0,0,0,0)'),
        legend=dict(title_text='',orientation='h',x=0,y=1.1,yanchor='bottom',
                    entrywidth=.5,entrywidthmode='fraction',font=plot_font(10)))
    # Strength steps hold long enough to compare positions; 3D scatter cannot tween.
    from .experiment_figures import STAGE_HOLD_MS
    from .ui_theme import ACCENT,PANEL
    for slider in fig.layout.sliders:
        slider.update(y=-.12,x=.18,len=.8,pad=dict(t=20),bgcolor=PANEL,activebgcolor=ACCENT,bordercolor=RULE,
                      tickcolor=RULE,font=plot_font(10),currentvalue=dict(prefix='Strength: ',font=plot_font(12)))
    for menu in fig.layout.updatemenus:
        menu.update(y=-.12,x=0,yanchor='top',pad=dict(t=20),bgcolor=PANEL,bordercolor=RULE,font=plot_font(12))
        menu.buttons[0].args[1]['frame']['duration']=STAGE_HOLD_MS
        menu.buttons[0].args[1]['transition']['duration']=0
    return fig
