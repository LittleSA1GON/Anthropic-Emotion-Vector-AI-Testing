"""Kaggle-native multi-model and CMPE187 controls, no server or external UI port."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
from .multi_model import REGISTRY,run_models
from .course_tests import starter_cases


def display_course_panel(project='.', before_run=None):  # user initiates each GPU action via a button
    import ipywidgets as w
    from IPython.display import display, clear_output
    base=Path(project).resolve()
    selected=w.SelectMultiple(description='Models',options=list(REGISTRY),value=('qwen05',),rows=5,layout=w.Layout(width='420px'))
    mode=w.Dropdown(description='Run mode',options=[('Direct inference','infer'),('18-case matrix','matrix')])
    prompts=w.Textarea(description='Prompts',value='A student receives unexpected good news. How do they feel?\nA student is angry about an unfair grade.',layout=w.Layout(width='90%',height='120px'))
    layer=w.IntText(description='Layer',value=5)
    tokens=w.IntSlider(description='New tokens',min=8,max=128,step=8,value=48)
    seed=w.IntText(description='Seed',value=42)
    review=w.Checkbox(description='Instructor approved proposed cosine intervals (otherwise UNGRADED)',value=False)
    execute=w.Button(description='RUN real models sequentially',button_style='primary')
    output=w.Output(layout=w.Layout(border='1px solid #94A3B8',max_height='400px',overflow='auto'))
    def on_execute(_):
        with output:
            clear_output()
            if not selected.value:
                print('Select at least one model');return
            models=[{**REGISTRY[n],'alias':n} for n in selected.value]
            pair_prompts=[{'id':f'line-{i+1}','prompt':p.strip()} for i,p in enumerate(prompts.value.splitlines()) if p.strip()]
            if mode.value=='infer' and not pair_prompts:
                print('Enter one prompt per line');return
            print('Running', [m['alias'] for m in models], 'sequentially; only real GPU results will be written')
            try:
                if before_run is not None:before_run()
                result=run_models(models,mode=mode.value,layer=int(layer.value),max_new_tokens=int(tokens.value),
                                  seed=int(seed.value),continue_on_error=True,
                                  prompts=pair_prompts if mode.value=='infer' else None,
                                  cases=starter_cases() if mode.value=='matrix' else None,
                                  approve_proposed_thresholds=bool(review.value))
                path=base/'runs'/'kaggle_multi_model'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'.json')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
                print('Wrote',path,'\nResults:',len(result['results']),'\nErrors:',result['failures'])
                for r in result['results'][:6]:
                    print(r.get('model_alias'),r.get('case_id',r.get('id')),r.get('status',r.get('output',{}).get('text','')))
            except Exception as exc:print('Real-model execution error:',type(exc).__name__,str(exc))
    execute.on_click(on_execute)
    display(w.VBox([w.HTML('<h3>Model Mountain — Kaggle multi-model research</h3><p>GPU is only used after clicking Run. The 18-case matrix follows Deliverable #2A.</p>'),selected,mode,prompts,layer,tokens,seed,review,execute,output]))
    return {'models':selected,'mode':mode,'prompts':prompts,'run':execute,'output':output}
