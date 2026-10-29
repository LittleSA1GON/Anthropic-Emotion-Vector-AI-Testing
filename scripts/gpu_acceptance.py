"""Opt-in real CUDA acceptance run, reusing the existing experiment modules."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from mm_emotion.artifacts import runtime_metadata
    from mm_emotion.multi_model import parse_models,run_models,export_flat_csv
    from mm_emotion.workbench import make_quick_vector,save_quick_vector
    from mm_emotion.dataset import load_pairs
    from mm_emotion.course_tests import starter_cases
    from mm_emotion.hf_backend import HFBackend
    from mm_emotion.course_report import make_docx
    from mm_emotion.blinding import export_blinded
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='qwen05,qwen15');p.add_argument('--out',required=True)
    p.add_argument('--compare-nf4',action='store_true',help='Also compare matched Qwen 0.5B FP16/NF4 captures')
    args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    status={'runtime':runtime_metadata(),'status':'running','completed':[],'failures':[]}
    def save(name,obj):
        (root/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    save('acceptance.json',status)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable; no model experiments attempted')
        models=parse_models(args.models)
        prompts=[{'id':'heldout-exam','prompt':'Explain how to approach an upcoming exam in two sentences.'}]
        pairs=[r for r in load_pairs('data/emotion_pairs_v1.jsonl') if r['emotion']=='joy' and r['split']=='train']
        direct=run_models(models,prompts=prompts,continue_on_error=True)
        save('direct.json',direct)
        if direct['failures']:raise RuntimeError('Direct inference failed; see direct.json')
        status['completed'].append('two_checkpoint_direct_inference')
        for m in models:
            backend=None
            try:
                backend=HFBackend(m['model_id'],precision=m['precision'])
                layers=[n for n in (5,11,17) if n<len(backend.layers)]
                dirs,meta=make_quick_vector(backend,pairs,layers,'mean_diff',label='joy')
                vec=save_quick_vector(root/'vectors',m['alias']+'_joy',dirs,meta)
                if any(block._forward_hooks for block in backend.layers):raise RuntimeError('Capture hook leaked')
            finally:
                if backend is not None:backend.close()
            sweep=run_models([m],prompts=prompts,mode='sweep',layer=5,
                vector_paths={m['alias']:str(vec)},coefficients=[-3.,0.,3.],seeds=[42,43],temperature=.7,
                continue_on_error=True,max_new_tokens=32)
            save(m['alias']+'_sweep.json',sweep)
            export_flat_csv(root/(m['alias']+'_sweep.json'),sweep)
            if sweep['failures']:raise RuntimeError('Sweep failed; inspect recorded failures')
            export_blinded(sweep,root/(m['alias']+'_blind'))
            status['completed'].append(m['alias']+'_native_vector_and_sweep')
            save('acceptance.json',status)
        matrix=run_models(models,mode='matrix',cases=starter_cases(),layer=5,continue_on_error=True)
        save('matrix.json',matrix);export_flat_csv(root/'matrix.json',matrix)
        if matrix['failures'] or len(matrix['results'])!=18*len(models):raise RuntimeError('Incomplete matrix; inspect matrix.json')
        make_docx(root/'CMPE187_Results.docx',run=matrix)
        from export_report_pdf import export_pdf
        export_pdf(root/'CMPE187_Results.docx',root/'CMPE187_Results.pdf')
        status['completed'].append('18_cases_per_model_ungraded_and_report')
        if args.compare_nf4:
            from mm_emotion.pipeline import run_experiment,compare_runs
            for precision in ('fp16','nf4'):
                run_experiment(backend_kind='hf',model='Qwen/Qwen2.5-0.5B-Instruct',precision=precision,
                    data_path='data/emotion_pairs_v1.jsonl',output=root/('precision_'+precision),
                    layers=[5],limit_per_split=2)
            compare_runs(root/'precision_fp16',root/'precision_nf4',root/'precision_comparison.json')
            status['completed'].append('matched_fp16_nf4_comparison')
        status['status']='completed'
    except Exception as exc:
        status['status']='failed';status['failures'].append(repr(exc))
        raise
    finally:
        save('acceptance.json',status)


if __name__=='__main__':main()
