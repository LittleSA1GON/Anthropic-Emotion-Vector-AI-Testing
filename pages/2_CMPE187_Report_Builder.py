"""Editable class matrix + draw.io and DOCX exports; never fabricates test results."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from mm_emotion.ui_theme import apply_lab_theme
apply_lab_theme()
from mm_emotion.course_tests import starter_cases, validate_cases, write_cases
from mm_emotion.course_report import make_docx
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/'cmpe187';OUT.mkdir(parents=True,exist_ok=True)
st.title('CMPE 187 — Deliverable #2 builder')
st.caption('Sections 1–4, complete 18-case matrix, native editable draw.io diagrams, evidence-based DOCX/PDF-ready report.')
st.warning('Original PDF pages 5–7 contain inverted cosine inequalities and ambiguous neutral expectations. Corrected intervals are proposals, NOT validated thresholds.')
if 'course_cases' not in st.session_state:st.session_state['course_cases']=starter_cases()
imported=st.file_uploader('Import custom 18-case CSV / JSON / JSONL',type=['csv','json','jsonl'])
if imported and st.button('Load uploaded test cases'):
    from mm_emotion.course_tests import read_cases
    target=OUT/('uploaded_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.'+imported.name.rsplit('.',1)[-1])
    target.write_bytes(imported.getvalue())
    try:
        st.session_state['course_cases']=read_cases(target,require_full_matrix=True)
        st.rerun()
    except Exception as exc:st.error(f'Invalid case file: {exc}')
cases=st.data_editor(pd.DataFrame(st.session_state['course_cases']),num_rows='fixed',use_container_width=True)
rows=cases.to_dict('records')
try:
    info=validate_cases(rows,require_full_matrix=True)
    st.success('Full 18-case primary matrix valid')
    import csv,io
    f=io.StringIO();writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    st.download_button('Download edited test-case CSV',f.getvalue(),'cmpe187_test_cases.csv')
except Exception as exc:st.error(str(exc));rows=None
st.subheader('Editable draw.io models')
for p in sorted((ROOT/'course/drawio').glob('*.drawio')):
    st.download_button('Download '+p.name,p.read_bytes(),p.name,mime='application/xml',key=p.name)
st.caption('Open the .drawio file in diagrams.net (draw.io) to edit; all diagrams are native nodes and connectors.')
st.subheader('Import actual multi-model test results')
uploaded_results=st.file_uploader('Existing evidence JSON (optional; do not supply synthetic smoke output)',type=['json'],key='results')
run=None
if uploaded_results:
    try:
        run=json.loads(uploaded_results.getvalue())
        if run.get('schema')!='mm-emotion.multi-model.v1' or run.get('mode')!='matrix':raise ValueError('Expected 18-case real multi-model matrix evidence JSON')
        if any(r.get('origin')!='real_model_inference' for r in run.get('results',[])):
            raise ValueError('Non-real evidence is not allowed in report')
        st.success(f"Found {len(run.get('results',[]))} actual-model test results")
    except Exception as exc:st.error(str(exc));run=None
if st.button('Generate DOCX Deliverable #2',disabled=rows is None,type='primary'):
    try:
        # Report currently displays first result per case ID when multiple models are supplied;
        # Section 4 disaggregates all models. Include clear model scope in report evidence.
        ts=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
        path=OUT/(ts+'_Deliverable2.docx')
        make_docx(path,cases=rows,run=run,diagram_path=ROOT/'course/drawio/CMPE187_All_Editable_DrawIO.drawio')
        st.session_state['last_report']=str(path)
        st.success('Editable report generated: '+path.name)
    except Exception as exc:st.error('Report creation failed: '+str(exc))
if st.session_state.get('last_report'):
    report=Path(st.session_state['last_report'])
    if report.exists():st.download_button('Download DOCX',report.read_bytes(),report.name)
    st.caption('DOCX contains explicit Not Run status until actual model inference evidence is uploaded.')
