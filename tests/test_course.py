from pathlib import Path
import json
import numpy as np
import pytest
from mm_emotion.course_tests import (starter_cases,training_pairs,validate_cases,write_cases,read_cases,cosine,grade_case,evaluate_model,summary,save_results)
from mm_emotion.drawio import make_diagrams
from mm_emotion.course_report import make_docx

class FakeBackend:
    model_id='test/mock';precision='fp32';layers=list(range(10))
    def capture(self,prompt,layers):
        s=prompt.lower()
        v=np.array([len(s)*0.001, len(s)*0.002, len(s)*0.003, 1.],dtype=np.float32)
        if any(t in s for t in ('happy','pleased','delighted','smiles','cheers','good results')): v[0]+=3
        if any(t in s for t in ('sad','tears','discouraged','disappointing')): v[1]+=3
        if any(t in s for t in ('angry','anger','frustrated','unfair','fists')): v[2]+=3
        return np.stack([v+0.01*i for i in layers])
    def generate(self,prompt,**kwargs):return {'text':'unit test output','new_tokens':3}

def test_full_18_matrix():
    cases=starter_cases()
    assert validate_cases(cases,require_full_matrix=True)['unique_classifications']==18
    assert len(training_pairs())==12
    assert set(x['intensity'] for x in cases)=={'neutral','neutral/moderate','moderate','moderate/high','high'}
    assert sum(x['threshold_status']=='proposed_not_approved' for x in cases)==18

def test_roundtrip_and_reject_bad(tmp_path):
    cases=starter_cases()
    for ext in ('csv','json','jsonl'):
        p=tmp_path/('case.'+ext);write_cases(p,cases)
        assert len(read_cases(p,require_full_matrix=True))==18
    bad=cases.copy();bad[0]=dict(cases[0],case_id=cases[1]['case_id'])
    with pytest.raises(ValueError):validate_cases(bad,require_full_matrix=True)

def test_cosine_zero_and_provisional_grading():
    assert cosine(np.zeros(3),np.ones(3)) is None
    c=starter_cases()[5]
    a=grade_case(c,{'happiness':0.9,'anger':0.2,'sadness':0.1})
    assert a['status']=='ungraded_provisional_thresholds'
    assert grade_case(c,{'happiness':0.9,'anger':0.2,'sadness':0.1},approve_proposed_thresholds=True)['status']=='pass'
    assert grade_case(c,{'happiness':0.5},approve_proposed_thresholds=True)['status']=='fail'
    assert grade_case(c,{},approve_proposed_thresholds=True)['status']=='unscorable_vector'

def test_real_backend_contract_with_fake_backend_only():
    rows=evaluate_model(FakeBackend(),starter_cases(),training_pairs(),5)
    assert len(rows)==18
    assert all(r['origin']=='real_model_inference' for r in rows) # mocked in unit test ONLY
    assert set(r['status'] for r in rows)<= {'ungraded_provisional_thresholds','unscorable_vector'}
    assert summary(rows)['n_cases']==18

def test_report_and_drawio_are_real_editable_artifacts(tmp_path):
    draw=make_diagrams(tmp_path/'drawio')
    doc=make_docx(tmp_path/'report.docx',diagram_path=draw)
    assert doc.stat().st_size>5000
    import xml.etree.ElementTree as ET
    assert len(ET.parse(draw).getroot().findall('diagram'))==4
    from docx import Document
    text='\n'.join(p.text for p in Document(doc).paragraphs)
    assert all(f'Section {i}' in text for i in (1,2,3,4))
    assert 'NOT RUN' in text

def test_save_results_is_append_only(tmp_path):
    p=tmp_path/'case.json'
    saved=save_results(p,starter_cases(),[],{'layer':5})
    assert saved['origin']=='not_run'
    with pytest.raises(FileExistsError):save_results(p,starter_cases(),[],{})
