"""Load all Streamlit pages without downloading weights or running inference."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

results={}
root=Path(__file__).resolve().parents[1]
for file in ('app_emotion.py','pages/1_Multi_Model_Experiments.py','pages/2_CMPE187_Report_Builder.py'):
    app=AppTest.from_file(str(root/file)).run(timeout=30)
    results[file]=[e.message for e in app.exception]
output=Path('evidence/ui-check.json');output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
print(json.dumps(results))
if any(results.values()):raise SystemExit(1)
