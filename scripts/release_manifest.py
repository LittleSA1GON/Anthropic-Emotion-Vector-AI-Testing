"""Versioned integrity list for source files; notebook/source-manifest excluded to avoid circular hashes."""
from pathlib import Path
import hashlib,json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from mm_emotion.packaging import source_files
files={}
for p in sorted(source_files(root)):
    if p.name=='SOURCE_SHA256.json' or p.suffix=='.pyc':continue
    files[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'SOURCE_SHA256.json').write_text(json.dumps({'schema':'mm-emotion.source-manifest.v3','files':files,'excluded':'notebooks,generated outputs,reference PDFs,self'},indent=2)+'\n',encoding='utf-8')
print('Wrote source hashes for',len(files),'files')
