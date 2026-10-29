"""Package source and final artifacts, excluding credentials, weights and runs."""
from pathlib import Path
import hashlib
import sys
import zipfile
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from mm_emotion.packaging import source_files
out=root/'dist'/'Model_Mountain_Emotion_Lab_FULL.zip'
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(source_files(root,artifacts=True))):
        z.write(p,arcname='Model_Mountain_Emotion_Lab/'+p.relative_to(root).as_posix())
out.with_suffix('.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
print(out)
