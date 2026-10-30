"""Explicit source inventory shared by notebook and release builders."""
from pathlib import Path
import os

SOURCE_DIRS = {'mm_emotion', 'scripts', 'tests', 'configs', 'course', 'data',
               'docs', 'integrations', 'work_handoff', 'pages', '.github', '.streamlit', 'examples', 'reference'}
ROOT_FILES = {'AGENTS.md', 'README.md', '.gitignore', '.gitattributes', '.env.example', 'pytest.ini',
              'launch.py', 'run.bat', 'run.sh',
              'app_emotion.py', 'SOURCE_SHA256.json', 'START_HERE_FOR_WORK.md',
              'MODEL_MOUNTAIN_EMOTION_AGENT_CONTEXT.txt'}


def source_files(root, *, artifacts=False):
    root = Path(root)
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root)
        dirs[:] = [d for d in dirs if (d in SOURCE_DIRS if rel == Path('.') else
                   not d.startswith('.') and d not in {'__pycache__', 'diagram_previews', 'generated'})]
        for name in sorted(files):
            p = Path(current)/name
            if p.is_symlink() or name.startswith('.env') and name != '.env.example':
                continue
            if name in {'kaggle.json', 'secrets.toml', 'access_token'} or p.suffix in {'.pyc','.pt','.bin','.safetensors','.zip','.log'}:
                continue
            if rel == Path('.') and name not in ROOT_FILES and not name.startswith('requirements-'):
                continue
            if not artifacts and p.suffix in {'.pdf','.docx'}:
                continue
            yield p
    if artifacts:
        for folder in ('course/generated', 'notebooks', 'evidence'):
            for p in sorted((root/folder).rglob('*')):
                if p.is_file() and 'PRIVATE' not in p.name and p.suffix in {'.docx','.pdf','.ipynb','.json','.txt','.xml','.npz','.csv'}:
                    yield p
