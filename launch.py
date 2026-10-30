"""Set up the lightweight client and open the Model Mountain experiment UI."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parent


def environment_python():
    directory=ROOT/'.venv'
    executable=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not executable.exists():
        print('Creating the project Python environment...',flush=True)
        subprocess.run([sys.executable,'-m','venv',str(directory)],check=True)
    return executable


def prepare_client(executable):
    files=[ROOT/'requirements-core.txt',ROOT/'requirements-ui.txt']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    marker=ROOT/'.venv'/'.emotion-client-requirements'
    check="import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(n) for n in ['streamlit','pandas','numpy','plotly','kaggle','requests']) else 1)"
    ready=subprocess.run([str(executable),'-c',check],capture_output=True).returncode==0
    if not ready or not marker.exists() or marker.read_text().strip()!=digest:
        print('Installing the UI and Kaggle client dependencies (first launch only)...',flush=True)
        subprocess.run([str(executable),'-m','pip','install','-r',str(ROOT/'requirements-ui.txt')],cwd=ROOT,check=True)
        marker.write_text(digest+'\n',encoding='ascii')


def healthy(url):
    try:
        with urllib.request.urlopen(url+'/_stcore/health',timeout=2) as response:
            return response.status==200
    except Exception:return False


def existing_server(executable,port):
    # Inspect only matching Streamlit commands; never reuse an unrelated app's port.
    code="""import pathlib,psutil,sys
root=pathlib.Path(sys.argv[1]).resolve();port=sys.argv[2]
for p in psutil.process_iter(['pid','cmdline','cwd']):
    try:
        args=p.info['cmdline'] or []
        if 'streamlit' not in args or 'run' not in args:continue
        target=pathlib.Path(args[args.index('run')+1])
        if not target.is_absolute():target=pathlib.Path(p.info['cwd'])/target
        if target.resolve()!=root/'app_emotion.py':continue
        selected=args[args.index('--server.port')+1] if '--server.port' in args else next((v.split('=',1)[1] for v in args if v.startswith('--server.port=')),'8501')
        if selected==port:print(p.info['pid']);break
    except (psutil.Error,ValueError,IndexError,TypeError):pass
"""
    result=subprocess.run([str(executable),'-c',code,str(ROOT),str(port)],capture_output=True,text=True)
    return result.returncode==0 and result.stdout.strip().isdigit()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8501)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--check',action='store_true',help='Set up and verify dependencies without starting a server')
    args=parser.parse_args()
    if sys.version_info<(3,11):raise SystemExit('Install Python 3.11 or newer, then run this launcher again.')
    if not 1024<=args.port<=65535:raise SystemExit('Choose a port from 1024 to 65535.')
    executable=environment_python();prepare_client(executable)
    if args.check:
        print('Client ready. No local GPU or model-weight download is required.');return
    url=f'http://127.0.0.1:{args.port}'
    if not (healthy(url) and existing_server(executable,args.port)):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',args.port))==0:
                raise SystemExit(f'Port {args.port} is used by another app. Run: python launch.py --port {args.port+1}')
        folder=ROOT/'runs';folder.mkdir(exist_ok=True)
        stdout=(folder/'launcher.stdout.log').open('a',encoding='utf8')
        stderr=(folder/'launcher.stderr.log').open('a',encoding='utf8')
        process=subprocess.Popen([str(executable),'-m','streamlit','run',str(ROOT/'app_emotion.py'),
            '--server.address=127.0.0.1',f'--server.port={args.port}','--server.headless=true',
            '--browser.gatherUsageStats=false'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
        (folder/'launcher.json').write_text(json.dumps({'pid':process.pid,'port':args.port,'root':str(ROOT)}),encoding='utf8')
        stdout.close();stderr.close()
        print('Starting the experiment UI...',flush=True)
        deadline=time.monotonic()+45
        while not healthy(url):
            if process.poll() is not None:raise SystemExit('The UI could not start. See runs/launcher.stderr.log.')
            if time.monotonic()>deadline:raise SystemExit('UI startup timed out. See runs/launcher.stderr.log.')
            time.sleep(.5)
    print('Emotion lab is ready: '+url,flush=True)
    if not args.no_browser:webbrowser.open(url)


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as exc:raise SystemExit(f'Setup failed (exit {exc.returncode}). Check the installation output and internet connection.')
