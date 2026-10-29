$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
python -m streamlit run app_emotion.py --server.address 127.0.0.1
