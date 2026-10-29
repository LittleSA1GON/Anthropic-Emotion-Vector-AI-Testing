$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
Set-Location $labRoot
$labPython = Join-Path $labRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) {
    throw 'Create .venv and install requirements-ui.txt before launching the Kaggle UI.'
}
Write-Host 'Open http://127.0.0.1:8501/Kaggle_Experiment_Lab'
& $labPython -m streamlit run app_emotion.py --server.address 127.0.0.1 --browser.gatherUsageStats false
