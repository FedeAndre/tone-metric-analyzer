$ErrorActionPreference = "Stop"

Write-Host "Tone-Metric Analyzer v0.16.3"
Write-Host "Creating/updating local virtual environment..."

py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe validate_release.py

Write-Host ""
Write-Host "Starting v0.16.3 on http://127.0.0.1:8000"
Write-Host "After startup, verify with:"
Write-Host "  Invoke-RestMethod http://127.0.0.1:8000/api/status"
Write-Host "Expected omr_driver: step-page-save"

& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
