param(
    [int]$Port = 8000
)

Write-Host "Starting FastAPI on port $Port" -ForegroundColor Cyan
python -m uvicorn api.main:app --host 0.0.0.0 --port $Port
