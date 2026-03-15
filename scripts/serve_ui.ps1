param(
    [int]$Port = 8501,
    [string]$ApiUrl = "http://localhost:8000"
)

Write-Host "Starting Streamlit UI on port $Port (API=$ApiUrl)" -ForegroundColor Cyan
$env:DDI_API_URL = $ApiUrl
streamlit run ui/app.py --server.port $Port --server.address 0.0.0.0
