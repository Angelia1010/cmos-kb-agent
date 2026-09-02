Set-Location .\universal-agent

$env:PYTHONPATH = "$PWD\src;$PWD\services"

..\.venv\Scripts\python.exe -m uvicorn `
    processing_service.app:app `
    --host 127.0.0.1 `
    --port 8000