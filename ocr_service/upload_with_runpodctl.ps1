# Upload files to RunPod using runpodctl
# Auto-installs runpodctl if missing

Write-Host "=== RunPod File Upload (runpodctl) ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check/Install runpodctl
$runpodctl = Get-Command "runpodctl" -ErrorAction SilentlyContinue
$exePath = ".\runpodctl.exe"

if ($runpodctl) {
    Write-Host "✓ runpodctl found in PATH" -ForegroundColor Green
    $cmd = "runpodctl"
}
elseif (Test-Path $exePath) {
    Write-Host "✓ runpodctl found in current directory" -ForegroundColor Green
    $cmd = ".\runpodctl.exe"
}
else {
    Write-Host "Installing runpodctl..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile $exePath
        Write-Host "✓ Installed successfully!" -ForegroundColor Green
        $cmd = ".\runpodctl.exe"
    }
    catch {
        Write-Host "Failed to download runpodctl: $_" -ForegroundColor Red
        exit 1
    }
}

# 2. Upload Files
$files = "config.py", "pdf_processor.py", "ocr_model.py", "s3_service.py", "main.py", "requirements.txt", "start_server.sh", ".env"

Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host "1. I will generate a CODE for each file." -ForegroundColor Gray
Write-Host "2. You must run the command shown on your RunPod terminal." -ForegroundColor Gray
Write-Host ""
Write-Host "Wait for the code to appear below..." -ForegroundColor Yellow
Write-Host ""

foreach ($file in $files) {
    if (-not (Test-Path $file)) { continue }

    Write-Host "------------------------------------------------" -ForegroundColor Cyan
    Write-Host "Sending: $file" -ForegroundColor Yellow
    
    # Run runpodctl and show output to user
    & $cmd send $file
    
    Write-Host ""
    Read-Host "Press Enter after the file is received on RunPod..."
}

Write-Host ""
Write-Host "=== Upload Complete! ===" -ForegroundColor Green
