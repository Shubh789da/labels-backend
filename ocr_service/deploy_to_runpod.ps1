# Deploy to RunPod using Full SSH (SCP)
# User must provide new IP/Port after pod restart

$sshKey = "$env:USERPROFILE\.ssh\runpod_key"

Write-Host "=== RunPod SCP Deployment ===" -ForegroundColor Cyan
Write-Host "Because you restarted your pod, the IP and Port have changed." -ForegroundColor Yellow
Write-Host "Please check the 'Connect' button on RunPod console." -ForegroundColor Yellow
Write-Host ""

# Prompt for connection details
$remoteIP = Read-Host "Enter Public IP (e.g. 213.173...)"
$remotePort = Read-Host "Enter Public Port (e.g. 47566)"
$remoteUser = "root"
$remotePath = "/workspace/ocr_service/"

if ([string]::IsNullOrWhiteSpace($remoteIP) -or [string]::IsNullOrWhiteSpace($remotePort)) {
    Write-Host "IP and Port are required!" -ForegroundColor Red
    exit 1
}

# 1. Create Directory
Write-Host "`n[1/2] Creating directory..." -ForegroundColor Cyan
ssh -p $remotePort -i $sshKey $remoteUser@$remoteIP "mkdir -p $remotePath"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error creating directory. Check IP/Port and try again." -ForegroundColor Red
    exit 1
}

# 2. Upload Files
Write-Host "`n[2/2] Uploading files..." -ForegroundColor Cyan
$files = "config.py", "pdf_processor.py", "ocr_model.py", "s3_service.py", "main.py", "requirements.txt", "start_server.sh", ".env"

scp -P $remotePort -i $sshKey $files $remoteUser@$remoteIP`:$remotePath

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[+] Deployment successful!" -ForegroundColor Green
    Write-Host "Verify with:" -ForegroundColor Gray
    Write-Host "ssh -p $remotePort -i $sshKey $remoteUser@$remoteIP 'ls -lh $remotePath'"
}
else {
    Write-Host "`n[!] SCP failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
