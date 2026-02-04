# Simple file upload to RunPod using cat over SSH
# Bypasses SCP subsystem issues

$SshKey = "$env:USERPROFILE\.ssh\runpod_key"
$RemoteHost = "1o82atl93pl1v7-64411d79@ssh.runpod.io"
$RemotePath = "/workspace/ocr_service"

$files = @(
    "config.py",
    "pdf_processor.py",
    "ocr_model.py",
    "s3_service.py",
    "main.py",
    "requirements.txt",
    "start_server.sh"
)

Write-Host "Uploading files to RunPod..." -ForegroundColor Cyan

# Create remote directory
& ssh -i $SshKey $RemoteHost "mkdir -p $RemotePath"

# Upload each file
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Uploading $file..." -ForegroundColor Yellow
        
        # Read file and upload via SSH stdin
        $content = Get-Content $file -Raw
        $content | & ssh -i $SshKey $RemoteHost "cat > $RemotePath/$file"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Success: $file" -ForegroundColor Green
        }
        else {
            Write-Host "  Failed: $file" -ForegroundColor Red
        }
    }
}

Write-Host "`nVerifying uploaded files..." -ForegroundColor Cyan
& ssh -i $SshKey $RemoteHost "ls -lh $RemotePath"
