# Upload files to RunPod using base64 encoding over SSH
# This is the most reliable method when SCP subsystem is not available

param(
    [string]$RemoteHost = "1o82atl93pl1v7-64411d79@ssh.runpod.io",
    [string]$SshKey = "$env:USERPROFILE\.ssh\runpod_key",
    [string]$RemotePath = "/workspace/ocr_service"
)

$files = @(
    "config.py",
    "pdf_processor.py",
    "ocr_model.py",
    "s3_service.py",
    "main.py",
    "requirements.txt",
    "start_server.sh"
)

Write-Host "=== Uploading files to RunPod using base64 ===" -ForegroundColor Cyan

# Create remote directory
Write-Host "`nCreating remote directory..." -ForegroundColor Yellow
& ssh -i $SshKey $RemoteHost "mkdir -p $RemotePath"

$successCount = 0
$failCount = 0

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        Write-Host "SKIP: $file (not found)" -ForegroundColor Gray
        continue
    }
    
    Write-Host "`nUploading: $file" -ForegroundColor Yellow
    
    # Read file as bytes and convert to base64
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $file))
    $base64 = [Convert]::ToBase64String($bytes)
    
    # Upload via SSH - decode base64 and write to file
    $remotePath = "$RemotePath/$file"
    $uploadCmd = "echo '$base64' | base64 -d > $remotePath && echo 'OK' || echo 'FAIL'"
    
    $result = & ssh -i $SshKey $RemoteHost $uploadCmd 2>&1 | Select-Object -Last 1
    
    if ($result -like "*OK*") {
        # Verify file size
        $localSize = $bytes.Length
        $remoteSize = & ssh -i $SshKey $RemoteHost "stat -c%s $remotePath 2>/dev/null || stat -f%z $remotePath 2>/dev/null"
        
        if ($remoteSize -eq $localSize) {
            Write-Host "  SUCCESS: $file ($localSize bytes)" -ForegroundColor Green
            $successCount++
        }
        else {
            Write-Host "  WARNING: $file size mismatch (local: $localSize, remote: $remoteSize)" -ForegroundColor Yellow
            $successCount++
        }
    }
    else {
        Write-Host "  FAILED: $file" -ForegroundColor Red
        $failCount++
    }
}

Write-Host "`n=== Upload Summary ===" -ForegroundColor Cyan
Write-Host "Success: $successCount files" -ForegroundColor Green
Write-Host "Failed: $failCount files" -ForegroundColor Red

Write-Host "`n=== Verifying remote files ===" -ForegroundColor Cyan
& ssh -i $SshKey $RemoteHost "ls -lh $RemotePath"

Write-Host "`nDone!" -ForegroundColor Green
