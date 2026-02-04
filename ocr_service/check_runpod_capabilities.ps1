# Check RunPod File Transfer Capabilities
# Tests Proxy SSH, Remote SSH Service, and Direct TCP Connection

$sshKey = "$env:USERPROFILE\.ssh\runpod_key"
$proxyHost = "1o82atl93pl1v7-64411d79@ssh.runpod.io"
$directIP = "213.173.102.207"
$directPort = "47566"

Write-Host "=== RunPod Connection Diagnostic ===" -ForegroundColor Cyan
Write-Host ""

# 1. Test Basic Proxy SSH (Always should work)
Write-Host "[1/3] Testing Basic Proxy SSH..." -ForegroundColor Yellow
$proxyTest = ssh -i $sshKey -o BatchMode=yes -o StrictHostKeyChecking=no $proxyHost "echo OK" 2>$null

if ($proxyTest -eq "OK") {
    Write-Host "✓ Proxy SSH is WORKING" -ForegroundColor Green
    Write-Host "  - Supported: runpodctl, manual copy-paste" -ForegroundColor Gray
}
else {
    Write-Host "✗ Proxy SSH is FAILING" -ForegroundColor Red
    Write-Host "  check your SSH key and internet connection." -ForegroundColor Gray
    exit 1
}

# 2. Test SSH Daemon Status (Required for SCP)
Write-Host "`n[2/3] Checking Remote SSH Service (sshd)..." -ForegroundColor Yellow
$sshdStatus = ssh -i $sshKey $proxyHost "service ssh status" 2>$null

if ($sshdStatus -match "is running") {
    Write-Host "✓ Remote SSH Service is RUNNING" -ForegroundColor Green
}
else {
    Write-Host "✗ Remote SSH Service is STOPPED" -ForegroundColor Red
    Write-Host "  - SCP will NOT work until started." -ForegroundColor Gray
    Write-Host "  - Fix: ssh ... 'service ssh start'" -ForegroundColor Gray
}

# 3. Test Direct TCP Connection (Required for SCP)
Write-Host "`n[3/3] Testing Direct TCP Port ($directIP : $directPort)..." -ForegroundColor Yellow
try {
    $tcpTest = Test-NetConnection -ComputerName $directIP -Port $directPort -WarningAction SilentlyContinue
    
    if ($tcpTest.TcpTestSucceeded) {
        Write-Host "✓ Direct Connection is OPEN" -ForegroundColor Green
        Write-Host "  - Supported: SCP, SFTP, rsync" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Direct Connection is CLOSED/BLOCKED" -ForegroundColor Red
        Write-Host "  - SCP will NOT work." -ForegroundColor Gray
        Write-Host "  - Cause: Firewall, wrong port, or sshd down." -ForegroundColor Gray
    }
}
catch {
    Write-Host "✗ Failed to test network connection." -ForegroundColor Red
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
if ($tcpTest.TcpTestSucceeded -and ($sshdStatus -match "is running")) {
    Write-Host "✅ SCP is FULLY ENABLED." -ForegroundColor Green
}
else {
    Write-Host "⚠️ SCP is DISABLED." -ForegroundColor Yellow
    Write-Host "Use 'runpodctl' or fix the issues above." -ForegroundColor Gray
}
