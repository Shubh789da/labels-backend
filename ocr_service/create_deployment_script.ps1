# Create a single shell script that contains all files as heredocs
# This can be copy-pasted into the RunPod terminal

$outputFile = "deploy_all_files.sh"

$scriptContent = @"
#!/bin/bash
# Deploy all OCR service files to RunPod
# Usage: Copy this entire script and paste it into your RunPod SSH terminal

set -e
cd /workspace/ocr_service
mkdir -p /workspace/ocr_service

echo "Creating files..."

# config.py
cat > config.py << 'EOF_CONFIG'
$(Get-Content config.py -Raw)
EOF_CONFIG
echo "✓ config.py"

# pdf_processor.py
cat > pdf_processor.py << 'EOF_PDF'
$(Get-Content pdf_processor.py -Raw)
EOF_PDF
echo "✓ pdf_processor.py"

# ocr_model.py
cat > ocr_model.py << 'EOF_OCR'
$(Get-Content ocr_model.py -Raw)
EOF_OCR
echo "✓ ocr_model.py"

# s3_service.py
cat > s3_service.py << 'EOF_S3'
$(Get-Content s3_service.py -Raw)
EOF_S3
echo "✓ s3_service.py"

# main.py
cat > main.py << 'EOF_MAIN'
$(Get-Content main.py -Raw)
EOF_MAIN
echo "✓ main.py"

# requirements.txt
cat > requirements.txt << 'EOF_REQ'
$(Get-Content requirements.txt -Raw)
EOF_REQ
echo "✓ requirements.txt"

# start_server.sh
cat > start_server.sh << 'EOF_START'
$(Get-Content start_server.sh -Raw)
EOF_START
chmod +x start_server.sh
echo "✓ start_server.sh"

echo ""
echo "=== All files created successfully ==="
ls -lh
"@

# Save to file
$scriptContent | Out-File -FilePath $outputFile -Encoding UTF8 -NoNewline

Write-Host "Created deployment script: $outputFile" -ForegroundColor Green
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host "1. SSH into your RunPod instance" -ForegroundColor Yellow
Write-Host "2. Open the file: $outputFile in a text editor" -ForegroundColor Yellow  
Write-Host "3. Copy ALL contents" -ForegroundColor Yellow
Write-Host "4. Paste into your RunPod SSH terminal" -ForegroundColor Yellow
Write-Host "5. Press Enter to execute" -ForegroundColor Yellow
Write-Host ""
Write-Host "Alternative: Open $outputFile to view the script" -ForegroundColor Gray
