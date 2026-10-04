$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TessDir = Join-Path $ProjectRoot 'assets\tessdata'
New-Item -ItemType Directory -Force -Path $TessDir | Out-Null
Write-Host 'Installing local OCR language packs into:' $TessDir
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata' -OutFile (Join-Path $TessDir 'eng.traineddata')
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/rus.traineddata' -OutFile (Join-Path $TessDir 'rus.traineddata')
Write-Host 'Done. Restart the GeoAI application. OCR will use eng+rus automatically.' -ForegroundColor Green
