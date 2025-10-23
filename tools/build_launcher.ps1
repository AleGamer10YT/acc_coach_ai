param(
    [string]$OutputName = "ACCCoachLauncher",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Assert-Command($command) {
    try {
        & $command --version > $null 2>&1
    } catch {
        Write-Error "Comando '$command' non trovato. Assicurati che sia installato e nel PATH."
        exit 1
    }
}

Assert-Command "python"

$root = Resolve-Path "$PSScriptRoot\.."
$entryPoint = Join-Path $root "launcher\app.py"

if ($Clean) {
    Write-Host "Pulizia cartelle build/dist..." -ForegroundColor Yellow
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue (Join-Path $root "build")
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue (Join-Path $root "dist")
}

Write-Host "Compilazione launcher in corso..." -ForegroundColor Cyan

python -m PyInstaller `
    "$entryPoint" `
    --name $OutputName `
    --onefile `
    --noconsole `
    --clean `
    --collect-all services `
    --collect-all shared `
    --hidden-import PySide6.QtXml

$exePath = Join-Path $root "dist\$OutputName.exe"

if (Test-Path $exePath) {
    Write-Host "Build completata con successo: $exePath" -ForegroundColor Green
} else {
    Write-Error "Compilazione fallita. Controlla l'output di PyInstaller."
}
