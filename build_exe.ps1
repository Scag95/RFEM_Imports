param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot "venv\\Scripts\\python.exe"

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

Write-Host "Using Python:" $pythonExe

& $pythonExe -m pip install -r (Join-Path $projectRoot "requirements-build.txt")

if ($LASTEXITCODE -ne 0) {
    throw "Failed to install build dependencies."
}

$pyInstallerArgs = @(
    "-m",
    "PyInstaller",
    "--noconfirm"
)

if ($Clean) {
    $pyInstallerArgs += "--clean"
}

$pyInstallerArgs += (Join-Path $projectRoot "rfem_imports.spec")

& $pythonExe @pyInstallerArgs

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

Write-Host ""
Write-Host "Build completed."
Write-Host "Executable folder:" (Join-Path $projectRoot "dist\\RFEM_Imports")
