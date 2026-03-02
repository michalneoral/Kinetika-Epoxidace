Param(
    [string]$Python = "py",
    [string]$VenvDir = ".venv-build",
    [string]$Spec = "build\\FAME_EPO_Manager.spec",
    # PyInstaller's default work directory is "build" which would clash with our repo's build/ scripts.
    # Use a dedicated folder to avoid deleting our own scripts/spec when cleaning.
    [string]$WorkDir = "pyi_build",
    [string]$DistDir = "dist",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Ensure we run from the repository root (important for relative paths in the .spec and scripts)
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "==> Repo root: $repoRoot"

Write-Host "==> Creating build venv: $VenvDir"
if (!(Test-Path $VenvDir)) {
    & $Python -m venv $VenvDir
}

$pip = Join-Path $VenvDir "Scripts\\pip.exe"
$py  = Join-Path $VenvDir "Scripts\\python.exe"

Write-Host "==> Upgrading pip"
& $py -m pip install --upgrade pip

# Install project deps
if (Test-Path "requirements.txt") {
    Write-Host "==> Installing requirements.txt"
    & $pip install -r requirements.txt
} else {
    Write-Warning "requirements.txt not found in repo root. Installing only PyInstaller + NiceGUI build helpers."
}

Write-Host "==> Installing PyInstaller"
& $pip install pyinstaller

Write-Host "==> Generating Inno Setup defines (version)"
& $py build\generate_inno_defines.py

Write-Host "==> Building: $Spec"
if (!(Test-Path $Spec)) {
    throw "Spec file not found: $Spec (did you copy the build/ folder from the release ZIP?)"
}
$arguments = @(
    "--noconfirm",
    "--clean",
    "--workpath", $WorkDir,
    "--distpath", $DistDir,
    $Spec
)

if ($Clean) {
    Write-Host "==> Cleaning dist/work folders"
    if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
    if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
}

& $py -m PyInstaller @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "==> Done. Output: $DistDir\\FAME_EPO_Manager\\FAME_EPO_Manager.exe"
