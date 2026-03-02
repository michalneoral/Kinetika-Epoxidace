@echo off
setlocal

REM Simple wrapper for PowerShell build script
powershell -ExecutionPolicy Bypass -File "%~dp0build_pyinstaller.ps1" %*

endlocal
