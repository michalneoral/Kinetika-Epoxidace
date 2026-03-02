; Inno Setup script to create a classic Windows installer (Setup.exe)
; Usage (after building with PyInstaller onedir):
;   1) dist\FAME_EPO_Manager\FAME_EPO_Manager.exe must exist
;   2) Open this .iss in Inno Setup Compiler and Build

; NOTE:
; - AppSlug is a stable, filesystem-safe identifier (used for install folder, filenames, updater asset prefix)
; - AppDisplayName is what users see in Start menu / installer UI
#define AppSlug "FAME_EPO_Manager"
#define AppDisplayName "FAME EPO Manažer"
#define AppExe  "FAME_EPO_Manager.exe"
#define DistDir "..\\dist\\FAME_EPO_Manager"

; Auto-generated version define (created by: build\generate_inno_defines.py)
; If missing, run:  py -3 build\generate_inno_defines.py
#include "_version.iss"

#ifexist "..\\build\\icon.ico"
  #define SetupIcon "..\\build\\icon.ico"
#endif

[Setup]
; Keep AppId stable across versions to enable in-place upgrades.
AppId={{C5BC7DE4-EC3C-4E44-B8D6-774E6F2D8893}}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppPublisher=FAME_EPO_MN_KD_UPCE
; Per-user install (recommended for fully silent self-updates without UAC)
PrivilegesRequired=lowest
DefaultDirName={localappdata}\{#AppSlug}
DefaultGroupName={#AppDisplayName}
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename={#AppSlug}_Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

#ifdef SetupIcon
SetupIconFile={#SetupIcon}
UninstallDisplayIcon={app}\{#AppExe}
#endif

[Languages]
Name: "czech"; MessagesFile: "compiler:Languages\\Czech.isl"

[Tasks]
Name: "desktopicon"; Description: "Vytvořit ikonu na ploše"; GroupDescription: "Ikony:"; Flags: unchecked

[Files]
Source: "{#DistDir}\\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\\{#AppDisplayName}"; Filename: "{app}\\{#AppExe}"
Name: "{group}\\{#AppDisplayName} - zkontrolovat aktualizace"; Filename: "{app}\\{#AppExe}"; Parameters: "--check-updates"; WorkingDir: "{app}"
Name: "{group}\\{#AppDisplayName} - ukončit aplikaci"; Filename: "{app}\\{#AppExe}"; Parameters: "--quit"; WorkingDir: "{app}"
; Use per-user desktop to avoid requiring admin privileges
Name: "{userdesktop}\\{#AppDisplayName}"; Filename: "{app}\\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\\{#AppExe}"; Description: "Spustit {#AppDisplayName}"; Flags: nowait postinstall skipifsilent
