; build/kinetika.iss
#define MyAppName "Kinetika_Epoxidace"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Michal_Neoral_Kristyna_Dvorakova_for_UPCE"
#define MyAppExeName "Kinetika_Epoxidace.exe"

[Setup]
AppId={{7B0F2D9C-1E8D-4B3D-9E48-1B9CCB6F8A11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=myapp-setup-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[Files]
; Point to your PyInstaller output
Source: "..\dist\myapp.exe"; DestDir: "{app}"; Flags: ignoreversion
; If you have extra files or folders, add them:
; Source: "..\dist\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
