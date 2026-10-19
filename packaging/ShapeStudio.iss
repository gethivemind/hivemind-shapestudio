; Inno Setup script for Presidium Shape Studio.
; Compiled by the release workflow on windows-latest:
;   iscc /DMyAppVersion=0.3.4 /DSourceDir=..\dist\ShapeStudio /DOutDir=..\dist packaging\ShapeStudio.iss
; Installs per user by default (no administrator prompt; %LOCALAPPDATA%\Programs), registers the
; app for Start, search and Pin to Start, adds an uninstaller to Settings > Apps.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\ShapeStudio"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

#define MyAppName "Presidium Shape Studio"
#define MyAppShort "Shape Studio"
#define MyAppPublisher "Presidium"
#define MyAppURL "https://presidiumdev.github.io/presidium-shapestudio/"
#define MyAppExeName "ShapeStudio.exe"

[Setup]
AppId={{9B3E6A41-5C2D-4F8A-8D1E-2F7B4C6A9E13}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL=https://github.com/presidiumdev/presidium-shapestudio/issues
AppUpdatesURL=https://github.com/presidiumdev/presidium-shapestudio/releases
AppCopyright=Copyright (C) 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
DefaultDirName={autopf}\Presidium Shape Studio
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename=ShapeStudio-{#MyAppVersion}-Setup
SetupIconFile=ShapeStudio.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
InfoBeforeFile=README-WINDOWS.txt
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start Shape Studio when I sign in (sits in the tray, ready for your phone)"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\TRADEMARK.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "README-WINDOWS.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Your own 3D studio. Free, local, honest."; AppUserModelID: "Hivemind.ShapeStudio"
Name: "{autodesktop}\{#MyAppShort}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; AppUserModelID: "Hivemind.ShapeStudio"
Name: "{userstartup}\{#MyAppShort}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--no-browser"; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; the app's own folder; models, images and settings under %LOCALAPPDATA%\shapestudio are kept on purpose
Type: filesandordirs; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    MsgBox('Shape Studio has been removed. Your models, images and settings were kept in' + #13#10 +
           ExpandConstant('{localappdata}\shapestudio') + #13#10 +
           'Delete that folder yourself if you want the disk space back (models are several GB).',
           mbInformation, MB_OK);
end;
