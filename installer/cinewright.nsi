; Cinewright Windows installer (NSIS 3, Modern UI 2, Unicode).
;
; Built by scripts/build.py, which passes VERSION, WHEEL, UV_EXE, LOCK_FILE, ICON, LICENSE_FILE and OUTFILE.
; Per-user install without elevation; supports /S (silent) and /D=<folder>.
; The installer carries uv.exe, the Cinewright wheel and requirements.lock (the dependencies pinned by
; sha256). uv downloads Python 3.12 and the locked dependencies at install time, so setup needs an internet connection. No telemetry.

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma

!macro REQUIRE_DEFINE NAME
  !ifndef ${NAME}
    !error "${NAME} is not defined; build the installer with scripts/build.py"
  !endif
!macroend
!insertmacro REQUIRE_DEFINE VERSION
!insertmacro REQUIRE_DEFINE WHEEL
!insertmacro REQUIRE_DEFINE UV_EXE
!insertmacro REQUIRE_DEFINE LOCK_FILE
!insertmacro REQUIRE_DEFINE ICON
!insertmacro REQUIRE_DEFINE LICENSE_FILE
!ifndef OUTFILE
  !define OUTFILE "CinewrightSetup-${VERSION}.exe"
!endif

; build.py only accepts pure-Python wheels, so the file name follows from the version.
!define WHEEL_NAME "cinewright-${VERSION}-py3-none-any.whl"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Cinewright"
!define DATA_DIR "$LOCALAPPDATA\Cinewright"
!define PYTHONW "$INSTDIR\runtime\Scripts\pythonw.exe"

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "x64.nsh"

Name "Cinewright"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\Cinewright"
InstallDirRegKey HKCU "${UNINST_KEY}" "InstallLocation"
RequestExecutionLevel user
ShowInstDetails show
ShowUninstDetails show
BrandingText "Cinewright ${VERSION}"

; Pages

!define MUI_ICON "${ICON}"
!define MUI_UNICON "${ICON}"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Install Cinewright ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "Cinewright generates images, video and audio on this computer with open-weight models.$\r$\n$\r$\nSetup installs it for your user account and needs no administrator rights. It downloads Python 3.12 and Cinewright's components during installation, so an internet connection is required."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${LICENSE_FILE}"
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE DirectoryLeave
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Start Cinewright"
!define MUI_FINISHPAGE_RUN_FUNCTION StartCinewright
!define MUI_FINISHPAGE_SHOWREADME
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Create a desktop shortcut"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION CreateDesktopShortcut
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; Helpers

!macro SET_ENV NAME VALUE
  System::Call 'kernel32::SetEnvironmentVariable(t "${NAME}", t "${VALUE}")'
!macroend
!macro CLEAR_ENV NAME
  System::Call 'kernel32::SetEnvironmentVariable(t "${NAME}", p 0)'
!macroend
; uv creates paths longer than MAX_PATH (e.g. pip's __pycache__ inside the managed Python), which
; RMDir deletes only in the \\?\ form. The plain form follows for what that form cannot express,
; such as network shares.
!macro RMDIR_TREE DIR
  RMDir /r "\\?\${DIR}"
  RMDir /r "${DIR}"
!macroend

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "Cinewright needs 64-bit Windows." /SD IDOK
    Abort
  ${EndIf}
FunctionEnd

; Sets $R0 to an error message when $INSTDIR must not be used, otherwise to "".
; The uninstaller deletes what setup put into the folder, so setup refuses the user data folder
; and non-empty folders that do not hold a Cinewright installation.
Function ValidateInstallDir
  StrCpy $R0 ""
  ${If} "$INSTDIR" == "${DATA_DIR}"
    StrCpy $R0 "$INSTDIR holds Cinewright's data (models, results, settings). Choose a different folder for the program."
    Return
  ${EndIf}
  ${If} ${FileExists} "$INSTDIR\uninstall.exe"
  ${AndIf} ${FileExists} "$INSTDIR\cinewright.ico"
    Return
  ${EndIf}
  FindFirst $R1 $R2 "$INSTDIR\*"
  ${DoUntil} $R2 == ""
    ${If} $R2 != "."
    ${AndIf} $R2 != ".."
      StrCpy $R0 "$INSTDIR is not empty and holds no Cinewright installation. Choose an empty or new folder."
      ${Break}
    ${EndIf}
    FindNext $R1 $R2
  ${Loop}
  FindClose $R1
FunctionEnd

Function DirectoryLeave
  Call ValidateInstallDir
  ${If} $R0 != ""
    MessageBox MB_OK|MB_ICONEXCLAMATION "$R0"
    Abort
  ${EndIf}
FunctionEnd

Function WriteUninstallEntry
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "Cinewright"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "Cinewright"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\cinewright.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" "$0"
FunctionEnd

Function StartCinewright
  SetOutPath "$INSTDIR"
  Exec '"${PYTHONW}" -m cinewright'
FunctionEnd

Function CreateDesktopShortcut
  SetOutPath "$INSTDIR"
  CreateShortCut "$DESKTOP\Cinewright.lnk" "${PYTHONW}" "-m cinewright" "$INSTDIR\cinewright.ico" 0
FunctionEnd

; Install

Section "Cinewright" SecMain
  SectionIn RO

  Call ValidateInstallDir
  ${If} $R0 != ""
    MessageBox MB_OK|MB_ICONSTOP "$R0" /SD IDOK
    Abort "$R0"
  ${EndIf}

  SetOutPath "$INSTDIR"

  ; An upgrade sets the previous runtime aside first: the rename fails while Cinewright is running,
  ; and a failed upgrade can put the old runtime back.
  !insertmacro RMDIR_TREE "$INSTDIR\runtime.old"
  ${If} ${FileExists} "$INSTDIR\runtime\*.*"
    ClearErrors
    Rename "$INSTDIR\runtime" "$INSTDIR\runtime.old"
    ${If} ${Errors}
      MessageBox MB_OK|MB_ICONSTOP "Cinewright is running, so setup cannot replace it. Quit Cinewright and run setup again." /SD IDOK
      Abort "Cinewright is running."
    ${EndIf}
  ${EndIf}

  DetailPrint "Copying files"
  Delete "$INSTDIR\cinewright-*.whl"
  File "/oname=uv.exe" "${UV_EXE}"
  File "/oname=${WHEEL_NAME}" "${WHEEL}"
  File "/oname=requirements.lock" "${LOCK_FILE}"
  File "/oname=LICENSE.txt" "${LICENSE_FILE}"
  File "/oname=cinewright.ico" "${ICON}"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  ; A first install is registered right away so even a failed one can be removed in Settings.
  ${IfNot} ${FileExists} "$INSTDIR\runtime.old\*.*"
    Call WriteUninstallEntry
  ${EndIf}

  ; uv keeps Python and its cache inside the installation folder and ignores uv config files.
  !insertmacro SET_ENV "UV_PYTHON_INSTALL_DIR" "$INSTDIR\python"
  !insertmacro SET_ENV "UV_CACHE_DIR" "$INSTDIR\cache"
  !insertmacro SET_ENV "UV_NO_CONFIG" "1"
  !insertmacro SET_ENV "UV_NO_PROGRESS" "1"

  DetailPrint "Downloading Python 3.12 and creating the Cinewright runtime"
  nsExec::ExecToLog '"$INSTDIR\uv.exe" venv --python 3.12 --managed-python "$INSTDIR\runtime"'
  Pop $0
  ${If} $0 == 0
    DetailPrint "Installing Cinewright's Python packages (pinned versions, hashes verified)"
    nsExec::ExecToLog '"$INSTDIR\uv.exe" pip install --python "$INSTDIR\runtime\Scripts\python.exe" --require-hashes --only-binary :all: -r "$INSTDIR\requirements.lock"'
    Pop $0
  ${EndIf}
  ${If} $0 == 0
    DetailPrint "Installing Cinewright ${VERSION}"
    nsExec::ExecToLog '"$INSTDIR\uv.exe" pip install --python "$INSTDIR\runtime\Scripts\python.exe" --no-deps "$INSTDIR\${WHEEL_NAME}"'
    Pop $0
  ${EndIf}

  !insertmacro CLEAR_ENV "UV_PYTHON_INSTALL_DIR"
  !insertmacro CLEAR_ENV "UV_CACHE_DIR"
  !insertmacro CLEAR_ENV "UV_NO_CONFIG"
  !insertmacro CLEAR_ENV "UV_NO_PROGRESS"

  ${If} $0 != 0
    DetailPrint "uv failed (exit code $0)"
    !insertmacro RMDIR_TREE "$INSTDIR\runtime"
    ${If} ${FileExists} "$INSTDIR\runtime.old\*.*"
      Rename "$INSTDIR\runtime.old" "$INSTDIR\runtime"
      DetailPrint "Restored the previous installation"
    ${EndIf}
    MessageBox MB_OK|MB_ICONSTOP "Setup could not download Python 3.12 or Cinewright's packages, so Cinewright is not installed. Check the internet connection and run setup again; the details list shows the error." /SD IDOK
    Abort "Installation failed: Python or packages could not be installed."
  ${EndIf}
  !insertmacro RMDIR_TREE "$INSTDIR\runtime.old"

  SetOutPath "$INSTDIR"
  CreateShortCut "$SMPROGRAMS\Cinewright.lnk" "${PYTHONW}" "-m cinewright" "$INSTDIR\cinewright.ico" 0
  Call WriteUninstallEntry
SectionEnd

; Uninstall

Section "Uninstall"
  ; Refuse to delete anything unless this uninstaller sits in the folder Setup registered for Cinewright
  ; and that folder holds Cinewright's icon. A copied or moved uninstall.exe would otherwise
  ; delete whatever folder it was started from.
  ClearErrors
  ReadRegStr $R0 HKCU "${UNINST_KEY}" "InstallLocation"
  ${If} ${Errors}
  ${OrIf} "$R0" != "$INSTDIR"
  ${OrIfNot} ${FileExists} "$INSTDIR\cinewright.ico"
    DetailPrint "Uninstall cancelled: $INSTDIR is not the registered Cinewright installation folder, nothing was deleted."
    MessageBox MB_OK|MB_ICONSTOP "This uninstaller is not in the folder where Cinewright is installed ($INSTDIR is not the registered installation folder), so nothing was removed. Remove Cinewright from Settings > Apps instead." /SD IDOK
    Abort "Uninstall cancelled: not the registered installation folder."
  ${EndIf}

  !insertmacro RMDIR_TREE "$INSTDIR\runtime.old"
  ${If} ${FileExists} "$INSTDIR\runtime\*.*"
    ClearErrors
    Rename "$INSTDIR\runtime" "$INSTDIR\runtime.old"
    ${If} ${Errors}
      MessageBox MB_OK|MB_ICONSTOP "Cinewright is running, so it cannot be removed. Quit Cinewright and run the uninstaller again." /SD IDOK
      Abort "Cinewright is running."
    ${EndIf}
  ${EndIf}

  Delete "$SMPROGRAMS\Cinewright.lnk"
  Delete "$DESKTOP\Cinewright.lnk"

  ; Everything setup and uv created in the installation folder.
  !insertmacro RMDIR_TREE "$INSTDIR\runtime.old"
  !insertmacro RMDIR_TREE "$INSTDIR\python"
  !insertmacro RMDIR_TREE "$INSTDIR\cache"
  Delete "$INSTDIR\cinewright-*.whl"
  Delete "$INSTDIR\uv.exe"
  Delete "$INSTDIR\requirements.lock"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\cinewright.ico"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"

  ${If} ${FileExists} "${DATA_DIR}\*.*"
    MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "Delete Cinewright's data as well?$\r$\n$\r$\n${DATA_DIR} holds downloaded models, generated results and settings. Models can take up a lot of space. Choose No to keep them for a later installation." /SD IDNO IDNO keep_data
    RMDir /r "${DATA_DIR}"
    DetailPrint "Deleted ${DATA_DIR}"
    keep_data:
  ${EndIf}
SectionEnd
