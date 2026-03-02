# Build Windows EXE + Installer (PyInstaller + Inno Setup)

Tento repozitář je připravený na build **Windows EXE** (PyInstaller) a následně na vytvoření **instalátoru Setup.exe** (Inno Setup).

## 0) Verze + GitHub updater

- Verze aplikace je v: `experimental_web/core/version.py` (`__version__`).
- Build skript automaticky vygeneruje `installer/_version.iss`, aby Inno Setup nemusel mít verzi natvrdo.
- Doporučený release flow:
  1. Změň `__version__`.
  2. Buildni EXE + instalátor.
  3. Na GitHubu vytvoř Release s tagem `vX.Y.Z`.
  4. Nahraj asset: `FAME_EPO_Manager_Setup-X.Y.Z.exe` (název vzniká automaticky z Inno Setup skriptu).
  5. (Volitelné) Vygeneruj `checksums.txt` a nahraj také (updater umí checksum ověřit):

     ```bat
     py -3 build\generate_checksums.py installer\FAME_EPO_Manager_Setup-X.Y.Z.exe
     ```

Updater (běží na pozadí při startu) kontroluje GitHub Releases a pokud najde novější verzi, stáhne setup a spustí ho v silent režimu.
Ve výchozím stavu běží jen ve "frozen" buildu (PyInstaller), ne při klasickém `python experimental_web/app.py`.
Repo/asset prefix lze přenastavit env proměnnými:

- `EXPERIMENTAL_WEB_UPDATE_OWNER`
- `EXPERIMENTAL_WEB_UPDATE_REPO`
- `EXPERIMENTAL_WEB_UPDATE_ASSET_PREFIX`

Updater lze vypnout pomocí `EXPERIMENTAL_WEB_DISABLE_UPDATE=1`.

## 1) Předpoklady

- Windows 10/11 64-bit
- Python 3.11+ (doporučeno), dostupný jako `py`
- (pro instalátor) Inno Setup Compiler

Pozn.: Build dělej vždy na Windows (PyInstaller produkuje platform-specific binárky).

## 2) EXE (PyInstaller, onedir)

V kořeni repa:

```bat
build\build_pyinstaller.bat -Clean
```

nebo PowerShell přímo:

```powershell
.\build\build_pyinstaller.ps1 -Clean
```

Výstup bude:

- `dist\FAME_EPO_Manager\FAME_EPO_Manager.exe`

### Ikona (volitelné)

Pokud chceš ikonu, dej ji jako:

- `build\icon.ico`

Spec soubor ji použije automaticky.

### Debug / logy

- aplikace podporuje `--debug` (0..4 nebo bez hodnoty)
- v `build\FAME_EPO_Manager.spec` lze přepnout `console=False/True`

## 3) Instalátor (Setup.exe)

1. Nejdřív vyrob EXE (krok 2).
2. Otevři `installer\FAME_EPO_Manager.iss` v Inno Setup Compiler.
3. Klikni **Build**.

Výstup (ve složce `installer\`):

- `FAME_EPO_Manager_Setup-<verze>.exe`

Pozn.: Instalátor je nastavený jako **per-user** (instaluje do `{localappdata}` a nevyžaduje admin práva).
To je důležité pro plně tiché auto-updaty přes GitHub (bez UAC promptu).

## 4) Poznámky k ukládání dat

Aplikace ukládá DB a NiceGUI persistence soubory do uživatelského data adresáře přes `platformdirs`.
Installer i updater jsou nastavené pro **per-user** instalaci, takže update může probíhat v tichém režimu
bez admin práv.


## Tray ikona

Ve frozen (PyInstaller) buildu se na Windows automaticky spustí tray ikona (otevřít / aktualizace / ukončit).
Pro vypnutí lze nastavit:
- `EXPERIMENTAL_WEB_DISABLE_TRAY=1`
