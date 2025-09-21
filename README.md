TODO: add readme

How to compile

```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --clean --collect-all nicegui --collect-data latex2mathml -n kinetika --onefile -i build/icon.ico app/kinetika_webapp.py
```
