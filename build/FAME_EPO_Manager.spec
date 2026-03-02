# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# NOTE:
# PyInstaller executes .spec files via exec() and does NOT guarantee that __file__ is defined
# (this can vary across PyInstaller versions). Therefore, avoid relying on __file__.
# Our build scripts run PyInstaller from the repository root, so CWD is the project root.
cwd = os.path.abspath(os.getcwd())
if os.path.basename(cwd).lower() == 'build' and os.path.exists(os.path.join(cwd, 'FAME_EPO_Manager.spec')):
    project_root = os.path.abspath(os.path.join(cwd, '..'))
else:
    project_root = cwd

# Optional icon
_icon = os.path.join(project_root, 'build', 'icon.ico')
icon = _icon if os.path.exists(_icon) else None

# Collect NiceGUI (and its runtime deps)
datas, binaries, hiddenimports = [], [], []

d, b, h = collect_all('nicegui')
datas += d
binaries += b
hiddenimports += h

# latex2mathml ships data files used for math rendering
try:
    datas += collect_data_files('latex2mathml')

except Exception:
    # If latex2mathml isn't installed, the build environment will fail earlier anyway.
    pass

# --- Tray icon support (Windows) ---
_tray_icon = os.path.join(project_root, 'build', 'icon.ico')
if os.path.exists(_tray_icon):
    datas += [(_tray_icon, 'build')]

# --- App static files (favicon etc.) ---
_favicon = os.path.join(project_root, 'static', 'favicon.ico')
if os.path.exists(_favicon):
    datas += [(_favicon, 'static')]

# pystray loads platform backends dynamically; keep them as hidden imports.
hiddenimports += ['pystray', 'pystray._win32']

# Matplotlib optional backends used by export (PDF/SVG).
hiddenimports += ['matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_svg']

# If you hit a missing-module error at runtime, add it here.
# Example:
# hiddenimports += ['some_dynamic_imported_module']


entry_script = os.path.join(project_root, 'pack_entry.py')
if not os.path.exists(entry_script):
    entry_script = os.path.join(project_root, 'build', 'pack_entry.py')

# IMPORTANT: keep the Analysis() object assigned to variable `a`.
# Otherwise later stages (PYZ/EXE/COLLECT) will fail.
a = Analysis(
    [entry_script],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FAME_EPO_Manager',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # set True if you want a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FAME_EPO_Manager',
)
