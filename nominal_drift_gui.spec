# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Nominal Drift GUI
# Usage: pyinstaller nominal_drift_gui.spec
block_cipher = None

a = Analysis(
    ['nominal_drift/gui/app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('nominal_drift/science/constants/arrhenius.json', 'nominal_drift/science/constants'),
        ('nominal_drift/llm/prompts', 'nominal_drift/llm/prompts'),
    ],
    hiddenimports=[
        'streamlit', 'pydantic', 'numpy', 'scipy', 'matplotlib',
        'sqlalchemy', 'requests', 'nominal_drift',
    ],
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
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='NominalDrift',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name='NominalDrift',
)
