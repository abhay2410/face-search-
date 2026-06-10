# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add data files (Models, Config, etc.)
# Format: (Source, Destination)
added_files = [
    ('data/insightface_models', 'data/insightface_models'),
    ('.env', '.'),
]

a = Analysis(
    ['face_check.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'insightface',
        'onnxruntime',
        'cryptography',
        'pyodbc',
        'cv2'
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
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceSearch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='data/icon.ico' if os.path.exists('data/icon.ico') else None
)
