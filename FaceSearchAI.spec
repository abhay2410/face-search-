import os
import sys

block_cipher = None

# Use absolute paths for the build process
current_dir = os.path.abspath(os.getcwd())

added_files = [
    (os.path.join(current_dir, 'frontend'), 'frontend'),
    (os.path.join(current_dir, 'data'), 'data'),
    (os.path.join(current_dir, '.env'), '.'),
]

# Onnxruntime and Insightface often need hidden imports
hidden_imports = [
    'structlog',
    'fastapi',
    'uvicorn',
    'aioodbc',
    'cv2',
    'insightface',
    'onnxruntime',
    'onnxruntime_gpu',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
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
    name='FaceSearchAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # This makes it run in background without CMD window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='frontend/favicon.ico' if os.path.exists('frontend/favicon.ico') else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceSearchAI',
)
