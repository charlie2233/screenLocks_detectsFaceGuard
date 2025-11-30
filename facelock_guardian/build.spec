# PyInstaller spec for FaceLock Guardian v2
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules
from PyQt6 import QtCore

block_cipher = None

project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "facelock_guardian" / "models"

hiddenimports = collect_submodules("ultralytics") + collect_submodules("onnxruntime")

datas = []
face_model = models_dir / "face_embedding.onnx"
if face_model.exists():
    datas.append((str(face_model), "facelock_guardian/models"))
if (models_dir / "yolov8n-face.pt").exists():
    datas.append((str(models_dir / "yolov8n-face.pt"), "facelock_guardian/models"))
qt_plugins = Path(QtCore.__file__).resolve().parent / "Qt6" / "plugins"
for folder in ("platforms", "styles"):
    plugin_dir = qt_plugins / folder
    if plugin_dir.exists():
        datas.append((str(plugin_dir), f"PyQt6/Qt6/plugins/{folder}"))

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    distpath=str(project_root / "dist"),
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceLockGuardian',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
)
