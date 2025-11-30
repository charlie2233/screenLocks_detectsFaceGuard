FaceLock Guardian v2
====================

Local Windows tray application that locks the workstation when the authorized user is not detected, using YOLOv8-face + ArcFace ONNX (no dlib/face_recognition). Includes anti-spoofing (blink + motion), encrypted storage, autostart, and PyInstaller build.

## Features
- YOLOv8n-face detection, ArcFace ONNX embeddings (112×112 crop), cosine similarity.
- Anti-spoofing: blink check from YOLO landmarks + optical-flow micro-motion; require similarity AND blink AND motion.
- Lock workstation after N seconds without an authorized live face; optional unlock automation (Ctrl+Alt+Del + password).
- Encrypted local storage (DPAPI) for embedding and password under `%APPDATA%/FaceLockGuardian`.
- PyQt6 tray app: pause/resume, settings window (enroll face, thresholds, lock delay, camera, autostart, password).
- Auto-download models on first run (`yolov8n-face.pt`, `arcface_r100.onnx`).
- PyInstaller spec for single-folder build including models.

## Requirements
- Windows, Python 3.12
- `pip install -r requirements.txt`

## Running
```powershell
python main.py
```
Tray icon appears. Open Settings to enroll the authorized face and set an unlock password (optional). Adjust similarity threshold and lock delay.

## Building (PyInstaller)
```powershell
pyinstaller facelock_guardian/build.spec
```
Creates `dist/FaceLockGuardian/` with the executable and bundled models.

## Notes
- Unlock simulation types the stored password after Ctrl+Alt+Del; use on a test account and ensure screen focus is correct.
- Anti-spoofing is lightweight; for production consider adding depth/IR. Keep ambient lighting stable.
- Models download to `facelock_guardian/models/` automatically.
