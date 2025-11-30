import base64
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from facelock_guardian import config
from facelock_guardian.services import secrets, storage

MODEL_URLS = [config.FACE_EMBEDDING_MODEL_URL, *getattr(config, "FACE_EMBEDDING_MODEL_MIRRORS", [])]


class RecognitionEngine:
    def __init__(self, similarity_threshold: float) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedding = self._load_embedding()
        self.yolo = None
        self.arcface_session = None
        self.lock = threading.Lock()
        storage.ensure_data_dir()
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_models()

    def _ensure_models(self) -> None:
        # YOLO auto-download
        yolo_path = config.MODELS_DIR / config.YOLO_MODEL_NAME
        if not yolo_path.exists():
            import requests

            resp = requests.get(config.YOLO_MODEL_URL, timeout=60)
            resp.raise_for_status()
            yolo_path.write_bytes(resp.content)
        self.yolo = YOLO(str(yolo_path))
        # ArcFace ONNX
        self.arcface_session = self._load_or_download_embedding()

    def _load_or_download_embedding(self) -> ort.InferenceSession:
        last_error: Optional[Exception] = None
        if config.FACE_EMBEDDING_MODEL_PATH.exists():
            try:
                return ort.InferenceSession(
                    str(config.FACE_EMBEDDING_MODEL_PATH),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                last_error = exc
                config.FACE_EMBEDDING_MODEL_PATH.unlink(missing_ok=True)
        for url in MODEL_URLS:
            try:
                self._download_file(url, config.FACE_EMBEDDING_MODEL_PATH)
                return ort.InferenceSession(
                    str(config.FACE_EMBEDDING_MODEL_PATH),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                last_error = exc
                config.FACE_EMBEDDING_MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to download face embedding model from provided mirrors: {last_error}")

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        import requests

        with requests.get(url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            with destination.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

    def _load_embedding(self) -> Optional[np.ndarray]:
        arr = secrets.load_embedding()
        if arr is None:
            return None
        return np.array(arr, dtype=np.float32)

    def save_embedding(self, embedding: np.ndarray) -> None:
        secrets.store_embedding(embedding.astype(np.float32))
        self.embedding = embedding.astype(np.float32)

    def has_enrollment(self) -> bool:
        return self.embedding is not None

    def set_similarity_threshold(self, value: float) -> None:
        self.similarity_threshold = value

    def detect_and_embed(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[dict]]:
        results = self.yolo.predict(source=frame_bgr, imgsz=640, conf=0.5, verbose=False)
        if not results:
            return None, None
        boxes = results[0].boxes
        if boxes is None or len(boxes) != 1:
            return None, None
        det = boxes[0]
        bbox = det.xyxy[0].cpu().numpy().astype(int)
        keypoints = None
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints[0].cpu().numpy()
        x1, y1, x2, y2 = bbox
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None, None
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        face_norm = face_resized.astype(np.float32) / 255.0
        face_norm = (face_norm - 0.5) / 0.5
        # CHW
        input_blob = np.transpose(face_norm, (2, 0, 1))[np.newaxis, ...]
        embedding = self.arcface_session.run(None, {"data": input_blob})[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        info = {"bbox": bbox, "keypoints": keypoints}
        return embedding.squeeze(), info

    def is_authorized(self, embedding: np.ndarray) -> tuple[bool, float]:
        if self.embedding is None:
            return False, 0.0
        sim = float(np.dot(embedding, self.embedding.T))
        return sim >= self.similarity_threshold, sim

    def clear_embedding(self) -> None:
        secrets.delete_embedding()
        self.embedding = None
