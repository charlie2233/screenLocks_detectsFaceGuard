import time

import cv2
from PyQt6 import QtCore
# import config
# print(">>> USING CAMERA INDEX:", config.CAMERA_INDEX)

from facelock_guardian import config
from facelock_guardian.core.recognition import RecognitionEngine
from facelock_guardian.core import spoof
from facelock_guardian.services import locker, secrets


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Camera failed to open with MSMF. Trying DirectShow...")
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    return cap


class CaptureWorker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    locked = QtCore.pyqtSignal()
    authorized = QtCore.pyqtSignal(float)
    liveness = QtCore.pyqtSignal(dict)

    def __init__(self, similarity_threshold: float, lock_delay: int, camera_index: int, static_threshold: float) -> None:
        super().__init__()
        self.engine = RecognitionEngine(similarity_threshold)
        self.lock_delay = lock_delay
        self.camera_index = camera_index
        self.static_threshold = static_threshold
        self.running = False
        self.paused = False
        self.last_seen = time.monotonic()
        self.last_unlock_time = 0.0
        self.spoof_state = spoof.SpoofState()
        self.password = secrets.load_password()

    def has_enrollment(self) -> bool:
        return self.engine.has_enrollment()

    def set_threshold(self, threshold: float) -> None:
        self.engine.set_similarity_threshold(threshold)

    def set_lock_delay(self, seconds: int) -> None:
        self.lock_delay = seconds

    def set_camera(self, index: int) -> None:
        self.camera_index = index

    def set_static_threshold(self, value: float) -> None:
        self.static_threshold = value

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def stop(self) -> None:
        self.running = False

    def _empty_liveness(self) -> dict:
        return {"blink_ok": False, "motion_ok": False, "motion": 0.0, "static_ok": False, "variance": 0.0}

    def enroll(self) -> bool:
        cap = open_camera()
        if not cap.isOpened():
            self.status.emit("Camera not available")
            return False
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.status.emit("Failed to capture")
            return False
        embedding, _ = self.engine.detect_and_embed(frame)
        if embedding is None:
            self.status.emit("No face found")
            return False
        self.engine.save_embedding(embedding)
        self.status.emit("Face enrolled")
        return True

    def clear_enrollment(self) -> None:
        self.engine.clear_embedding()
        self.status.emit("Enrollment removed")

    def refresh_password(self) -> None:
        self.password = secrets.load_password()

    def run(self) -> None:
        self.running = True
        cap = None
        while self.running:
            if not self.engine.has_enrollment():
                if cap is not None:
                    cap.release()
                    cap = None
                self.status.emit("Enrollment required")
                self.last_seen = time.monotonic()
                time.sleep(1.0)
                continue
            if cap is None:
                cap = open_camera()
                if not cap.isOpened():
                    self.status.emit("Camera not available")
                    time.sleep(1.0)
                    continue
                self.status.emit("Monitoring started")
            if self.paused:
                time.sleep(0.2)
                continue
            ret, frame = cap.read()
            if not ret:
                self.liveness.emit(self._empty_liveness())
                time.sleep(0.1)
                continue
            embedding, info = self.engine.detect_and_embed(frame)
            if embedding is not None and info is not None:
                bbox = info["bbox"]
                kp = info["keypoints"]
                authorized, sim = self.engine.is_authorized(embedding)
                live = False
                liveness_details = self._empty_liveness()
                live, liveness_details = spoof.liveness_score(
                    frame,
                    tuple(bbox),
                    kp,
                    sim,
                    self.engine.similarity_threshold,
                    self.static_threshold,
                    self.spoof_state,
                )
                self.liveness.emit(liveness_details)
                if authorized and live:
                    current_time = time.monotonic()
                    self.last_seen = current_time
                    self.last_unlock_time = current_time
                    self.authorized.emit(sim)
                    locker.unlock_screen(self.password)
                else:
                    self.status.emit("Face not authorized or spoof")
            else:
                self.liveness.emit(self._empty_liveness())
            now = time.monotonic()
            if now - self.last_seen > self.lock_delay:
                if now - self.last_unlock_time < config.UNLOCK_GRACE_PERIOD:
                    self.status.emit("Unlock grace period active")
                else:
                    locker.lock_screen()
                    self.locked.emit()
                    self.last_seen = now
            time.sleep(0.05)
        if cap is not None:
            cap.release()
        self.status.emit("Monitoring stopped")
