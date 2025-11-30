from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from facelock_guardian import config


def eye_aspect_ratio(eye: np.ndarray) -> float:
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C + 1e-6)


@dataclass
class SpoofState:
    blink_counter: int = 0
    blinks: int = 0
    prev_gray: Optional[np.ndarray] = None
    prev_points: Optional[np.ndarray] = None


def detect_blink_from_keypoints(keypoints: np.ndarray, state: SpoofState) -> bool:
    # YOLOv8-face keypoints order: left_eye(0), right_eye(1), nose(2), mouth_left(3), mouth_right(4)
    if keypoints is None or len(keypoints) < 2:
        return False
    left_eye = keypoints[0][:2]
    right_eye = keypoints[1][:2]
    # Create synthetic eye polygons for EAR approximation
    eye_poly = np.array(
        [
            left_eye,
            left_eye + np.array([0, -1]),
            left_eye + np.array([1, 0]),
            right_eye,
            right_eye + np.array([0, 1]),
            right_eye + np.array([-1, 0]),
        ],
        dtype="float32",
    )
    ear = eye_aspect_ratio(eye_poly)
    if ear < config.EAR_THRESHOLD:
        state.blink_counter += 1
    else:
        if state.blink_counter >= config.EAR_CONSEC_FRAMES:
            state.blinks += 1
        state.blink_counter = 0
    return state.blinks > 0


def movement_score(gray: np.ndarray, points: np.ndarray, state: SpoofState) -> float:
    if points is None or points.size == 0:
        return 0.0
    pts = points.astype(np.float32)
    if state.prev_gray is None or state.prev_points is None:
        state.prev_gray = gray
        state.prev_points = pts
        return 0.0
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(state.prev_gray, gray, state.prev_points, None)
    if next_pts is None or status is None:
        state.prev_gray = gray
        state.prev_points = pts
        return 0.0
    drift = float(np.mean(np.linalg.norm(next_pts - state.prev_points, axis=1)))
    state.prev_gray = gray
    state.prev_points = pts
    return drift


def frame_variance_ok(gray: np.ndarray, bbox: Tuple[int, int, int, int], threshold: float) -> tuple[bool, float]:
    x1, y1, x2, y2 = bbox
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    variance = float(np.var(roi))
    return variance >= threshold, variance


def is_live(blink_ok: bool, motion_ok: bool, static_ok: bool) -> bool:
    return blink_ok and motion_ok and static_ok


def liveness_score(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], keypoints: Optional[np.ndarray],
                   similarity: float, threshold: float, static_threshold: float,
                   state: SpoofState) -> tuple[bool, dict]:
    x1, y1, x2, y2 = bbox
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blink_ok = detect_blink_from_keypoints(keypoints, state)
    motion = movement_score(gray, np.array(keypoints) if keypoints is not None else None, state)
    motion_ok = motion > config.MOTION_MIN_DRIFT
    static_ok, variance = frame_variance_ok(gray, (x1, y1, x2, y2), static_threshold)
    live = similarity >= threshold and is_live(blink_ok, motion_ok, static_ok)
    return live, {
        "blink_ok": blink_ok,
        "motion": motion,
        "motion_ok": motion_ok,
        "static_ok": static_ok,
        "variance": variance,
    }
