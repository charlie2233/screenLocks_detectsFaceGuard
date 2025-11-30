import sys
import threading

from PyQt6 import QtGui, QtWidgets

from facelock_guardian import config
from facelock_guardian.core.capture import CaptureWorker
from facelock_guardian.services import storage
from facelock_guardian.ui.settings import SettingsWindow
from facelock_guardian.ui.tray import TrayApp


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    cfg = storage.load_config()
    worker = CaptureWorker(
        similarity_threshold=float(cfg.get("similarity_threshold", config.DEFAULT_SIMILARITY_THRESHOLD)),
        lock_delay=int(cfg.get("lock_delay", config.DEFAULT_LOCK_DELAY)),
        camera_index=int(cfg.get("camera_index", config.DEFAULT_CAMERA_INDEX)),
        static_threshold=float(cfg.get("static_threshold", config.DEFAULT_STATIC_VARIANCE_THRESHOLD)),
    )

    settings = SettingsWindow(worker)
    settings.show()
    settings.raise_()
    settings.activateWindow()

    def on_pause():
        worker.set_paused(True)
        settings.pause_checkbox.setChecked(True)

    def on_resume():
        worker.set_paused(False)
        settings.pause_checkbox.setChecked(False)

    def on_quit():
        worker.stop()
        QtWidgets.QApplication.quit()

    tray_icon = QtGui.QIcon.fromTheme("camera")
    if tray_icon.isNull():
        tray_icon = app.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon)
    tray = TrayApp(tray_icon, settings, on_pause, on_resume, on_quit)
    tray.show()

    settings.pause_toggled.connect(worker.set_paused)
    settings.threshold_changed.connect(worker.set_threshold)
    settings.lock_delay_changed.connect(worker.set_lock_delay)
    settings.camera_changed.connect(worker.set_camera)
    settings.static_threshold_changed.connect(worker.set_static_threshold)
    worker_thread = {"thread": None, "started": False}

    def start_worker_thread() -> None:
        if worker_thread["started"]:
            return
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        worker_thread["thread"] = thread
        worker_thread["started"] = True

    def handle_enroll() -> None:
        if worker.enroll():
            start_worker_thread()

    settings.enroll_requested.connect(handle_enroll)

    worker.status.connect(lambda msg: settings.setWindowTitle(f"{config.APP_NAME} - {msg}"))
    worker.locked.connect(lambda: settings.setWindowTitle(f"{config.APP_NAME} - Locked"))
    worker.authorized.connect(lambda sim: settings.setWindowTitle(f"{config.APP_NAME} - Authorized ({sim:.2f})"))
    worker.liveness.connect(settings.update_liveness_state)

    if worker.has_enrollment():
        start_worker_thread()
    else:
        worker.status.emit("Enrollment required")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
