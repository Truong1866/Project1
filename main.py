"""Điểm khởi chạy ứng dụng Smart Vision (giao diện desktop).

    python main.py                       # mở app trống, kéo thả video / chọn camera trong ⚙ Cài đặt
    python main.py video.mp4 rtsp://...  # mở sẵn các nguồn
"""
import argparse
import sys
from BusinessLayer.discord_notifier import DiscordNotifier
try:  # nạp OpenVINO TRƯỚC cv2 / onnxruntime (tránh xung đột DLL trên một số máy Windows)
    import openvino  # noqa: F401
except ImportError:
    pass

from PySide6.QtWidgets import QApplication

from BusinessLayer.pipeline import SmartVisionPipeline
from DataLayer.Repositories.event_repository import EventRepository
from DataLayer.sqlite_db import SQLiteDB
from DataLayer.vector_db import FaceDatabase
from PresentLayer.main_window import MainWindow
from Utils.config import Config
from Utils.event_bus import EventBus
from Utils.logger import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Vision")
    parser.add_argument("sources", nargs="*", help="file video hoặc URL camera (rtsp://, http://)")
    args = parser.parse_args()

    cfg = Config.load()
    setup_logging(cfg.get("app.log_level", "INFO"), cfg.root / "logs")

    db = FaceDatabase(threshold=float(cfg.get("face.threshold", 0.45)))
    repo = EventRepository(SQLiteDB(cfg.path("events.db_path", "DataLayer/events.db")))
    bus = EventBus()
    pipeline = SmartVisionPipeline(cfg, db=db, bus=bus, event_repo=repo)
    notifier = DiscordNotifier.from_config(cfg, bus)  # None nếu chưa cấu hình webhook

    app = QApplication(sys.argv)
    window = MainWindow(cfg, pipeline, bus, db, initial_sources=args.sources)
    window.show()
    pipeline.start()
    code = app.exec()
    if notifier:
        notifier.stop()
    return code
    # nạp mô hình ở luồng ngầm, giao diện hiện ngay không bị đơ
    return code


if __name__ == "__main__":
    sys.exit(main())