"""Đóng gói toàn bộ quy trình AI thành các hàm đơn giản cho pipeline:

    detect_persons(frame)            -> [{'box': [x1,y1,x2,y2], 'confidence': float}]
    process_faces(frame, person_box) -> [{'embedding', 'abs_box', 'det_score'}]
    largest_face(frame)              -> dùng khi đăng ký người quen

YOLO chạy bằng OpenVINO native (yolo_native.py); khuôn mặt chạy InsightFace + OpenVINO EP (GPU_FP16).
Một khoá chung bảo vệ các lệnh suy luận (pipeline + giao diện đăng ký có thể gọi đồng thời).
"""
from __future__ import annotations

import threading
from pathlib import Path

try:  # nạp OpenVINO TRƯỚC cv2/onnxruntime để tránh xung đột DLL trên một số máy
    import openvino as ov  # noqa: F401
except ImportError:  # pragma: no cover
    ov = None

import numpy as np

from BusinessLayer.Inference.face_recognizer import DEFAULT_DEVICE_TYPE, FaceRecognizer
from BusinessLayer.Inference.yolo_native import YOLOv8OpenVINO
from Utils.logger import get_logger

log = get_logger("Engine")


class InferenceEngine:
    def __init__(self, yolo_model_path, face_model_dir, device: str = "GPU", imgsz: int = 640,
                 person_conf: float = 0.30, person_iou: float = 0.45,
                 face_det_size=(320, 320), face_det_thresh: float = 0.5,
                 head_ratio: float = 0.6, min_face_px: int = 36,
                 face_device_type: str = DEFAULT_DEVICE_TYPE):
        log.info("Đang khởi tạo Inference Engine...")
        self._lock = threading.Lock()
        yolo_path = Path(yolo_model_path)
        if yolo_path.suffix == ".pt":
            raise RuntimeError(
                f"'{yolo_path}' là file PyTorch. Hãy chạy Tools/setup_models.py để export sang OpenVINO "
                "rồi trỏ models.yolo_path tới thư mục *_openvino_model.")

        self.device = device
        try:
            self.yolo = self._make_yolo(yolo_path, device, imgsz, person_conf, person_iou)
        except Exception as e:
            if device.upper() == "CPU":
                raise
            log.warning("YOLO không chạy được trên %s (%s) -> chuyển sang CPU", device, e)
            self.device = "CPU"
            self.yolo = self._make_yolo(yolo_path, "CPU", imgsz, person_conf, person_iou)
        log.info("YOLO sẵn sàng trên %s", self.device)

        self.faces = FaceRecognizer(face_model_dir, det_size=face_det_size, det_thresh=face_det_thresh,
                                    head_ratio=head_ratio, min_face_px=min_face_px,
                                    device_type=face_device_type)
        self.face_device = self.faces.device_label
        log.info("Khuôn mặt sẵn sàng trên %s", self.face_device)

        self._warmup()
        log.info("=> Inference Engine sẵn sàng!")

    @staticmethod
    def _make_yolo(path: Path, device, imgsz, conf, iou):
        return YOLOv8OpenVINO(path, device=device, imgsz=imgsz, conf=conf, iou=iou, classes=[0],
                              cache_dir=str(path.parent / ".ov_cache"), performance_hint="LATENCY")

    def _warmup(self) -> None:
        """Chạy thử 1 khung đen: biên dịch kernel GPU của YOLO trước (InsightFace tự warm-up trong FaceRecognizer)."""
        try:
            self.yolo(np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception:
            log.exception("Warm-up YOLO lỗi")

    # ------------------------------------------------------------------ API
    def set_person_conf(self, conf: float) -> None:
        self.yolo.conf = float(conf)

    def detect_persons(self, frame: np.ndarray) -> list[dict]:
        with self._lock:
            det = self.yolo(frame)
        persons = []
        for box, conf in zip(det.boxes, det.conf):
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            persons.append({"box": [x1, y1, x2, y2], "confidence": float(conf)})
        return persons

    def process_faces(self, frame: np.ndarray, person_box) -> list[dict]:
        with self._lock:
            return self.faces.extract(frame, person_box, head_only=True)

    def largest_face(self, frame: np.ndarray) -> dict | None:
        """Mặt lớn nhất trong khung: thử qua box người trước, không có thì quét cả khung."""
        candidates: list[dict] = []
        for p in sorted(self.detect_persons(frame),
                        key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])):
            candidates = self.process_faces(frame, p["box"])
            if candidates:
                break
        if not candidates:
            with self._lock:
                candidates = self.faces.extract(frame, None)
        if not candidates:
            return None
        return max(candidates, key=lambda f: (f["abs_box"][2] - f["abs_box"][0]) * (f["abs_box"][3] - f["abs_box"][1]))