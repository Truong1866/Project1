"""Phát hiện mặt + trích xuất vector 512 chiều bằng InsightFace (buffalo_s).

Chỉ nạp 2 module cần thiết (detection + recognition), bỏ landmark 3D/2D và giới tính/tuổi
-> nhanh hơn nhiều so với FaceAnalysis mặc định.
"""
from __future__ import annotations

import numpy as np

from Utils.logger import get_logger

log = get_logger("Face")


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FaceRecognizer:
    def __init__(self, face_model_dir, det_size=(320, 320), det_thresh=0.5,
                 head_ratio=0.6, min_face_px=36, providers=None):
        from insightface.app import FaceAnalysis  # import trễ: nạp nặng

        if providers is None:
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
            except Exception:
                available = ["CPUExecutionProvider"]
            preferred = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        log.info("InsightFace providers: %s", providers)

        self.head_ratio = head_ratio
        self.min_face_px = min_face_px
        self.app = FaceAnalysis(name="buffalo_s", root=str(face_model_dir), providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0, det_size=tuple(det_size), det_thresh=det_thresh)

    def extract(self, frame: np.ndarray, person_box=None, head_only: bool = True) -> list[dict]:
        """Trả về danh sách {'embedding', 'abs_box', 'det_score'} với toạ độ tuyệt đối trên ảnh gốc."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h
        if person_box is not None:
            x1, y1, x2, y2 = person_box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if head_only:  # mặt luôn nằm ở phần trên cơ thể -> crop nhỏ hơn, nhanh hơn, ít nhầm hơn
                y2 = y1 + int((y2 - y1) * self.head_ratio)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            return []

        out = []
        for face in self.app.get(crop):
            fw = face.bbox[2] - face.bbox[0]
            fh = face.bbox[3] - face.bbox[1]
            if min(fw, fh) < self.min_face_px:
                continue
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)
            out.append({
                "embedding": emb,
                "det_score": float(face.det_score),
                "abs_box": [int(face.bbox[0] + x1), int(face.bbox[1] + y1),
                            int(face.bbox[2] + x1), int(face.bbox[3] + y1)],
            })
        return out
