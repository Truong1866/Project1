"""Phát hiện mặt + trích xuất vector 512 chiều bằng InsightFace (buffalo_s).

Chỉ nạp 2 module cần thiết (detection + recognition), bỏ landmark 3D/2D và giới tính/tuổi
-> nhanh hơn nhiều so với FaceAnalysis mặc định.

Mặc định chạy bằng OpenVINO Execution Provider trên iGPU (GPU_FP16); lỗi thì tự rơi về CPU.
"""
from __future__ import annotations

import numpy as np

from Utils.logger import get_logger

log = get_logger("Face")

DEFAULT_DEVICE_TYPE = "GPU_FP16"


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FaceRecognizer:
    def __init__(self, face_model_dir, det_size=(320, 320), det_thresh=0.5,
                 head_ratio=0.6, min_face_px=36, providers=None, provider_options=None,
                 device_type: str = DEFAULT_DEVICE_TYPE):
        self.head_ratio = head_ratio
        self.min_face_px = min_face_px
        self.face_model_dir = str(face_model_dir)
        self.det_size = tuple(det_size)
        self.det_thresh = det_thresh

        # Danh sách cấu hình sẽ thử lần lượt: (providers, provider_options)
        if providers is not None:  # người dùng tự chỉ định -> tôn trọng, chỉ thêm CPU làm đường lui
            attempts = [(list(providers), provider_options)]
        else:
            attempts = [(["OpenVINOExecutionProvider"], [{"device_type": device_type}])]
        attempts.append((["CPUExecutionProvider"], None))

        self.app = None
        self.providers_used: list[str] = []
        last_err: Exception | None = None
        for prov, opts in attempts:
            try:
                log.info("Khởi tạo InsightFace: providers=%s options=%s", prov, opts)
                self.app = self._build_app(prov, opts)
                self.providers_used = prov
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("InsightFace không chạy được với %s (%s: %s)", prov, type(e).__name__, e)

        if self.app is None:
            raise RuntimeError(f"Không khởi tạo được InsightFace: {last_err}")

        self._log_real_providers()

    # ------------------------------------------------------------------ nội bộ
    def _build_app(self, providers: list, provider_options):
        from insightface.app import FaceAnalysis  # import trễ: nạp nặng

        kwargs = dict(
            name="buffalo_s",
            root=self.face_model_dir,            # chứa models/buffalo_s/*.onnx
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        if provider_options is not None:
            kwargs["provider_options"] = provider_options  # FaceAnalysis chuyển tiếp xuống onnxruntime
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        return app

    def _log_real_providers(self) -> None:
        """ORT có thể âm thầm lùi về CPU -> in ra provider THỰC SỰ của từng model để kiểm tra."""
        for task, model in self.app.models.items():
            sess = getattr(model, "session", None)
            if sess is None:
                continue
            real = sess.get_providers()
            log.info("InsightFace [%s] đang chạy bằng: %s", task, real)
            if "OpenVINOExecutionProvider" not in real and "OpenVINOExecutionProvider" in self.providers_used:
                log.warning("[%s] OpenVINO EP KHÔNG được nạp, đang chạy CPU. "
                            "Kiểm tra: pip uninstall onnxruntime && pip install onnxruntime-openvino", task)

    # ------------------------------------------------------------------ API
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