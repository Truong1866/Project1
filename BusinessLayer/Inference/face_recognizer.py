"""Phát hiện mặt + trích xuất vector 512 chiều bằng InsightFace (buffalo_s).

Chỉ nạp 2 module cần thiết (detection + recognition) -> nhanh hơn FaceAnalysis mặc định.
Mặc định chạy OpenVINO Execution Provider trên iGPU (GPU_FP16); lỗi thì rơi về CPU.
"""
from __future__ import annotations

import importlib
import os

import numpy as np

from Utils.logger import get_logger

log = get_logger("Face")

MODULE_VERSION = "face_recognizer v3 (OpenVINO EP, GPU_FP16)"
DEFAULT_DEVICE_TYPE = "GPU_FP16"
OV_EP = "OpenVINOExecutionProvider"
CPU_EP = "CPUExecutionProvider"


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _prepare_onnxruntime():
    """Nạp onnxruntime, thêm DLL OpenVINO vào PATH (Windows) và in thông tin chẩn đoán."""
    import onnxruntime as ort

    if os.name == "nt":
        try:  # có trong gói onnxruntime-openvino; cần khi OpenVINO cài bằng pip
            importlib.import_module("onnxruntime.tools.add_openvino_win_libs").add_openvino_libs_to_path()
        except Exception as e:  # noqa: BLE001
            log.debug("Bỏ qua add_openvino_libs_to_path: %s", e)

    from importlib import metadata
    found = []
    for dist in ("onnxruntime", "onnxruntime-openvino", "onnxruntime-gpu", "onnxruntime-directml", "openvino"):
        try:
            found.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pass
    log.info("ORT %s | gói đã cài: %s | providers khả dụng: %s",
             ort.__version__, ", ".join(found) or "?", ort.get_available_providers())
    if any(f.startswith("onnxruntime==") for f in found) and any(f.startswith("onnxruntime-openvino") for f in found):
        log.warning("Đang cài CẢ 'onnxruntime' và 'onnxruntime-openvino' -> chúng đè lên nhau. "
                    "Chạy: pip uninstall -y onnxruntime onnxruntime-openvino && pip install onnxruntime-openvino")
    return ort


class FaceRecognizer:
    def __init__(self, face_model_dir, det_size=(320, 320), det_thresh=0.5,
                 head_ratio=0.6, min_face_px=36, providers=None, provider_options=None,
                 device_type: str = DEFAULT_DEVICE_TYPE):
        log.info("Nạp %s từ %s", MODULE_VERSION, __file__)
        self.head_ratio = head_ratio
        self.min_face_px = min_face_px
        self.face_model_dir = str(face_model_dir)
        self.det_size = tuple(det_size)
        self.det_thresh = det_thresh
        self.device_type = device_type

        _prepare_onnxruntime()

        # Các cấu hình thử lần lượt: (providers, provider_options)
        if providers is not None:
            attempts = [(list(providers), provider_options)]
        else:
            attempts = [([OV_EP], [{"device_type": device_type}])]
        attempts.append(([CPU_EP], None))

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

        self.on_openvino = self._check_real_providers()
        self.device_label = self.device_type if self.on_openvino else "CPU"
        self.warmup()

    # ------------------------------------------------------------------ nội bộ
    def _build_app(self, providers: list, provider_options):
        from insightface.app import FaceAnalysis  # import trễ: nạp nặng

        kwargs = dict(name="buffalo_s", root=self.face_model_dir, providers=providers,
                      allowed_modules=["detection", "recognition"])
        if provider_options is not None:
            kwargs["provider_options"] = provider_options
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        return app

    def _check_real_providers(self) -> bool:
        """ORT có thể âm thầm lùi về CPU -> in provider THỰC SỰ của từng model."""
        wanted = OV_EP in self.providers_used
        ok = True
        for task, model in self.app.models.items():
            sess = getattr(model, "session", None)
            if sess is None:
                continue
            real = sess.get_providers()
            log.info("InsightFace [%s] đang chạy bằng: %s", task, real)
            if wanted and OV_EP not in real:
                ok = False
        if wanted and not ok:
            log.warning("OpenVINO EP KHÔNG được nạp cho InsightFace -> đang chạy CPU. "
                        "Xem dòng 'gói đã cài' phía trên để biết thiếu/thừa gói nào.")
        return wanted and ok

    def warmup(self) -> None:
        """Chạy thử 1 lần để GPU biên dịch kernel, khung thật đầu tiên không bị khựng."""
        try:
            noise = np.random.randint(0, 255, (self.det_size[1], self.det_size[0], 3), dtype=np.uint8)
            self.app.get(noise)
            rec = self.app.models.get("recognition")
            if rec is not None:
                rec.get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
        except Exception:
            log.exception("Warm-up InsightFace lỗi")

    # ------------------------------------------------------------------ API
    def extract(self, frame: np.ndarray, person_box=None, head_only: bool = True) -> list[dict]:
        """Trả về danh sách {'embedding', 'abs_box', 'det_score'} với toạ độ tuyệt đối trên ảnh gốc."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h
        if person_box is not None:
            x1, y1, x2, y2 = person_box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if head_only:
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