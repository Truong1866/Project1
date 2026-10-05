#!/usr/bin/env python3
"""Test nhanh: nạp 2 model rồi quét 1 ảnh (mặc định: input.jpg).

  * YOLOv8n  : chạy qua BusinessLayer/Inference/yolo_native.py (OpenVINO native) trên iGPU Intel Iris Xe.
  * InsightFace (buffalo_s): chạy qua onnxruntime-openvino (OpenVINOExecutionProvider, GPU_FP16).

Chạy (từ thư mục gốc project, hoặc đặt file này trong Tools/):
    python test_models.py
    python test_models.py --image input.jpg --out result_test.jpg

Yêu cầu: pip install openvino opencv-python numpy insightface onnxruntime-openvino
(LƯU Ý: chỉ cài onnxruntime-openvino, KHÔNG cài song song onnxruntime thường.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---- xác định thư mục gốc project để import được BusinessLayer.*
ROOT = Path(__file__).resolve().parent
if not (ROOT / "BusinessLayer").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import openvino as ov  # noqa: E402,F401  (nạp OpenVINO TRƯỚC cv2 / onnxruntime để tránh xung đột DLL)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from BusinessLayer.Inference.yolo_native import YOLOv8OpenVINO, box_label  # noqa: E402

MODELS_DIR = ROOT / "Models"
YOLO_PATH = MODELS_DIR / "yolov8n_openvino_model"   # thư mục chứa .xml/.bin
FACE_ROOT = MODELS_DIR / "face_models"              # chứa models/buffalo_s/*.onnx

YOLO_DEVICE = "GPU"            # Intel Iris Xe iGPU
FACE_DEVICE_TYPE = 'GPU_FP16'  # device_type của OpenVINOExecutionProvider
PERSON_CONF = 0.30
PERSON_IOU = 0.45
FACE_DET_SIZE = (320, 320)
FACE_DET_THRESH = 0.5


# ---------------------------------------------------------------------------------------
# Nạp model
# ---------------------------------------------------------------------------------------
def load_yolo() -> YOLOv8OpenVINO:
    print("=== [1/2] Nạp YOLOv8n (OpenVINO native) ===")
    core = ov.Core()
    print(f"  Thiết bị OpenVINO thấy được: {core.available_devices}")
    t0 = time.perf_counter()
    yolo = YOLOv8OpenVINO(
        YOLO_PATH,
        device=YOLO_DEVICE,
        imgsz=640,
        conf=PERSON_CONF,
        iou=PERSON_IOU,
        classes=[0],                              # chỉ lấy "person"
        cache_dir=str(YOLO_PATH.parent / ".ov_cache"),
        performance_hint="LATENCY",
    )
    print(f"  => OK trên {yolo.device} | input={tuple(yolo.input_layer.shape)} | {time.perf_counter() - t0:.2f}s")
    return yolo


def load_face():
    print("\n=== [2/2] Nạp InsightFace buffalo_s (onnxruntime-openvino) ===")
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    avail = ort.get_available_providers()
    print(f"  onnxruntime providers khả dụng: {avail}")
    if "OpenVINOExecutionProvider" not in avail:
        raise RuntimeError(
            "Không có OpenVINOExecutionProvider. Hãy chạy:\n"
            "    pip uninstall -y onnxruntime onnxruntime-gpu\n"
            "    pip install onnxruntime-openvino"
        )


    t0 = time.perf_counter()
    app = FaceAnalysis(
        name="buffalo_s",
        root=str(FACE_ROOT),
        providers=['OpenVINOExecutionProvider'],          # KHÔNG thêm CPU -> lỗi thì báo lỗi, không lặng lẽ lùi về CPU
        provider_options=[{'device_type': 'GPU_FP16'}],
        allowed_modules=["detection", "recognition"],     # bỏ landmark / giới tính / tuổi cho nhẹ
    )
    app.prepare(ctx_id=0, det_size=FACE_DET_SIZE, det_thresh=FACE_DET_THRESH)
    print(f"  => Khởi tạo xong | {time.perf_counter() - t0:.2f}s")

    # onnxruntime có thể âm thầm lùi về CPU -> in provider THỰC SỰ của từng sub-model
    on_gpu = True
    for task, model in app.models.items():
        sess = getattr(model, "session", None)
        if sess is None:
            continue
        real = sess.get_providers()
        print(f"  - [{task}] đang chạy bằng: {real}")
        if "OpenVINOExecutionProvider" not in real:
            on_gpu = False
    if not on_gpu:
        print("  !!! CẢNH BÁO: có model đang chạy CPU (OpenVINO EP không được nạp).")
    return app


# ---------------------------------------------------------------------------------------
# Quét ảnh
# ---------------------------------------------------------------------------------------
def run_scan(yolo: YOLOv8OpenVINO, face_app, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    print(f"\n=== QUÉT ẢNH ({w}x{h}) ===")

    # Warm-up: lần chạy đầu biên dịch kernel GPU nên rất chậm, không tính vào kết quả
    yolo(img)
    face_app.get(img)

    # ---- YOLO
    det = yolo(img)
    s = det.speed
    print(f"[YOLO] {len(det)} người | pre {s['preprocess']:.1f} ms, "
          f"infer {s['Inference']:.1f} ms, post {s['postprocess']:.1f} ms")
    for i, (b, c) in enumerate(zip(det.boxes, det.conf), 1):
        print(f"   #{i}: conf={c:.2f} box={[int(v) for v in b]}")

    # ---- InsightFace
    t0 = time.perf_counter()
    faces = face_app.get(img)
    face_ms = (time.perf_counter() - t0) * 1000
    print(f"[FACE] {len(faces)} khuôn mặt | {face_ms:.1f} ms")
    for i, f in enumerate(faces, 1):
        emb = f.normed_embedding
        print(f"   #{i}: det_score={f.det_score:.2f} bbox={[int(v) for v in f.bbox]} "
              f"embedding={emb.shape} |v|={np.linalg.norm(emb):.3f}")

    # ---- Vẽ kết quả
    out = img.copy()
    for b, c in zip(det.boxes, det.conf):
        box_label(out, b, f"person {c:.2f}", (0, 200, 0))
    for f in faces:
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"face {f.det_score:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Test YOLO (OpenVINO) + InsightFace (ORT-OpenVINO) trên 1 ảnh")
    ap.add_argument("--image", default="input.jpg", help="đường dẫn ảnh đầu vào (mặc định: input.jpg)")
    ap.add_argument("--out", default="result_test.jpg", help="ảnh kết quả đã vẽ box")
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ kết quả")
    a = ap.parse_args()

    img_path = Path(a.image)
    if not img_path.exists() and (ROOT / a.image).exists():
        img_path = ROOT / a.image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Lỗi: không đọc được ảnh '{a.image}' (đã tìm: {img_path.resolve()})")
        return 1

    yolo = load_yolo()
    face_app = load_face()

    result = run_scan(yolo, face_app, img)

    cv2.imwrite(a.out, result)
    print(f"\nĐã lưu ảnh kết quả: {Path(a.out).resolve()}")
    if a.show:
        cv2.imshow("Test models", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())