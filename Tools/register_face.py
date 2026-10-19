"""Đăng ký khuôn mặt vào cơ sở dữ liệu.

Chạy YOLO và Face recognition hoàn toàn bằng OpenVINO (không cần PyTorch/CUDA).
- YOLO  : YOLOOpenVINO từ benchmark/yolo_native.py (standalone, không cần ultralytics)
- Faces : FaceRecognizer (InsightFace + OpenVINO EP)

Quy trình đăng ký nhiều góc nhìn:
  - Nhấn 's' để chụp từng góc (mặt thẳng, trái, phải, ngẩng, cúi…)
  - Mỗi lần chụp thành công, embedding được thu thập vào danh sách tạm
  - Khi đủ số lượng MẪU_TỐI_THIỂU, hệ thống tự tính vector trung bình → chuẩn hóa L2 → lưu DB
  - Nhấn 'q' để lưu sớm (nếu đã có ít nhất 1 mẫu) hoặc thoát
"""

import sys
from pathlib import Path

# ── Thêm thư mục benchmark vào sys.path để import yolo_native trực tiếp ───────
_BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import cv2
import numpy as np

# Import YOLOOpenVINO từ benchmark/yolo_native.py (không qua ultralytics/torch)
try:
    import openvino as ov  # noqa: F401  # nạp OpenVINO TRƯỚC cv2 / onnxruntime
except ImportError:
    pass

from benchmark.yolo_native import YOLOOpenVINO  # benchmark/yolo_native.py

from BusinessLayer.Inference.face_recognizer import FaceRecognizer
from DataLayer.vector_db import FaceDatabase

# ── Cấu hình model ─────────────────────────────────────────────────────────────
CURRENT_FILE  = Path(__file__).resolve()
MODELS_DIR    = CURRENT_FILE.parent.parent / "Models"

# Đường dẫn model YOLO OpenVINO (thư mục xuất từ `yolo export format=openvino`)
YOLO_MODEL    = MODELS_DIR / "yolov26n_640_openvino_model"  # thư mục chứa *.xml

# Thư mục chứa model InsightFace (buffalo_s)
FACE_MODEL_DIR = MODELS_DIR / "face_models"

# Thiết bị chạy YOLO: "GPU" cho Intel iGPU, "CPU" nếu không có GPU
YOLO_DEVICE       = "GPU"
YOLO_IMGSZ        = 640
YOLO_CONF         = 0.40
YOLO_IOU          = 0.45
YOLO_CLASSES      = [0]   # chỉ detect Person
PERFORMANCE_HINT  = "LATENCY"

# Thiết bị chạy InsightFace: "GPU_FP16" (OpenVINO EP) hoặc "CPU"
FACE_DEVICE_TYPE  = "GPU_FP16"
FACE_DET_SIZE     = (320, 320)
FACE_DET_THRESH   = 0.5
HEAD_RATIO        = 0.6   # tỷ lệ phần đầu trên bounding box người

# ── Cấu hình thu thập nhiều góc nhìn ──────────────────────────────────────────
# Số mẫu cần chụp trước khi tự động lưu (mặc định 5 góc)
NUM_SAMPLES   = 5
# Ngưỡng cosine similarity tối thiểu giữa các mẫu liên tiếp (lọc ảnh trùng góc)
MIN_COSINE_DIFF = 0.02   # loại bỏ mẫu quá giống mẫu trước (người không xoay đầu)

# Gợi ý góc nhìn theo thứ tự để hướng dẫn người dùng
ANGLE_HINTS = [
    "thẳng",
    "xoay trái nhẹ",
    "xoay phải nhẹ",
    "ngẩng đầu nhẹ",
    "cúi đầu nhẹ",
]


# ── Khởi tạo model ─────────────────────────────────────────────────────────────
def _build_yolo() -> YOLOOpenVINO:
    """Khởi tạo YOLO OpenVINO; tự động rơi về CPU nếu GPU không dùng được."""
    try:
        yolo = YOLOOpenVINO(
            YOLO_MODEL,
            device=YOLO_DEVICE,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=YOLO_CLASSES,
            performance_hint=PERFORMANCE_HINT,
            cache_dir=str(YOLO_MODEL.parent / ".ov_cache"),
        )
        print(f"[YOLO] Thiết bị: {YOLO_DEVICE} | Input: {yolo.input_layer.shape}")
        return yolo
    except Exception as e:
        if YOLO_DEVICE.upper() == "CPU":
            raise
        print(f"[YOLO] Khởi tạo trên {YOLO_DEVICE} thất bại ({e}) → chuyển về CPU.")
        yolo = YOLOOpenVINO(
            YOLO_MODEL,
            device="CPU",
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=YOLO_CLASSES,
            performance_hint=PERFORMANCE_HINT,
            cache_dir=str(YOLO_MODEL.parent / ".ov_cache"),
        )
        print(f"[YOLO] Thiết bị: CPU | Input: {yolo.input_layer.shape}")
        return yolo


def _detect_persons(yolo: YOLOOpenVINO, frame) -> list[dict]:
    """Chạy YOLO và trả về list {'box': [x1,y1,x2,y2], 'confidence': float}."""
    det = yolo(frame)
    persons = []
    for box, conf in zip(det.boxes, det.conf):
        x1, y1, x2, y2 = (int(v) for v in box)
        if (x2 - x1) < 8 or (y2 - y1) < 16:
            continue
        persons.append({"box": [x1, y1, x2, y2], "confidence": float(conf)})
    return persons


def _extract_face_embedding(faces: FaceRecognizer, frame, person_box: list):
    """Trích xuất embedding khuôn mặt từ vùng bounding box người."""
    return faces.extract(frame, person_box, head_only=True)
    # list[{'embedding', 'abs_box', 'det_score'}]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa hai vector đã chuẩn hóa (hoặc chưa)."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Tính trung bình rồi chuẩn hóa L2 → unit vector."""
    mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(mean_vec)
    return (mean_vec / (norm + 1e-9)).astype(np.float32)


def _draw_overlay(frame: np.ndarray, collected: int, total: int, hint: str) -> np.ndarray:
    """Vẽ thông tin hướng dẫn lên frame hiển thị (không ảnh hưởng frame gốc)."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Thanh tiến trình
    bar_w = int(w * 0.6)
    bar_x = (w - bar_w) // 2
    bar_y = h - 50
    bar_h = 18
    fill = int(bar_w * collected / total)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 80), -1)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(vis, f"{collected}/{total}", (bar_x + bar_w + 8, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Gợi ý góc
    label = f"Goc: {hint}  |  Nhan 's' de chup  |  'q' luu & thoat"
    cv2.putText(vis, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # Flash xanh khi vừa chụp thành công (collected > 0 và fill > 0)
    if collected > 0 and fill == bar_w:
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    return vis


# ── Main ────────────────────────────────────────────────────────────────────────
def register():
    name = input("Nhập tên người cần thêm (VD: Truong): ").strip()
    if not name:
        print("Tên không được để trống.")
        return

    print("Đang khởi động AI (OpenVINO)...")

    # Kiểm tra thư mục model
    if not YOLO_MODEL.exists():
        print(
            f"Lỗi: Không tìm thấy model YOLO tại '{YOLO_MODEL}'.\n"
            "Hãy xuất model trước bằng lệnh:\n"
            "  yolo export model=yolov8n.pt format=openvino imgsz=640\n"
            "rồi đặt thư mục *_openvino_model vào Models/"
        )
        return

    # Khởi tạo YOLO (OpenVINO)
    yolo = _build_yolo()

    # Khởi tạo Face Recognizer (InsightFace + OpenVINO EP)
    faces = FaceRecognizer(
        FACE_MODEL_DIR,
        det_size=FACE_DET_SIZE,
        det_thresh=FACE_DET_THRESH,
        head_ratio=HEAD_RATIO,
        device_type=FACE_DEVICE_TYPE,
    )
    print(f"[Face] Thiết bị: {faces.device_label}")

    # Warm-up YOLO
    yolo(np.zeros((480, 640, 3), dtype=np.uint8))
    print("AI sẵn sàng!")

    db = FaceDatabase()
    cam = cv2.VideoCapture(0)

    # ── Danh sách embedding thu thập từ nhiều góc ──────────────────────────────
    embeddings: list[np.ndarray] = []
    last_emb: np.ndarray | None = None        # để kiểm tra độ khác biệt

    print(
        f"\n=== Chụp {NUM_SAMPLES} góc khác nhau để đăng ký ===\n"
        "  's' → chụp mẫu hiện tại\n"
        "  'q' → lưu & thoát (cần ít nhất 1 mẫu)\n"
        "  'r' → chụp lại mẫu vừa chụp (nếu nhận ra mặt trùng góc)\n"
    )

    flash_frames = 0  # số frame hiển thị flash xanh khi chụp thành công

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        collected = len(embeddings)
        hint_idx  = min(collected, len(ANGLE_HINTS) - 1)
        hint      = ANGLE_HINTS[hint_idx] if collected < NUM_SAMPLES else "Đủ mẫu!"

        # Vẽ overlay hướng dẫn
        vis = _draw_overlay(frame, collected, NUM_SAMPLES, hint)

        # Flash xanh nhất thời sau khi chụp thành công
        if flash_frames > 0:
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            flash_frames -= 1

        cv2.imshow("Register Face (OpenVINO)", vis)
        key = cv2.waitKey(1) & 0xFF

        # ── Phím 's': chụp mẫu ──────────────────────────────────────────────
        if key == ord('s'):
            # 1. Phát hiện người
            persons = _detect_persons(yolo, frame)
            if not persons:
                print(f"  [!] Không thấy người. Thử lại (góc: {hint})")
                continue

            main_person = persons[0]

            # 2. Trích xuất embedding
            face_results = _extract_face_embedding(faces, frame, main_person["box"])
            if not face_results:
                print(f"  [!] Không thấy rõ khuôn mặt. Thử lại (góc: {hint})")
                continue

            emb = np.asarray(face_results[0]["embedding"], dtype=np.float32)

            # 3. Kiểm tra không quá giống góc trước (tránh chụp trùng)
            if last_emb is not None:
                sim = _cosine_sim(emb, last_emb)
                if sim > (1.0 - MIN_COSINE_DIFF):
                    print(
                        f"  [~] Góc quá giống lần trước (sim={sim:.4f}). "
                        "Hãy xoay đầu thêm rồi nhấn 's'."
                    )
                    continue

            # 4. Thu thập mẫu
            embeddings.append(emb)
            last_emb = emb
            flash_frames = 8  # hiển thị flash ~8 frame ≈ 0.25s
            print(
                f"  [✓] Mẫu {len(embeddings)}/{NUM_SAMPLES} thu thập xong "
                f"(det_score={face_results[0]['det_score']:.2f}, góc: {hint})"
            )

            # 5. Tự động lưu khi đủ NUM_SAMPLES
            if len(embeddings) >= NUM_SAMPLES:
                avg_vec = _average_embedding(embeddings)
                db.add_person(name, avg_vec)
                print(
                    f"\n==> Đăng ký thành công '{name}' với vector trung bình "
                    f"từ {len(embeddings)} góc nhìn! Nhấn 'q' để thoát."
                )
                embeddings.clear()  # reset nếu muốn chụp thêm người khác

        # ── Phím 'r': chụp lại mẫu cuối (xóa mẫu vừa chụp) ────────────────
        elif key == ord('r'):
            if embeddings:
                embeddings.pop()
                last_emb = embeddings[-1] if embeddings else None
                print(f"  [r] Đã xóa mẫu cuối. Còn {len(embeddings)}/{NUM_SAMPLES} mẫu.")
            else:
                print("  [r] Chưa có mẫu nào để xóa.")

        # ── Phím 'q': lưu sớm & thoát ───────────────────────────────────────
        elif key == ord('q'):
            if embeddings:
                avg_vec = _average_embedding(embeddings)
                db.add_person(name, avg_vec)
                print(
                    f"\n==> Đã lưu '{name}' với vector trung bình "
                    f"từ {len(embeddings)} mẫu (lưu sớm)."
                )
            else:
                print("\n[!] Chưa có mẫu nào, thoát mà không lưu.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register()
