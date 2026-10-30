from pathlib import Path
import shutil
from dataclasses import dataclass, field
from ultralytics import YOLO
from insightface.app import FaceAnalysis

current_file_path = Path(__file__).resolve()
models_dir = current_file_path.parent.parent / 'Models'
models_dir.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# DANH SÁCH MODEL CẦN TẢI VÀ XUẤT OPENVINO
# ---------------------------------------------------------------------------
# Thêm / bớt / chỉnh sửa các entry dưới đây để kiểm soát quá trình setup.
#
# Mỗi entry là một ModelConfig với:
#   target_filename : tên file .pt sẽ được lưu trong thư mục Models/
#   candidates      : danh sách tên thử khi tải tự động (thử lần lượt từ trên xuống)
#   export_sizes    : list[int] – các giá trị imgsz sẽ xuất OpenVINO (ví dụ [640, 320])
#   export_prefix   : tiền tố tên thư mục OpenVINO (ví dụ "yolov8n" → yolov8n_640_openvino_model)
#                     Nếu để None, sẽ tự suy từ target_filename (bỏ đuôi .pt)
#   quantize        : chế độ lượng tử hóa khi xuất ('FP16' | 'INT8' | None)
#   run_smoke_test  : True → chạy thử inference sau khi tải để kiểm tra model
# ===========================================================================

@dataclass
class ModelConfig:
    target_filename: str                    # Tên file .pt lưu trong Models/
    candidates: list[str]                   # Tên thay thế khi tải tự động
    export_sizes: list[int] = field(        # Các imgsz sẽ xuất OpenVINO
        default_factory=lambda: [640, 320]
    )
    export_prefix: str | None = None        # Tiền tố thư mục OpenVINO (None = tự suy)
    quantize: str | None = 'FP16'           # Chế độ lượng tử hóa ('FP16' | 'INT8' | None)
    run_smoke_test: bool = True             # Chạy thử inference sau khi tải


# ─── CHỈNH SỬA DANH SÁCH NÀY ĐỂ THÊM / BỚT MODEL ───────────────────────────
YOLO_MODELS: list[ModelConfig] = [
    ModelConfig(
        target_filename='yolov8n.pt',
        candidates=['yolov8n.pt'],
        export_sizes=[640, 320],
        export_prefix='yolov8n',
        quantize='FP16',
    ),
    ModelConfig(
        target_filename='yolov26n.pt',
        candidates=['yolo26n.pt', 'yolov26n.pt'],
        export_sizes=[640, 320],
        export_prefix='yolov26n',
        quantize='FP16',
    ),
    # ── Thêm model mới vào đây ──────────────────────────────────────────────
    # ModelConfig(
    #     target_filename='yolov9c.pt',
    #     candidates=['yolov9c.pt'],
    #     export_sizes=[640],
    #     export_prefix='yolov9c',
    #     quantize='FP16',
    # ),
    # ModelConfig(
    #     target_filename='yolo11n.pt',
    #     candidates=['yolo11n.pt'],
    #     export_sizes=[640, 320],
    #     export_prefix='yolo11n',
    #     quantize='INT8',
    # ),
]
# ─────────────────────────────────────────────────────────────────────────────


# ===========================================================================
# Các hàm tiện ích
# ===========================================================================

def get_model_path(target_filename: str, candidate_names: list[str]) -> Path:
    """
    Đảm bảo mô hình PyTorch (.pt) tồn tại trong thư mục Models.
    Nếu chưa có, tự động tải về qua Ultralytics và lưu vào thư mục Models.
    """
    target_path = models_dir / target_filename
    if target_path.exists():
        print(f"=> Đã tìm thấy {target_filename} tại: {target_path}")
        return target_path

    # Kiểm tra các tên ứng viên khác đã có sẵn trong Models chưa
    for name in candidate_names:
        candidate_path = models_dir / name
        if candidate_path.exists():
            print(f"=> Đã tìm thấy {name} tại: {candidate_path}")
            return candidate_path

    # Nếu chưa có, tiến hành tải tự động
    print(f"-> Đang tải mô hình {target_filename} qua Ultralytics...")
    for name in candidate_names:
        try:
            print(f"   Thử tải model với tên '{name}'...")
            model = YOLO(name)
            ckpt = Path(model.ckpt_path) if getattr(model, 'ckpt_path', None) else Path(name)
            if ckpt.exists():
                shutil.copy2(ckpt, target_path)
            elif Path(name).exists():
                shutil.move(name, target_path)

            print(f"=> [Thành công] Đã lưu mô hình vào: {target_path}")
            return target_path
        except Exception as e:
            print(f"   Không tải được '{name}': {e}")
            continue

    if target_path.exists():
        return target_path

    raise FileNotFoundError(
        f"Không thể tìm hoặc tải mô hình {target_filename}. Các tên đã thử: {candidate_names}"
    )


def export_openvino(model_path: Path, cfg: ModelConfig) -> None:
    """
    Biên dịch mô hình sang OpenVINO theo cấu hình ModelConfig:
      - xuất từng imgsz trong cfg.export_sizes
      - dùng cfg.quantize làm chế độ lượng tử hóa
      - lưu vào Models/<prefix>_<imgsz>_openvino_model/
    """
    # Xác định tiền tố tên thư mục
    prefix = cfg.export_prefix or Path(cfg.target_filename).stem

    q_label = cfg.quantize if cfg.quantize else 'FP32'
    print(f"\n-> Đang biên dịch '{prefix}' sang OpenVINO ({q_label})...")

    for imgsz in cfg.export_sizes:
        target_dir = models_dir / f"{prefix}_{imgsz}_openvino_model"
        print(f"   Xuất imgsz={imgsz}, quantize='{q_label}'  →  {target_dir.name}")

        # Khởi tạo instance YOLO riêng biệt cho mỗi lần export
        model = YOLO(str(model_path))
        try:
            if cfg.quantize:
                export_result = model.export(
                    format='openvino',
                    imgsz=imgsz,
                    quantize=cfg.quantize,
                )
            else:
                export_result = model.export(
                    format='openvino',
                    imgsz=imgsz,
                )
        except Exception as e:
            print(f"   [Cảnh báo] Xuất với quantize='{q_label}' gặp lỗi ({e}), thử fallback half=True...")
            export_result = model.export(
                format='openvino',
                imgsz=imgsz,
                half=True,
            )

        exported_path = Path(export_result)
        exported_dir = exported_path if exported_path.is_dir() else exported_path.parent

        if target_dir.exists() and target_dir.resolve() != exported_dir.resolve():
            shutil.rmtree(target_dir)

        if exported_dir.resolve() != target_dir.resolve():
            shutil.move(str(exported_dir), str(target_dir))

        print(f"   => [Thành công] Lưu tại: {target_dir}")


def smoke_test(model_path: Path, model_name: str) -> bool:
    """Chạy thử inference nhanh để kiểm tra model có hoạt động không."""
    print(f"-> Đang chạy smoke test cho {model_name}...")
    try:
        m = YOLO(str(model_path))
        m.predict(source="https://ultralytics.com/images/bus.jpg", imgsz=640, verbose=False)
        print(f"=> [Thành công] {model_name} hoạt động bình thường!")
        return True
    except Exception as e:
        print(f"=> [Thất bại] {model_name} lỗi khi chạy thử: {e}")
        return False


# ===========================================================================
# Hàm setup chính – chạy theo danh sách YOLO_MODELS
# ===========================================================================

def setup_yolo(models: list[ModelConfig] | None = None) -> None:
    """
    Tải và xuất OpenVINO cho từng model trong danh sách.
    Nếu `models` là None, dùng danh sách mặc định YOLO_MODELS.
    """
    model_list = models if models is not None else YOLO_MODELS

    print(f"=== SETUP YOLO: {len(model_list)} model sẽ được xử lý ===\n")

    for idx, cfg in enumerate(model_list, start=1):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  [{idx}/{len(model_list)}] MODEL: {cfg.target_filename}")
        print(f"  Export sizes : {cfg.export_sizes}")
        print(f"  Quantize     : {cfg.quantize or 'FP32 (không lượng tử hóa)'}")
        print(sep)

        # Bước 1: Đảm bảo file .pt tồn tại
        try:
            model_path = get_model_path(cfg.target_filename, cfg.candidates)
        except FileNotFoundError as e:
            print(f"[LỖI] Bỏ qua model này: {e}\n")
            continue

        # Bước 2: Smoke test (tuỳ chọn)
        if cfg.run_smoke_test:
            ok = smoke_test(model_path, cfg.target_filename)
            if not ok:
                print("[Cảnh báo] Smoke test thất bại, vẫn tiếp tục xuất OpenVINO...\n")

        # Bước 3: Xuất OpenVINO
        try:
            export_openvino(model_path, cfg)
        except Exception as e:
            print(f"[LỖI] Xuất OpenVINO cho '{cfg.target_filename}' thất bại: {e}\n")

    print("\n=== HOÀN THÀNH SETUP YOLO ===")


def setup_face_models():
    print("\n=== TẢI MÔ HÌNH NHẬN DIỆN KHUÔN MẶT ===")
    face_dir = models_dir / 'face_models'
    face_dir.mkdir(parents=True, exist_ok=True)

    try:
        # buffalo_s là bộ model nhẹ (Small) rất phù hợp cho CPU/iGPU
        # Bao gồm cả Detection (tìm mặt) và Recognition (trích xuất vector)
        app = FaceAnalysis(name='buffalo_s', root=face_dir)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("=> [Thành công] Mô hình khuôn mặt (ONNX) đã tải xong và sẵn sàng dùng!")
    except Exception as e:
        print(f"=> [Thất bại] Lỗi khi tải mô hình khuôn mặt: {e}")


if __name__ == '__main__':
    setup_yolo()
    # setup_face_models()

