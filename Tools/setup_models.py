from pathlib import Path
import os
import shutil
from ultralytics import YOLO
from insightface.app import FaceAnalysis

current_file_path = Path(__file__).resolve()
models_dir = current_file_path.parent.parent / 'Models'
models_dir.mkdir(parents=True, exist_ok=True)


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


def export_openvino(model_path: Path, model_name: str):
    """
    Biên dịch mô hình sang OpenVINO với quantize='FP16' cho 2 phiên bản imgsz: 640 và 320.
    """
    print(f"\n-> Đang biên dịch {model_name} sang OpenVINO (FP16) cho Intel Iris Xe iGPU/CPU...")

    prefix = model_name if model_name.endswith('n') else f"{model_name}n"

    for imgsz in (640, 320):
        print(f"-> Đang xuất bản {prefix} với imgsz={imgsz}, quantize='FP16'...")
        target_dir = models_dir / f"{prefix}_{imgsz}_openvino_model"

        # Khởi tạo instance YOLO riêng biệt cho từng export để đảm bảo cấu hình imgsz độc lập
        model = YOLO(str(model_path))
        try:
            export_result = model.export(
                format='openvino',
                imgsz=imgsz,
                quantize='FP16'
            )
        except Exception as e:
            print(f"   [Cảnh báo] quantize='FP16' gặp lỗi ({e}), chuyển sang fallback half=True...")
            export_result = model.export(
                format='openvino',
                imgsz=imgsz,
                half=True
            )

        exported_path = Path(export_result)
        exported_dir = exported_path if exported_path.is_dir() else exported_path.parent

        if target_dir.exists() and target_dir.resolve() != exported_dir.resolve():
            shutil.rmtree(target_dir)

        if exported_dir.resolve() != target_dir.resolve():
            shutil.move(str(exported_dir), str(target_dir))

        print(f"=> [Thành công] Mô hình {prefix} (imgsz={imgsz}) đã được lưu tại: {target_dir}")


def setup_yolo():
    print("=== BƯỚC 1: TẢI VÀ KIỂM TRA YOLOv8 ===")
    path_v8 = get_model_path('yolov8n.pt', ['yolov8n.pt'])
    modelv8 = YOLO(str(path_v8))

    print("-> Đang chạy thử nghiệm (Inference Test) cho YOLOv8n...")
    try:
        results = modelv8.predict(source="https://ultralytics.com/images/bus.jpg", imgsz=640, verbose=False)
        print("=> [Thành công] Mô hình YOLOv8n gốc hoạt động bình thường!")
    except Exception as e:
        print(f"=> [Thất bại] Lỗi khi chạy mô hình YOLOv8n: {e}")
        return

    print("\n=== BƯỚC 2: TẢI VÀ KIỂM TRA YOLOv26 ===")
    path_v26 = get_model_path('yolov26n.pt', ['yolo26n.pt', 'yolov26n.pt'])
    modelv26 = YOLO(str(path_v26))

    print("-> Đang chạy thử nghiệm (Inference Test) cho YOLOv26n...")
    try:
        results = modelv26.predict(source="https://ultralytics.com/images/bus.jpg", imgsz=640, verbose=False)
        print("=> [Thành công] Mô hình YOLOv26n gốc hoạt động bình thường!")
    except Exception as e:
        print(f"=> [Thất bại] Lỗi khi chạy mô hình YOLOv26n: {e}")
        return

    print("\n=== BƯỚC 3: CHUYỂN HÓA SANG OPENVINO (quantize='FP16', imgsz=640 & 320) ===")
    export_openvino(path_v8, "yolov8")
    export_openvino(path_v26, "yolov26")


def setup_face_models():
    print("\n=== BƯỚC 4: TẢI MÔ HÌNH NHẬN DIỆN KHUÔN MẶT ===")
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
    setup_face_models()
