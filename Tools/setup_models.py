from ultralytics import YOLO
from insightface.app import FaceAnalysis
import os
import shutil
from pathlib import Path

current_file_path = Path(__file__).resolve()
models_dir = current_file_path.parent.parent / 'Models'

def setup_yolo():
    print("=== BƯỚC 1: TẢI VÀ KIỂM TRA YOLOv8n ===")
    # 1. Tải mô hình gốc định dạng PyTorch (.pt)
    model = YOLO(models_dir / 'yolov8n.pt')

    print("-> Đang chạy thử nghiệm (Inference Test) để kiểm tra lỗi...")
    try:
        # Dùng một bức ảnh mẫu mặc định của thư viện để benchmark
        results = model.predict(source="https://ultralytics.com/images/bus.jpg", imgsz=640, verbose=False)
        print("=> [Thành công] Mô hình YOLOv8n gốc hoạt động bình thường!")
    except Exception as e:
        print(f"=> [Thất bại] Lỗi khi chạy mô hình: {e}")
        return

    print("\n=== BƯỚC 2: CHUYỂN HÓA SANG OPENVINO ===")
    print("-> Đang biên dịch sang OpenVINO (FP16) cho Intel Iris Xe iGPU...")
    # Lệnh export: half=True sẽ chuyển trọng số sang FP16 (Float16)
    # Giúp giảm một nửa dung lượng RAM/VRAM và tăng tốc độ xử lý mà không giảm độ chính xác
    default_export_dir = models_dir / 'yolov8n_openvino_model'
    # --- Xuất bản imgsz = 640 ---
    print("-> Đang xuất bản imgsz=640...")
    model.export(format='openvino', imgsz=640, quantize = 'FP16')
    export_640_path = models_dir / 'yolov8n_640_openvino_model'

    # Xóa thư mục cũ nếu đã tồn tại và đổi tên thư mục mặc định thành tên mới
    if export_640_path.exists():
        shutil.rmtree(export_640_path)
    os.rename(default_export_dir, export_640_path)

    # --- Xuất bản imgsz = 320 ---
    print("-> Đang xuất bản imgsz=320...")
    model.export(format='openvino', imgsz=320, quantize = 'FP16')
    export_320_path = models_dir / 'yolov8n_320_openvino_model'

    # Xóa thư mục cũ nếu đã tồn tại và đổi tên thư mục mặc định thành tên mới
    if export_320_path.exists():
        shutil.rmtree(export_320_path)
    os.rename(default_export_dir, export_320_path)

    print(f"=> [Thành công] Mô hình đã được chuyển hóa và lưu tại 2 thư mục:")
    print(f"   1. {export_640_path}")
    print(f"   2. {export_320_path}")


def setup_face_models():
    print("\n=== BƯỚC 3: TẢI MÔ HÌNH NHẬN DIỆN KHUÔN MẶT ===")
    # Tạo thư mục chứa model cho gọn gàng
    os.makedirs('../Models/face_models', exist_ok=True)

    try:
        # buffalo_s là bộ model nhẹ (Small) rất phù hợp cho CPU/iGPU
        # Bao gồm cả Detection (tìm mặt) và Recognition (trích xuất vector)
        app = FaceAnalysis(name='buffalo_s', root=models_dir / 'face_models')
        app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 dùng CPU làm mặc định khi chuẩn bị
        print("=> [Thành công] Mô hình khuôn mặt (ONNX) đã tải xong và sẵn sàng dùng với OpenVINO!")
    except Exception as e:
        print(f"=> [Thất bại] Lỗi khi tải mô hình khuôn mặt: {e}")


if __name__ == '__main__':
    setup_yolo()
    setup_face_models()