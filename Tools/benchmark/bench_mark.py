import os
import time
import openvino as ov  # noqa: F401  (chỉ để nạp openvino TRƯỚC cv2; KHÔNG tạo ov.Core() ở file này)
import cv2
import numpy as np
from yolo_native import YOLOOpenVINO, box_label
import gc

VIDEO_PATH = "input.mp4"  # Đường dẫn video benchmark

# Dùng CHUNG cho cả 2 phương pháp để so sánh công bằng
CONF = 0.5
IOU = 0.4
CLASSES = [0]  # Chỉ detect Person

MODEL_PATH = ["yolov8n_320_openvino_model", "yolov8n_640_openvino_model", "yolov26n_320_openvino_model", "yolov26n_640_openvino_model"]
GPU_DEVICE = "GPU"  # "GPU" cho đồ họa Intel, hoặc "CPU" nếu muốn benchmark Native CPU
PERFORMANCE_HINT = "LATENCY"  # "LATENCY" | "THROUGHPUT"


# ==========================================


def avg_fps_from_times(times):
    """FPS trung bình từ danh sách thời gian mỗi frame (đã bỏ frame warmup)."""
    if not times:
        return 0.0
    return len(times) / sum(times)

def build_openvino_model(model_path):
    image_size = int(model_path.split("_")[1])
    print(f"Image size: {image_size}")
    def make(dev):
        return YOLOOpenVINO(
            model_path,
            device=dev,
            imgsz=image_size,  # chỉ dùng nếu model có shape động
            conf=CONF,
            iou=IOU,
            classes=CLASSES,
            performance_hint=PERFORMANCE_HINT,
            cache_dir=None,  # không dùng cache kernel để điều kiện benchmark giống nhau
        )

    try:
        return make(GPU_DEVICE), GPU_DEVICE
    except Exception as e:  # noqa: BLE001
        if GPU_DEVICE.upper() == "CPU":
            raise
        print(f"Cảnh báo: Khởi tạo trên {GPU_DEVICE} thất bại ({type(e).__name__}: {e}).")
        print("-> Chuyển về CPU cho OpenVINO.")
        return make("CPU"), "CPU"


def run_native_openvino(model_path):
    """Chạy Native OpenVINO (không dính Torch/YOLO) - dùng bộ xử lý yolov8_openvino.py"""
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    print(f"BẮT ĐẦU TEST: {model_path} TRÊN {GPU_DEVICE}")
    yolo, actual_device = build_openvino_model(model_path)
    print(f"Input model: {yolo.input_layer.shape} | Device: {actual_device}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    out_video = cv2.VideoWriter(
        os.path.join(output_dir, "output_" + model_path + ".mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_count = 0
    times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Đo: tiền xử lý (LetterBox) + Inference + hậu xử lý (NMS, scale_boxes)
        # -> tương đương phần model.predict() của Ultralytics (không tính phần vẽ)
        start_time = time.perf_counter()
        det = yolo(frame)
        infer_time = time.perf_counter() - start_time

        if frame_count > 0:  # Bỏ qua frame warmup
            times.append(infer_time)

        # Vẽ kết quả (nằm ngoài vùng đo thời gian, giống bên YOLO CPU)
        for box, score in zip(det.boxes, det.conf):
            box_label(frame, box, f"Person {score:.2f}", (0, 255, 0))

        current_fps = 1.0 / infer_time if infer_time > 0 else 0
        cv2.putText(frame, f"Native OV {actual_device} | FPS: {current_fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out_video.write(frame)
        frame_count += 1

    cap.release()
    out_video.release()

    avg_fps = avg_fps_from_times(times)
    print(f"Model {model_path} hoàn thành Quét: {frame_count} frames | Tốc độ: {avg_fps:.2f} FPS\n")

    del yolo  # Xóa tham chiếu tới model
    gc.collect()  # Ép Python dọn dẹp rác C++ dưới nền
    time.sleep(1)  # Nghỉ 1 giây để driver GPU hoàn tất giải phóng VRAM
    return avg_fps, actual_device


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Lỗi: Không tìm thấy video '{VIDEO_PATH}' (đang tìm tại: {os.path.abspath(VIDEO_PATH)}).")
        exit()

    print("=== BẮT ĐẦU BENCHMARK SO SÁNH ===")

    for model_dir in MODEL_PATH:
        if os.path.exists(model_dir):
            print("=== MODEL: " + model_dir +  " ===")
            fps_gpu, ov_device = run_native_openvino(model_dir)
        else:
            print(f"Bỏ qua Native OV: Không tìm thấy model '{model_dir}' "
                f"(đang tìm tại: {os.path.abspath(model_dir)}).)"
                f"Hãy kiểm tra lại tên thư mục chứa file .xml/.bin.")
