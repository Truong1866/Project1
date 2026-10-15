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

MODEL_PATH = ["yolov8n_320_openvino_model", "yolov8n_640_openvino_model", "yolov26n_320_openvino_model",
              "yolov26n_640_openvino_model"]
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


def evaluate_accuracy(model_path):
    """Hàm đánh giá mAP của mô hình trên tập coco128 bằng engine của Ultralytics"""
    print(f"\n--- ĐÁNH GIÁ ĐỘ CHUẨN (mAP) CHO {model_path} ---")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Lỗi: Không tìm thấy thư viện 'ultralytics'. Hãy chạy: pip install ultralytics")
        return

    image_size = int(model_path.split("_")[1])

    try:
        # Nạp model OpenVINO bằng Ultralytics
        model = YOLO(model_path, task='detect')

        # Chạy validation
        # Nên dùng thiết bị CPU để đo mAP cho ổn định, tránh lỗi tràn VRAM iGPU khi chạy nhiều model
        results = model.val(
            data="coco128.yaml",
            imgsz=image_size,
            device="cpu",
            verbose=False,  # Tắt bớt log dài dòng của ultralytics
            plots=False  # Tắt việc vẽ ảnh kết quả lưu ra ổ cứng để tiết kiệm thời gian
        )

        # Lấy mAP@0.5 và mAP@0.5:0.95 trung bình cho toàn bộ các lớp (All Classes)
        map50_all = results.box.map50
        map_all = results.box.map

        # Lấy độ chuẩn riêng cho lớp Người (Person) - trong COCO, Person là class id 0
        person_map50 = 0.0
        person_map = 0.0

        # results.box.ap_class_index chứa danh sách các class id thực sự xuất hiện trong dataset
        if 0 in results.box.ap_class_index:
            # Tìm vị trí (index) của class 0 trong mảng kết quả
            idx = list(results.box.ap_class_index).index(0)
            person_map50 = results.box.ap50[idx]
            person_map = results.box.ap[idx]

        print(f"Độ chuẩn Trung bình (Tất cả Class): mAP@0.5 = {map50_all:.4f} | mAP@0.5:0.95 = {map_all:.4f}")
        print(f"Độ chuẩn riêng lớp Người (Person) : mAP@0.5 = {person_map50:.4f} | mAP@0.5:0.95 = {person_map:.4f}")
        print("-" * 50)

    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá mAP: {e}")
        print("-" * 50)


def run_native_openvino(model_path):
    """Chạy Native OpenVINO (không dính Torch/YOLO) - dùng bộ xử lý yolo_native.py"""
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    print(f"BẮT ĐẦU TEST TỐC ĐỘ: {model_path} TRÊN {GPU_DEVICE}")
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

        start_time = time.perf_counter()
        det = yolo(frame)
        infer_time = time.perf_counter() - start_time

        if frame_count > 0:  # Bỏ qua frame warmup
            times.append(infer_time)

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

    del yolo
    gc.collect()
    time.sleep(1)
    return avg_fps, actual_device


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Lỗi: Không tìm thấy video '{VIDEO_PATH}' (đang tìm tại: {os.path.abspath(VIDEO_PATH)}).")
        exit()

    # Tạo trước thư mục dataset nếu cần thiết
    print("=== BẮT ĐẦU BENCHMARK SO SÁNH ===")

    for model_dir in MODEL_PATH:
        if os.path.exists(model_dir):
            print(f"\n=======================================================")
            print(f"=== XỬ LÝ MODEL: {model_dir} ===")
            print(f"=======================================================")

            # 1. Chạy đánh giá độ chuẩn trước
            # evaluate_accuracy(model_dir)

            # 2. Chạy test tốc độ thực tế (Native OpenVINO)
            fps_gpu, ov_device = run_native_openvino(model_dir)
        else:
            print(f"Bỏ qua: Không tìm thấy model '{model_dir}'. Hãy kiểm tra lại.")