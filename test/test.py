import os
import time
import cv2
import shutil
from ultralytics import YOLO
import openvino as ov

def export_models(base_model_path='yolov8n.pt'):
    """
    Tải và export model YOLOv8 sang định dạng OpenVINO với các mức lượng tử hóa và resolution khác nhau.
    """
    print(f"--- BẮT ĐẦU EXPORT MODEL: {base_model_path} ---")
    resolutions = [640, 320] # 640 là default, 320 là mức thấp hơn
    precisions = ['fp32', 'fp16', 'int8']

    exported_paths = {}

    for sz in resolutions:
        for prec in precisions:
            print(f"\n[Exporting] Resolution: {sz}x{sz} | Precision: {prec.upper()}")
            model = YOLO(base_model_path)

            export_kwargs = {
                'format': 'openvino',
                'imgsz': sz,
            }

            # Cấu hình cờ lượng tử hóa tương ứng
            if prec == 'fp16':
                export_kwargs['half'] = True
            elif prec == 'int8':
                export_kwargs['int8'] = True
                export_kwargs['data'] = 'coco8.yaml' # INT8 cần dữ liệu để calibration

            exported_dir = model.export(**export_kwargs)

            # Đổi tên thư mục model sau khi export để không bị ghi đè
            new_dir_name = f"yolov8n_ov_{prec}_{sz}"
            if os.path.exists(new_dir_name):
                shutil.rmtree(new_dir_name)
            os.rename(exported_dir, new_dir_name)

            exported_paths[f"{prec}_{sz}"] = new_dir_name
            print(f"Đã lưu model tại: {new_dir_name}")

    return exported_paths

def setup_openvino_device():
    """Kiểm tra thiết bị AI hiện có và quyết định device chạy."""
    core = ov.Core()
    devices = core.available_devices
    print("\n--- KIỂM TRA THIẾT BỊ OPENVINO ---")
    print(f"Các thiết bị khả dụng: {devices}")

    # Ưu tiên GPU (như Intel Iris Xe trên i7-1355U), fallback về CPU
    if 'GPU' in devices:
        print("-> Đã tìm thấy GPU! Sẽ ép chạy inference trên GPU.")
        return 'GPU'
    else:
        print("-> Không tìm thấy hoặc không khởi tạo được GPU. Fallback sử dụng CPU.")
        return 'CPU'

def evaluate_video(model_dir, video_path, output_dir="result", device='CPU', target_classes=[0]):
    """
    Chạy model OpenVINO trên video, đo FPS, vẽ bounding box và lưu video đầu ra.
    target_classes: [0] là class 'person' trong hệ COCO.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ VIDEO VỚI MODEL {model_dir} TRÊN {device} ---")

    # Load model đã export bằng Ultralytics
    # Cảnh báo: khi load OpenVINO model bằng YOLO, tham số task='detect' là bắt buộc
    model = YOLO(model_dir, task='detect')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Thông số video đầu ra
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    out_video_path = os.path.join(output_dir, f"output_{model_dir}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps_video, (width, height))

    frame_count = 0
    total_inference_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Chạy inference. Sử dụng tham số device='GPU' hoặc 'CPU' tương ứng
        results = model.predict(frame, device=device, classes=target_classes, verbose=False)

        end_time = time.time()
        inference_time = end_time - start_time

        # Bỏ qua frame đầu tiên (warm-up OpenVINO/GPU) để tính trung bình chính xác hơn
        if frame_count > 0:
            total_inference_time += inference_time

        # Vẽ bounding box lên frame
        annotated_frame = results[0].plot()

        # Ghi FPS lên góc video để dễ kiểm tra trực quan
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    # Tính toán kết quả
    valid_frames = frame_count - 1 if frame_count > 1 else 1
    avg_time_per_frame = total_inference_time / valid_frames
    avg_fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0

    print("\n[KẾT QUẢ ĐÁNH GIÁ]")
    print(f"- Số frame đã quét: {frame_count}")
    print(f"- Thời gian trung bình 1 frame: {avg_time_per_frame*1000:.2f} ms")
    print(f"- FPS trung bình: {avg_fps:.2f} FPS")
    print(f"- Video kết quả được lưu tại: {out_video_path}")

if __name__ == "__main__":
    # 1. Định nghĩa file video input của bạn ở đây
    input_video = "input.mp4" # Đổi tên thành đường dẫn video của bạn

    # Tạo video giả lập để code không bị lỗi nếu bạn chưa có sẵn file
    if not os.path.exists(input_video):
        print(f"Không tìm thấy {input_video}, vui lòng thay đường dẫn thực tế.")
        exit()

    # 2. Thực hiện Export
    exported_models = export_models('yolov8n.pt')

    # 3. Detect thiết bị chạy (ưu tiên GPU iGPU của chip Intel)
    target_device = setup_openvino_device()

    # 4. Chạy evaluate với một model cụ thể (VD: Lượng tử hóa INT8 với ảnh 320x320 để xem tốc độ tối đa)
    # Bạn có thể bỏ vào vòng lặp for exported_models.values() để quét toàn bộ.
    model_to_test = exported_models.get('int8_320', 'yolov8n_ov_int8_320')

    # Chỉ detect class 0 (Person). Nếu bạn có model chuyên dụng ghép mặt, xoá params `target_classes`
    evaluate_video(model_dir=model_to_test,
                   video_path=input_video,
                   output_dir="result",
                   device=target_device,
                   target_classes=[0])