import openvino as ov
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2
import numpy as np


class InferenceEngine:
    def __init__(self, yolo_model_path, face_model_dir):
        """
        Khởi tạo các mô hình và đẩy lên thiết bị phần cứng tối ưu.
        """
        print("Đang khởi tạo Inference Engine...")
        self.core = ov.Core()

        # 1. Tìm thiết bị GPU (Intel Iris Xe)
        # Nếu không tìm thấy GPU, hệ thống sẽ tự động rơi về (fallback) CPU.
        self.device = 'GPU' if 'GPU' in self.core.available_devices else 'CPU'
        print(f"-> Thiết bị Inference mặc định: {self.device}")

        # 2. Khởi tạo YOLOv8 bằng OpenVINO
        try:
            print(f"-> Nạp mô hình YOLOv8 từ: {yolo_model_path} lên {self.device}...")
            # Ultralytics tự động xử lý giao tiếp OpenVINO qua định dạng thư mục
            self.yolo_model = YOLO(yolo_model_path, task='detect')
        except Exception as e:
            raise RuntimeError(f"Lỗi nạp YOLO: {e}")

        # 3. Khởi tạo InsightFace (Phát hiện mặt & Trích xuất vector)
        try:
            print(f"-> Nạp mô hình Khuôn mặt từ: {face_model_dir}...")
            # providers: Dùng OpenVINOExecutionProvider để chạy ONNX bằng Intel Iris Xe
            # Nếu máy không có OpenVINO provider cài sẵn, nó sẽ dùng CPUExecutionProvider
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']

            self.face_app = FaceAnalysis(name='buffalo_s',
                                         root=face_model_dir,
                                         providers=providers)

            # Cấu hình ngưỡng det_thresh để phát hiện khuôn mặt (từ 0 đến 1)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        except Exception as e:
            raise RuntimeError(f"Lỗi nạp InsightFace: {e}")

        print("=> Inference Engine sẵn sàng!")

    def detect_persons(self, frame):
        """
        Nhận đầu vào là 1 khung hình (numpy array), trả về danh sách các bounding box của 'người'.
        """
        # Ép YOLOv8 sử dụng CPU thay vì tự động tìm card NVIDIA
        results = self.yolo_model.predict(
            source=frame,
            classes=[0],  # 0 là ID của nhãn 'person' trong COCO
            device='cpu',  # BẮT BUỘC SỬA THÀNH 'cpu'
            verbose=False
        )

        # Kết quả là một list, do ta truyền 1 ảnh nên lấy phần tử đầu tiên
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # Tọa độ [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # Độ tin cậy

        persons = []
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            persons.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(conf)
            })

        return persons

    def process_faces(self, frame, person_box):
        """
        Cắt (crop) vùng có người và nhận diện khuôn mặt trong vùng đó.
        Trả về danh sách các vector đặc trưng khuôn mặt (embeddings).
        """
        x1, y1, x2, y2 = person_box

        # Cắt lấy vùng ảnh chỉ chứa 1 người
        # Cần đảm bảo tọa độ không vượt quá kích thước ảnh
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        person_crop = frame[y1:y2, x1:x2]

        # Bỏ qua nếu khung cắt quá nhỏ (rác/lỗi)
        if person_crop.shape[0] < 50 or person_crop.shape[1] < 50:
            return []

        # Cho InsightFace tìm mặt và trích xuất vector TRONG khung ảnh người đã cắt
        faces = self.face_app.get(person_crop)

        face_data = []
        for face in faces:
            # face.embedding là vector 512 chiều đại diện cho khuôn mặt
            face_data.append({
                "embedding": face.embedding,
                # Tọa độ mặt này là tọa độ tương đối trong khung person_crop
                # Cần cộng thêm x1, y1 để ra tọa độ tuyệt đối trên ảnh gốc
                "abs_box": [
                    int(face.bbox[0] + x1), int(face.bbox[1] + y1),
                    int(face.bbox[2] + x1), int(face.bbox[3] + y1)
                ]
            })

        return face_data


if __name__ == '__main__':
    # Đường dẫn dựa theo kết quả chạy setup_models.py của bạn
    YOLO_DIR = "../../Models/yolov8n_openvino_model/"
    FACE_DIR = "../../Models/face_models/"

    try:
        engine = InferenceEngine(YOLO_DIR, FACE_DIR)
        print("\nĐã nạp thành công mô hình lên", engine.device)
    except Exception as e:
        print("Có lỗi xảy ra:", e)