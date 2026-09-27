import cv2
import time
from BusinessLayer.camera_manager import CameraStream
from BusinessLayer.inference.inference_engine import InferenceEngine  # Hoặc core.inference.inference_engine tuỳ thư mục của bạn


class SmartVisionPipeline:
    def __init__(self, yolo_path, face_dir, camera_sources):
        """
        Khởi tạo Pipeline với đường dẫn model và danh sách camera.
        """
        print("=== KHỞI TẠO HỆ THỐNG SMART VISION ===")
        # 1. Nạp AI Models (Inference Engine)
        self.engine = InferenceEngine(yolo_path, face_dir)

        # 2. Khởi tạo danh sách Camera
        self.cameras = []
        for i, source in enumerate(camera_sources):
            # Khởi tạo nhưng chưa start vội
            cam = CameraStream(camera_id=f"Cam_{i + 1}", source=source)
            self.cameras.append(cam)

    def run(self):
        """
        Bắt đầu vòng lặp chính của hệ thống.
        """
        # Khởi động tất cả các luồng đọc camera
        for cam in self.cameras:
            cam.start()

        print("\n=> Hệ thống đang chạy! Nhấn 'q' trên cửa sổ video để thoát.")

        try:
            while True:
                # Duyệt qua từng camera để lấy ảnh
                for cam in self.cameras:
                    frame = cam.read()

                    # Nếu frame None (nghĩa là không có chuyển động), bỏ qua
                    if frame is not None:
                        self._process_and_display(cam.camera_id, frame)

                # Cần waitKey để giao diện OpenCV cập nhật
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Dọn dẹp tài nguyên khi thoát
            print("\nĐang tắt hệ thống...")
            for cam in self.cameras:
                cam.stop()
            cv2.destroyAllWindows()

    def _process_and_display(self, camera_id, frame):
        """
        Hàm nội bộ: Đưa ảnh qua AI và vẽ khung hình.
        """
        # 1. AI phát hiện người (YOLOv8)
        persons = self.engine.detect_persons(frame)

        for p in persons:
            x1, y1, x2, y2 = p["box"]

            # Vẽ khung xanh dương cho Người
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 2. AI trích xuất khuôn mặt trong vùng người đó (InsightFace)
            face_data = self.engine.process_faces(frame, p["box"])

            for face in face_data:
                fx1, fy1, fx2, fy2 = face["abs_box"]
                # vector_dac_trung = face["embedding"] # Sẽ dùng cái này ở bước sau

                # Vẽ khung xanh lá cho Khuôn mặt
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (fx1, fy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. Hiển thị kết quả của camera tương ứng
        cv2.imshow(f"Live - {camera_id}", frame)