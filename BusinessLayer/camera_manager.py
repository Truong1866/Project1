import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import threading
import queue
import time
import numpy as np


class CameraStream:
    def __init__(self, camera_id, source, queue_size=3, motion_threshold=1000):
        self.camera_id = camera_id
        self.source = source
        self.motion_threshold = motion_threshold

        # Hàng chờ chứa frame. maxsize nhỏ (2-3) để đảm bảo luôn xử lý ảnh mới nhất (Real-time)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False

        # Khởi tạo thuật toán trừ nền (Background Subtraction MOG2)
        # detectShadows=False giúp thuật toán chạy nhanh hơn
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        self.cap = cv2.VideoCapture(self.source)
        # Khởi động luồng (thread) chạy ngầm
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True

    def start(self):
        print(f"[Camera {self.camera_id}] Bắt đầu đọc luồng từ {self.source}")
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                print(f"[Camera {self.camera_id}] Mất kết nối. Đang thử lại...")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            # Thu nhỏ ảnh để thuật toán phát hiện chuyển động chạy siêu nhanh
            small_frame = cv2.resize(frame, (640, 480))

            # Tạo mặt nạ đen trắng (trắng = có vật thể chuyển động)
            fg_mask = self.bg_subtractor.apply(small_frame)

            # Đếm số lượng điểm ảnh (pixel) màu trắng
            motion_level = cv2.countNonZero(fg_mask)

            # Nếu mức độ chuyển động vượt ngưỡng (có người/vật đi qua)
            if motion_level > self.motion_threshold:
                # Nếu hàng chờ đầy, bỏ frame cũ nhất đi (Drop frame)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Đẩy frame gốc (độ phân giải cao) vào hàng chờ cho AI xử lý
                self.frame_queue.put(frame)

            # Thêm 1 độ trễ cực nhỏ để CPU không bị vắt kiệt (100%)
            time.sleep(0.01)

    def read(self):
        """Lấy frame từ hàng chờ. Trả về None nếu không có chuyển động."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
        print(f"[Camera {self.camera_id}] Đã dừng.")


if __name__ == '__main__':
    # Bạn có thể thay 0 bằng đường dẫn RTSP của camera thật
    # VD: "rtsp://admin:password@192.168.1.10:554/stream1"
    cam_source = 0

    cam = CameraStream(camera_id="Cam_Test", source=cam_source).start()

    print("Đang hiển thị Camera (Nhấn 'q' để thoát)...")
    print("LƯU Ý: Khung hình chỉ cập nhật khi có chuyển động!")

    try:
        while True:
            frame = cam.read()

            if frame is not None:
                # Ghi chú lên ảnh để biết frame này đã lọt qua bộ lọc
                cv2.putText(frame, "MOTION DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Test Camera Manager", frame)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()