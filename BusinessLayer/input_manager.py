import cv2
import threading
import time
import queue


class InputManager:
    def __init__(self, source):
        """
        Khởi tạo bộ quản lý đầu vào.
        - source: có thể là số (0, 1 cho webcam),
                  đường dẫn file ("video.mp4"),
                  hoặc URL camera mạng ("rtsp://...").
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở nguồn đầu vào: {source}")

        # Trích xuất thông tin cơ bản
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Phân loại nguồn: Nếu là luồng mạng (rtsp, http) hoặc webcam (số) -> Live Stream
        self.is_live = isinstance(source, int) or str(source).startswith(('rtsp://', 'http://', 'https://'))

        self.running = True
        self.latest_frame = None

        if self.is_live:
            # Dùng Thread để đọc liên tục xóa buffer cho Live Stream
            self.lock = threading.Lock()
            self.thread = threading.Thread(target=self._update_stream, daemon=True)
            self.thread.start()

            # Đợi một chút để thread lấy được frame đầu tiên
            time.sleep(0.5)

    def _update_stream(self):
        """Luồng chạy ngầm liên tục đọc camera để tránh đầy buffer (Chỉ dùng cho luồng Live)"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Nếu đứt kết nối, thử tự động kết nối lại (Auto-reconnect)
                print(f"Cảnh báo: Mất tín hiệu luồng {self.source}. Đang thử kết nối lại...")
                time.sleep(2)
                self.cap.open(self.source)
                continue

            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        """
        Lấy frame chuẩn hóa để đưa vào AI xử lý.
        Trả về: (ret, frame)
        """
        if self.is_live:
            # Trả về frame mới nhất từ luồng nền
            with self.lock:
                if self.latest_frame is not None:
                    return True, self.latest_frame.copy()
                return False, None
        else:
            # Đọc tuần tự cho file MP4
            ret, frame = self.cap.read()
            return ret, frame

    def release(self):
        """Giải phóng tài nguyên"""
        self.running = False
        if self.is_live and self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()