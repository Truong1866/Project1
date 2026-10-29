"""[ĐÃ LỖI THỜI] CameraStream — chức năng đã được chuyển sang InputManager.

Lớp CameraStream ban đầu đọc camera bằng cv2.VideoCapture và phát hiện
chuyển động đơn giản. Toàn bộ chức năng đó đã được thay thế bởi:
  - BusinessLayer/input_manager.py  (InputManager + đọc luồng)
  - BusinessLayer/motion_detector.py (MotionDetector có lọc nhiễu nâng cao)

File này được giữ lại để không làm hỏng bất kỳ import bên ngoài nào,
nhưng KHÔNG NÊN dùng CameraStream trong code mới.
"""
import warnings

import cv2
import threading
import time
import numpy as np


class CameraStream:
    """[Deprecated] Dùng InputManager + MotionDetector thay thế."""

    def __init__(self, camera_id, source, motion_threshold=1000):
        warnings.warn(
            "CameraStream đã lỗi thời. Hãy dùng InputManager và MotionDetector thay thế.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.camera_id = camera_id
        self.source = source
        self.motion_threshold = motion_threshold

        self.current_frame = None
        self.has_motion = False
        self.stopped = False

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.cap = cv2.VideoCapture(self.source)

        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True

    def start(self):
        print(f"[Camera {self.camera_id}] Bắt đầu đọc luồng...")
        self.thread.start()
        return self

    def _update(self):
        # Luồng này chạy ngầm độc lập, liên tục rút khung hình để giữ độ mượt 30 FPS
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            # Thuật toán lọc chuyển động
            small_frame = cv2.resize(frame, (640, 480))
            fg_mask = self.bg_subtractor.apply(small_frame)
            motion_level = cv2.countNonZero(fg_mask)

            # Cập nhật liên tục khung hình mới nhất và trạng thái chuyển động
            self.current_frame = frame
            self.has_motion = (motion_level > self.motion_threshold)

    def read(self):
        """Trả về tuple: (khung_hình, có_chuyển_động_không)"""
        if self.current_frame is not None:
            return self.current_frame.copy(), self.has_motion
        return None, False

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
