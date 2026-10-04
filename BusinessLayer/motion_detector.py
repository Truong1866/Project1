"""Phát hiện chuyển động có LỌC NHIỄU, mục tiêu: chỉ "đánh thức" AI khi có khả năng là NGƯỜI.

Các lớp lọc (rẻ -> đắt), tất cả chạy trên ảnh thu nhỏ (~320px) nên chỉ tốn vài ms:
 1. Làm mờ Gauss + MOG2 (detectShadows=True) và CHỈ lấy điểm "foreground thật" (=255),
    bỏ hẳn điểm bóng đổ (=127).
 2. Morphology: mở (xoá hạt nhiễu lấm tấm của lá/mưa) rồi đóng (nối các phần thân người).
 3. Lọc theo từng vùng (blob): diện tích tối thiểu, độ "đặc" (solidity: lá rung rất thưa),
    tỉ lệ rộng/cao (bóng kéo dài rất dẹt).
 4. Đổi sáng toàn cục: nếu >45% khung hình đổi cùng lúc (bật đèn, auto-exposure, rung máy) -> bỏ qua.
 5. BẢN ĐỒ NHIỄU HỌC ĐƯỢC (clutter map): khi AI chạy mà KHÔNG thấy người, pipeline báo "báo động giả"
    -> vùng chuyển động đó được cộng điểm. Vùng đủ điểm (cây, cờ, quạt...) sẽ bị bỏ qua ở lần sau.
    Điểm tự phai dần khi vùng đó yên tĩnh, và bị xoá ngay khi có người xuất hiện tại đó.
Việc xác nhận cuối cùng "có phải người không" luôn là YOLO; bộ lọc này chỉ để không gọi YOLO vô ích.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MotionResult:
    valid: bool = False
    boxes: list = field(default_factory=list)      # vùng hợp lệ (x1,y1,x2,y2) theo toạ độ ảnh gốc
    rejected: list = field(default_factory=list)   # vùng bị loại (để debug hiển thị)
    fg_ratio: float = 0.0
    reason: str = ""


class MotionDetector:
    def __init__(self, width: int = 320, min_area_ratio: float = 0.003, min_solidity: float = 0.30,
                 max_aspect: float = 4.0, global_change_ratio: float = 0.45, history: int = 300,
                 var_threshold: float = 32, warmup_frames: int = 20, clutter_limit: float = 0.6,
                 clutter_gain: float = 0.5, clutter_decay: float = 0.9995, clutter_threshold: float = 1.0):
        self.width = int(width)
        self.min_area_ratio = float(min_area_ratio)
        self.min_solidity = float(min_solidity)
        self.max_aspect = float(max_aspect)
        self.global_change_ratio = float(global_change_ratio)
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.warmup_frames = int(warmup_frames)
        self.clutter_limit = float(clutter_limit)
        self.clutter_gain = float(clutter_gain)
        self.clutter_decay = float(clutter_decay)
        self.clutter_threshold = float(clutter_threshold)

        self._lock = threading.Lock()
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._shape = None
        self._scale = 1.0
        self._init_buffers(None)

    # ------------------------------------------------------------------ nội bộ
    def _new_mog(self):
        mog = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.var_threshold,
                                                 detectShadows=True)
        mog.setShadowThreshold(0.5)
        return mog

    def _init_buffers(self, shape) -> None:
        self.mog = self._new_mog()
        self._frames = 0
        self._shape = shape
        if shape is not None:
            self.clutter = np.zeros(shape, dtype=np.float32)
            self._window = np.zeros(shape, dtype=np.uint8)
        else:
            self.clutter = self._window = None

    # ------------------------------------------------------------------ API chính
    def update(self, frame: np.ndarray) -> MotionResult:
        h0, w0 = frame.shape[:2]
        small_h = max(8, int(round(h0 * self.width / w0)))
        small = cv2.resize(frame, (self.width, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._scale = w0 / self.width

        with self._lock:
            if gray.shape != self._shape:  # lần đầu hoặc đổi độ phân giải
                self._init_buffers(gray.shape)

            fg = self.mog.apply(gray)
            self._frames += 1
            if self._frames <= self.warmup_frames:
                return MotionResult(reason="warmup")

            mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]   # bỏ bóng (127)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close)
            fg_ratio = float(cv2.countNonZero(mask)) / mask.size

            # Bản đồ nhiễu chỉ phai ở nơi đang yên tĩnh (cây còn lay thì vẫn giữ nguyên "án")
            near = cv2.dilate(mask, self._k_near) > 0
            np.multiply(self.clutter, self.clutter_decay, out=self.clutter, where=~near)

            if fg_ratio > self.global_change_ratio:
                return MotionResult(fg_ratio=fg_ratio, reason="global_change")

            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            clutter_mask = self.clutter >= self.clutter_threshold
            min_area = max(12, int(self.min_area_ratio * mask.size))
            s = self._scale

            valid_boxes, rejected = [], []
            valid_mask = np.zeros_like(mask)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                if area < min_area:
                    continue
                box = (int(x * s), int(y * s), int((x + w) * s), int((y + h) * s))
                solidity = area / float(w * h)
                aspect = w / float(h)
                region = labels[y:y + h, x:x + w] == i
                if solidity < self.min_solidity:
                    rejected.append(box)       # thưa -> lá/cành rung
                elif aspect > self.max_aspect or aspect < 1.0 / (self.max_aspect * 1.5):
                    rejected.append(box)       # quá dẹt/quá mảnh -> bóng, vệt sáng
                elif float(clutter_mask[y:y + h, x:x + w][region].mean()) >= self.clutter_limit:
                    rejected.append(box)       # vùng đã học là nhiễu
                else:
                    valid_boxes.append(box)
                    valid_mask[y:y + h, x:x + w][region] = 255

            if valid_boxes:
                cv2.bitwise_or(self._window, valid_mask, dst=self._window)
            return MotionResult(valid=bool(valid_boxes), boxes=valid_boxes, rejected=rejected,
                                fg_ratio=fg_ratio, reason="ok" if valid_boxes else "filtered")

    # ------------------------------------------------------------------ phản hồi từ AI
    def begin_window(self) -> None:
        """Bắt đầu một đợt chuyển động mới (khi cổng AI mở)."""
        with self._lock:
            if self._window is not None:
                self._window[:] = 0

    def report_false_alarm(self) -> None:
        """AI đã chạy cả đợt mà không thấy người -> học vùng chuyển động của đợt này là nhiễu."""
        with self._lock:
            if self.clutter is None:
                return
            region = self._window > 0
            self.clutter[region] = np.minimum(self.clutter[region] + self.clutter_gain, 2.0)
            self._window[:] = 0

    def report_person(self, boxes) -> None:
        """Có người ở các box này (toạ độ ảnh gốc) -> xoá bản đồ nhiễu tại đó cho khỏi bỏ sót."""
        with self._lock:
            if self.clutter is None or self._shape is None:
                return
            hh, ww = self._shape
            for x1, y1, x2, y2 in boxes:
                a, b = max(0, int(x1 / self._scale)), max(0, int(y1 / self._scale))
                c, d = min(ww, int(x2 / self._scale) + 1), min(hh, int(y2 / self._scale) + 1)
                if c > a and d > b:
                    self.clutter[b:d, a:c] *= 0.3

    def reset(self) -> None:
        with self._lock:
            self._init_buffers(None)
