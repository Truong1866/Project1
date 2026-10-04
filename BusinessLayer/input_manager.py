from __future__ import annotations

import os
import threading
import time

# Dùng TCP cho RTSP để đỡ vỡ hình/mất gói. Phải đặt trước khi mở VideoCapture đầu tiên.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

from Utils.logger import get_logger  # noqa: E402

log = get_logger("Input")


class SourceKind:
    WEBCAM = "webcam"
    STREAM = "stream"
    FILE = "file"


class Status:
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    ERROR = "error"
    STOPPED = "stopped"


def classify_source(source) -> str:
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        return SourceKind.WEBCAM
    s = str(source).strip().lower()
    if s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")):
        return SourceKind.STREAM
    return SourceKind.FILE


class InputManager:
    def __init__(self, source, kind: str | None = None, name: str | None = None,
                 loop: bool = False, reconnect_delay: float = 2.0):
        self.source = int(source) if isinstance(source, str) and source.strip().isdigit() else source
        self.kind = kind or classify_source(self.source)
        self.name = name or str(source)
        self.loop = loop
        self.reconnect_delay = reconnect_delay

        self.status = Status.CONNECTING
        self.fps = 0.0            # FPS khai báo của nguồn (dùng để canh nhịp file video)
        self.measured_fps = 0.0   # FPS thực tế đo được
        self.width = 0
        self.height = 0
        self.progress = 0.0       # 0..1, chỉ có nghĩa với file video

        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._last_pub = 0.0
        self._stop_evt = threading.Event()
        self._cap = None
        self._thread = threading.Thread(target=self._run, name=f"Input-{self.name}", daemon=True)

    # ------------------------------------------------------------------ vòng đời
    @property
    def is_live(self) -> bool:
        return self.kind != SourceKind.FILE

    @property
    def ended(self) -> bool:
        return self.status == Status.ENDED

    def start(self) -> "InputManager":
        log.info("[%s] Bắt đầu đọc (%s): %s", self.name, self.kind, self.source)
        self._thread.start()
        return self

    def stop(self, wait: bool = True) -> None:
        self._stop_evt.set()
        if wait and self._thread.is_alive():
            self._thread.join(timeout=2.0)  # luồng sở hữu cap nên tự release khi thoát
        self.status = Status.STOPPED

    def get_frame(self):
        """Trả về (frame_id, frame). frame=None nếu chưa có khung nào."""
        with self._lock:
            return self._frame_id, self._frame

    # ------------------------------------------------------------------ nội bộ
    def _publish(self, frame) -> None:
        now = time.perf_counter()
        with self._lock:
            self._frame = frame
            self._frame_id += 1
        if self._last_pub:
            dt = now - self._last_pub
            if dt > 0:
                inst = 1.0 / dt
                self.measured_fps = inst if self.measured_fps == 0 else 0.9 * self.measured_fps + 0.1 * inst
        self._last_pub = now

    def _open(self) -> bool:
        try:
            if self.kind == SourceKind.WEBCAM:
                backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                cap = cv2.VideoCapture(int(self.source), backend)
            elif self.kind == SourceKind.STREAM:
                cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(str(self.source))
        except Exception:
            log.exception("[%s] Lỗi mở nguồn", self.name)
            return False
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1 or fps > 240:
            fps = 30.0
        self.fps = float(fps)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        return True

    def _run(self) -> None:
        first = True
        while not self._stop_evt.is_set():
            self.status = Status.CONNECTING if first else Status.RECONNECTING
            first = False
            if not self._open():
                if self.kind == SourceKind.FILE:
                    log.error("[%s] Không mở được file: %s", self.name, self.source)
                    self.status = Status.ERROR
                    return
                log.warning("[%s] Không kết nối được, thử lại sau %.0fs", self.name, self.reconnect_delay)
                self.status = Status.RECONNECTING
                if self._stop_evt.wait(self.reconnect_delay):
                    break
                continue

            self.status = Status.RUNNING
            self._last_pub = 0.0
            try:
                finished = self._read_file() if self.kind == SourceKind.FILE else self._read_live()
            finally:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
            if self.kind == SourceKind.FILE:
                if not self._stop_evt.is_set():
                    self.status = Status.ENDED if finished else Status.ERROR
                return
            if self._stop_evt.is_set():
                break
            log.warning("[%s] Mất tín hiệu, đang kết nối lại...", self.name)
            self.status = Status.RECONNECTING
            if self._stop_evt.wait(self.reconnect_delay):
                break
        self.status = Status.STOPPED

    def _read_live(self) -> bool:
        """Đọc liên tục để luôn có khung mới nhất. Trả về khi mất tín hiệu."""
        fails = 0
        while not self._stop_evt.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                fails += 1
                if fails >= 10:
                    return False
                time.sleep(0.05)
                continue
            fails = 0
            self._publish(frame)
        return True

    def _read_file(self) -> bool:
        """Phát file đúng FPS gốc theo đồng hồ thực. True nếu phát hết file bình thường."""
        cap = self._cap
        interval = 1.0 / self.fps
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        start = time.perf_counter()
        idx = 0
        while not self._stop_evt.is_set():
            lag = time.perf_counter() - (start + idx * interval)
            if lag < 0:
                if self._stop_evt.wait(-lag):
                    return True
            elif lag > 2 * interval:  # decode không kịp -> bỏ bớt khung để bắt kịp thời gian thực
                skip = int(lag / interval)
                for _ in range(skip):
                    if not cap.grab():
                        break
                idx += skip

            ret, frame = cap.read()
            if not ret or frame is None:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start, idx = time.perf_counter(), 0
                    continue
                self.progress = 1.0
                return True
            idx += 1
            if total > 0:
                self.progress = min(1.0, idx / total)
            self._publish(frame)
        return True