"""Điều khiển FOCUS: bam theo một mục tiêu, phóng to mượt bằng "camera ảo".

Máy trạng thái của mỗi nguồn:

    IDLE ──start──> ACQUIRE ──(đủ N lần thấy)──> TRACK ──(mất liên tiếp / quá hạn)──> LOST
                       ^                                                                 │
                       └───────────────(tìm lại được bằng định danh / ngoại hình)────────┘
    cancel() ở bất kỳ trạng thái nào -> IDLE

  * ACQUIRE : quét toàn khung 2-3 lượt để xác nhận mục tiêu; camera ảo zoom DẦN vào trong lúc quét.
  * TRACK   : theo dõi. Chế độ "eco" dùng YOLO nhỏ (320px) quét TOÀN KHUNG (không dùng ROI),
              tần suất thích ứng theo tốc độ. Không chạy ByteTrack. Không phát hiện chuyển động.
              Chế độ "full" quét cả khung bằng model chính + ByteTrack.
  * LOST    : mục tiêu bị che/rời đi -> camera zoom xa ra dần, hệ thống về trạng thái bình thường
              (phát hiện chuyển động + quét thăm dò thưa) cho tới khi tìm lại được hoặc người dùng huỷ.

Chuyển động (từ điểm tâm box YOLO):
  * Bộ lọc Kalman gia tốc không đổi (x, y, vận tốc, gia tốc) cho tâm mục tiêu, chạy theo nhịp AI.
  * Speed damping: vận tốc được nhân hệ số suy giảm tỉ lệ nghịch với tốc độ — chuyển động nhỏ tắt
    rất nhanh, chuyển động lớn giữ nguyên. Tránh trôi camera khi người đứng yên mà box rung nhỏ.
  * Camera ảo chạy theo nhịp giao diện (~60 lần/giây): lò xo giảm chấn tới hạn bám theo vị trí
    ngoại suy, feed-forward theo vận tốc/gia tốc TƯƠNG ĐỐI.

Deadzone & Stable Center (chống dao động khi cắt khung):
  * DEADZONE_RATIO : bán kính vùng chết tính theo chiều cao người. Bên trong vùng này không cập nhật
                     stable_center -> viewport cắt không thay đổi khi box rung nhỏ.
  * COMMIT_RATIO   : tâm box mới phải cách stable_center ít nhất khoảng này mới cập nhật
                     (đảm bảo người thực sự di chuyển, không phải dao động detection).

Lớp này KHÔNG phụ thuộc Qt hay OpenVINO; chỉ cần numpy + cv2.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from BusinessLayer.Inference.tracker import Track
from DataLayer.vector_db import UNKNOWN

MODE_ECO = "eco"      # tiết kiệm: dùng YOLO 320 toàn khung, không ByteTrack, màn chính tự phóng to
MODE_FULL = "full"    # toàn bộ: quét cả khung + ByteTrack, mục tiêu hiện ở ô riêng bên cạnh


class FocusState:
    IDLE = "idle"
    ACQUIRE = "acquire"
    TRACK = "track"
    LOST = "lost"


@dataclass
class Cand:
    """Một ứng viên mà lượt quét tìm được (từ tracker hoặc từ phát hiện thô trong ROI)."""
    box: np.ndarray                    # [x1, y1, x2, y2] toạ độ ảnh gốc
    score: float = 0.0
    track_id: int | None = None
    identity: str | None = None        # tên | "Unknown" | None (chưa biết)


class FocusTrack(Track):
    """Track "giả" do FocusController sở hữu: dùng lại logic nhận diện mặt/phát sự kiện của pipeline
    khi chế độ tiết kiệm không chạy ByteTrack."""

    def __init__(self, track_id: int, box, score: float):
        super().__init__(track_id, box, score)
        self.box = np.asarray(box, dtype=np.float64)

    @property
    def tlbr(self) -> np.ndarray:
        return self.box


# ====================================================================== ước lượng chuyển động
class MotionEstimator:
    """Kalman gia tốc không đổi cho (cx, cy). Làm việc trong đơn vị chuẩn hoá = chiều cao khung hình.

    Đặc điểm thêm so với bộ lọc Kalman thuần tuý:
      - Speed damping: sau mỗi lần update, vận tốc được nhân hệ số <= 1 tỉ lệ nghịch với tốc độ.
        Chuyển động rất nhỏ -> hệ số gần 0, vận tốc tắt ngay; chuyển động lớn -> hệ số ~1, giữ nguyên.
      - Tham số DAMP_SPEED_THRESHOLD: ngưỡng tốc độ (đơn vị chiều-cao-người/s) dưới đó bắt đầu damp.
      - Tham số DAMP_MIN_FACTOR    : hệ số suy giảm tối thiểu khi gần như đứng yên.
    """

    MAX_EXTRAP = 0.35        # giây: không ngoại suy xa hơn (dữ liệu cũ thì không tin)
    A_MAX = 4.0              # chiều-cao-khung / s²

    # Speed damping: tốc độ < DAMP_SPEED_THRESHOLD * person_height/s thì damp mạnh
    DAMP_SPEED_THRESHOLD = 0.08   # chiều-cao-người / giây (chuẩn hoá theo scale)
    DAMP_MIN_FACTOR = 0.10        # hệ số giữ lại tối thiểu khi gần đứng yên

    def __init__(self, scale: float, jerk_std: float = 8.0, meas_std: float = 0.012):
        self.scale = max(float(scale), 1.0)
        self.q = float(jerk_std) ** 2
        self.r = float(meas_std) ** 2
        self.x = np.zeros((3, 2))
        self.P = np.diag([self.r, 0.5, 5.0])
        self._person_height_norm = 0.3   # chiều cao người / scale, cập nhật từ bên ngoài

    def reset(self, pos_px) -> None:
        self.x = np.zeros((3, 2))
        self.x[0] = np.asarray(pos_px, dtype=np.float64) / self.scale
        self.P = np.diag([self.r, 0.5, 5.0])

    def set_person_height(self, height_px: float) -> None:
        """Cho bộ lọc biết chiều cao người (px) để tính ngưỡng speed damping."""
        self._person_height_norm = max(height_px, 1.0) / self.scale

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        dt = min(dt, 1.0)
        F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], dtype=np.float64)
        Q = self.q * np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6, dt ** 2 / 2, dt]], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, pos_px) -> None:
        z = np.asarray(pos_px, dtype=np.float64) / self.scale
        S = self.P[0, 0] + self.r
        K = self.P[:, 0] / S
        self.x = self.x + np.outer(K, z - self.x[0])
        P = self.P - np.outer(K, self.P[0, :])
        self.P = (P + P.T) / 2
        # --- speed damping: giảm nhanh vận tốc khi chuyển động nhỏ ---
        speed_norm = float(np.hypot(*self.x[1]))          # đơn vị: 1/s (đã chuẩn hoá)
        threshold = self.DAMP_SPEED_THRESHOLD * self._person_height_norm
        if threshold > 0 and speed_norm < threshold:
            # factor = 0 khi speed=0, factor = 1 khi speed = threshold
            factor = self.DAMP_MIN_FACTOR + (1.0 - self.DAMP_MIN_FACTOR) * (speed_norm / threshold)
            self.x[1] *= factor
            self.x[2] *= factor   # giảm gia tốc cùng lúc để tránh tích luỹ

    # ---- đại lượng đọc ra (đơn vị pixel)
    @property
    def pos(self) -> np.ndarray:
        return self.x[0] * self.scale

    @property
    def vel(self) -> np.ndarray:
        return self.x[1] * self.scale

    @property
    def acc(self) -> np.ndarray:
        return np.clip(self.x[2], -self.A_MAX, self.A_MAX) * self.scale

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.vel))

    @property
    def acc_mag(self) -> float:
        return float(np.hypot(*self.acc))

    def extrapolate(self, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 0.0, self.MAX_EXTRAP))
        return self.pos + self.vel * dt + 0.5 * self.acc * dt * dt

    def velocity_at(self, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 0.0, self.MAX_EXTRAP))
        return self.vel + self.acc * dt


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _smoothstep(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


# ====================================================================== controller
class FocusController:
    # --- hằng số deadzone & stable-center ---
    DEADZONE_RATIO = 0.04     # bán kính vùng chết (tỉ lệ chiều cao người). Dao động nhỏ hơn: bỏ qua.
    COMMIT_RATIO = 0.10       # tâm mới phải xa stable_center ít nhất (tỉ lệ chiều cao người) để cập nhật.

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.acquire_frames = max(2, int(c.get("acquire_frames", 3)))
        self.full_fps = float(c.get("full_fps", 12))
        self.eco_fps_min = float(c.get("eco_fps_min", 6))
        self.eco_fps_max = float(c.get("eco_fps_max", 15))
        self.min_margin = float(c.get("min_margin", 0.30))
        self.speed_gain = float(c.get("speed_gain", 1.5))
        self.expand = float(c.get("expand_on_miss", 1.6))
        self.min_roi = float(c.get("min_roi", 80))
        self.lost_misses = int(c.get("lost_misses", 4))
        self.lost_timeout = float(c.get("lost_timeout", 1.5))
        self.zoom_margin = float(c.get("zoom_margin", 1.8))
        self.max_zoom = float(c.get("max_zoom", 4.0))
        self.smooth_time = float(c.get("smooth_time", 0.25))
        self.zoom_smooth_time = float(c.get("zoom_smooth_time", 0.5))
        self.reid_threshold = float(c.get("reid_threshold", 0.72))
        # --- tần suất quét mặt khi focus (chậm hơn chế độ bình thường) ---
        self.face_interval_pending_focus = float(c.get("face_interval_pending_focus", 1.0))
        self.face_interval_unknown_focus = float(c.get("face_interval_unknown_focus", 3.0))

        self._lock = threading.RLock()
        self.state = FocusState.IDLE
        self.mode = MODE_ECO
        self.ftrack: FocusTrack | None = None
        self.track_id: int | None = None
        self._clear()

    # ------------------------------------------------------------------ vòng đời
    def _clear(self) -> None:
        self.est: MotionEstimator | None = None
        self.box = np.zeros(4)
        self.size = (1.0, 1.0)
        self.sig = None
        self.t_obs = self.last_seen = 0.0
        self.acq_hits = 0
        self.misses = 0
        self.last_disp = 0.0
        self.fw, self.fh = 640, 480
        self.cam_c = np.array([320.0, 240.0])
        self.cam_v = np.zeros(2)
        self.zoom_h = 480.0
        self.zoom_v = 0.0
        self._t_cam = 0.0
        # --- stable center (cho cắt khung, chống dao động) ---
        self.stable_center: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.state != FocusState.IDLE

    def start(self, box, track_id: int, score: float, frame: np.ndarray, mode: str, now: float,
              src_track: Track | None = None) -> None:
        with self._lock:
            self._clear()
            self.fh, self.fw = frame.shape[:2]
            self.mode = mode if mode in (MODE_ECO, MODE_FULL) else MODE_ECO
            self.track_id = track_id
            self.ftrack = FocusTrack(track_id, box, score)
            if src_track is not None:                      # thừa hưởng định danh đã nhận diện
                self.ftrack.identity = src_track.identity
                self.ftrack.identity_score = src_track.identity_score
                self.ftrack.known_votes = dict(src_track.known_votes)
                self.ftrack.unknown_votes = src_track.unknown_votes
            b = np.asarray(box, dtype=np.float64)
            self.box = b
            self.size = (max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0))
            self.est = MotionEstimator(self.fh)
            self.est.set_person_height(self.size[1])
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            self.est.reset((cx, cy))
            self.stable_center = np.array([cx, cy])
            self.sig = self._signature(frame, b)
            self.t_obs = self.last_seen = now
            self.acq_hits = 1                              # lần xác định ban đầu (chính là lần nhấp chọn)
            self.state = FocusState.ACQUIRE
            self.cam_c = np.array([self.fw / 2, self.fh / 2])
            self.zoom_h = float(self.fh)
            self._t_cam = now

    def cancel(self) -> None:
        with self._lock:
            self.state = FocusState.IDLE
            self.ftrack = None
            self.track_id = None
            self._clear()

    def set_mode(self, mode: str) -> None:
        with self._lock:
            if mode in (MODE_ECO, MODE_FULL):
                self.mode = mode
                self.misses = 0
                self.track_id = None      # đổi chế độ => tracker bị reset, id cũ không còn giá trị

    @property
    def label(self) -> str:
        ft = self.ftrack
        if ft is None:
            return ""
        if ft.identity is None:
            return f"Người #{ft.track_id}"
        return "Người lạ" if ft.identity == UNKNOWN else str(ft.identity)

    # ------------------------------------------------------------------ lập lịch quét
    def scan_interval(self) -> float:
        """Khoảng cách giữa 2 lượt quét: eco thích ứng theo tốc độ, còn lại cố định."""
        with self._lock:
            if self.state != FocusState.TRACK or self.mode != MODE_ECO or self.est is None:
                return 1.0 / self.full_fps
            rel = self.est.speed / max(self.size[1], 1.0)          # chiều-cao-người / giây
            k = min(1.0, rel / 1.0)
            return 1.0 / (self.eco_fps_min + (self.eco_fps_max - self.eco_fps_min) * k)

    def scan_roi(self, frame_shape, now: float):
        """Không còn dùng trong chế độ eco (eco giờ quét toàn khung bằng YOLO 320).
        Giữ lại để tương thích với các lời gọi cũ; luôn trả về None."""
        return None

    # ------------------------------------------------------------------ ngoại hình
    @staticmethod
    def _signature(frame: np.ndarray, box):
        """Biểu đồ màu HSV của thân trên và thân dưới (rẻ, đủ để phân biệt người mặc đồ khác nhau)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 16:
            return None
        xa, xb = int(max(0, x1 + 0.15 * bw)), int(min(w, x2 - 0.15 * bw))
        sig = []
        for fa, fb in ((0.12, 0.45), (0.50, 0.90)):
            ya, yb = int(max(0, y1 + fa * bh)), int(min(h, y1 + fb * bh))
            crop = frame[ya:yb, xa:xb]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                return None
            crop = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
            sig.append(hist.astype(np.float32))
        return sig

    @staticmethod
    def _sim(a, b) -> float:
        if a is None or b is None:
            return 0.5
        d = [cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b)]
        return float(1.0 - sum(d) / len(d))

    # ------------------------------------------------------------------ ghép mục tiêu
    def _predict_box(self, now: float) -> np.ndarray:
        cx, cy = self.est.extrapolate(now - self.t_obs)
        w, h = self.size
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def _associate(self, cands: list[Cand], frame: np.ndarray, now: float, strict: bool):
        if not cands:
            return None
        if self.track_id is not None:
            for c in cands:
                if c.track_id is not None and c.track_id == self.track_id:
                    return c
        pred = self._predict_box(now)
        ph = max(pred[3] - pred[1], 1.0)
        pcx, pcy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        my_id = self.ftrack.identity if self.ftrack is not None else None
        best, best_score = None, -1.0
        for c in cands:
            app = self._sim(self.sig, self._signature(frame, c.box))
            ident = 1.0 if (my_id not in (None, UNKNOWN) and c.identity == my_id) else 0.0
            if strict:      # tìm lại sau khi mất: vị trí không còn đáng tin, dựa vào định danh + ngoại hình
                ok = (ident and app >= 0.5) or app >= self.reid_threshold
                score = app + 0.3 * ident
            else:
                iou = _iou(pred, c.box)
                ccx, ccy = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
                d = math.hypot(ccx - pcx, ccy - pcy) / ph
                pos = 0.6 * iou + 0.4 * max(0.0, 1.0 - d / 2.0)
                score = 0.55 * pos + 0.35 * app + 0.10 * ident
                ok = score >= 0.38
            if ok and score > best_score:
                best, best_score = c, score
        return best

    # ------------------------------------------------------------------ nhận kết quả quét
    def observe(self, cands: list[Cand], frame: np.ndarray, now: float):
        """Gọi sau mỗi lượt quét (luồng AI). Trả về ứng viên được chọn làm mục tiêu hoặc None."""
        with self._lock:
            if self.state == FocusState.IDLE or self.est is None:
                return None
            self.fh, self.fw = frame.shape[:2]
            if self.state == FocusState.LOST:
                m = self._associate(cands, frame, now, strict=True)
                if m is not None:
                    self._relock(m, frame, now)
                return m

            m = self._associate(cands, frame, now, strict=False)
            if m is not None:
                self._accept(m, frame, now)
                if self.state == FocusState.ACQUIRE:
                    self.acq_hits += 1
                    if self.acq_hits >= self.acquire_frames:
                        self.state = FocusState.TRACK
                return m

            self.misses += 1
            if self.misses >= self.lost_misses or now - self.last_seen > self.lost_timeout:
                self._to_lost()
            return None

    def _accept(self, c: Cand, frame: np.ndarray, now: float) -> None:
        box = np.asarray(c.box, dtype=np.float64)
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        prev = self.est.pos.copy()
        self.est.predict(now - self.t_obs)
        self.est.set_person_height(max(box[3] - box[1], 1.0))
        self.est.update(center)             # speed damping áp dụng bên trong update()
        self.last_disp = float(np.hypot(*(center - prev)))
        a = 0.35
        self.size = (self.size[0] * (1 - a) + max(box[2] - box[0], 1.0) * a,
                     self.size[1] * (1 - a) + max(box[3] - box[1], 1.0) * a)
        self.box = box
        self.t_obs = self.last_seen = now
        self.misses = 0
        ft = self.ftrack
        ft.box, ft.score = box, float(c.score)
        if c.track_id is not None:
            self.track_id = ft.track_id = c.track_id
        if c.identity and (c.identity != UNKNOWN or ft.identity is None):
            ft.identity = c.identity
        new_sig = self._signature(frame, box)               # cập nhật ngoại hình chậm (tránh học nhầm khi bị che)
        if new_sig is not None:
            self.sig = new_sig if self.sig is None else [0.9 * s + 0.1 * n for s, n in zip(self.sig, new_sig)]
        # --- cập nhật stable_center chỉ khi tâm dịch đủ xa (chống dao động khi đứng im) ---
        person_h = self.size[1]
        deadzone = self.DEADZONE_RATIO * person_h
        commit_dist = self.COMMIT_RATIO * person_h
        if self.stable_center is None:
            self.stable_center = center.copy()
        else:
            dist = float(np.hypot(*(center - self.stable_center)))
            if dist > commit_dist:
                # Di chuyển stable_center đến vị trí tâm mới nhưng giữ lề deadzone:
                # dịch vào gần center thêm một đoạn = dist - deadzone, theo hướng center
                move = max(0.0, dist - deadzone)
                if dist > 0:
                    self.stable_center = self.stable_center + (center - self.stable_center) * (move / dist)

    def _to_lost(self) -> None:
        self.state = FocusState.LOST
        self.track_id = None          # id cũ không còn giá trị (tracker có thể đã reset)
        self.misses = 0

    def _relock(self, c: Cand, frame: np.ndarray, now: float) -> None:
        box = np.asarray(c.box, dtype=np.float64)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.est.reset(center)
        self.size = (max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0))
        self.est.set_person_height(self.size[1])
        self.box = box
        self.t_obs = self.last_seen = now
        self.misses = 0
        self.last_disp = 0.0
        self.acq_hits = 1
        self.stable_center = np.array(list(center))
        self.state = FocusState.ACQUIRE
        ft = self.ftrack
        ft.box = box
        if c.track_id is not None:
            self.track_id = ft.track_id = c.track_id
        else:
            self.track_id = ft.track_id
        if c.identity and (c.identity != UNKNOWN or ft.identity is None):
            ft.identity = c.identity
        new_sig = self._signature(frame, box)
        if new_sig is not None:
            self.sig = new_sig if self.sig is None else [0.7 * s + 0.3 * n for s, n in zip(self.sig, new_sig)]

    # ------------------------------------------------------------------ camera ảo (chạy theo nhịp giao diện)
    def _want_zoom_h(self) -> float:
        bh = self.size[1]
        rel = self.est.speed / max(bh, 1.0)
        want = bh * self.zoom_margin * (1.0 + min(0.6, 0.25 * rel))   # chạy nhanh thì lùi ra một chút
        return float(min(self.fh, max(self.fh / self.max_zoom, want)))

    def _vp_size(self, aspect: float) -> tuple[float, float]:
        vh = min(max(self.zoom_h, 1.0), float(self.fh))
        vw = vh * aspect
        if vw > self.fw:
            vw = float(self.fw)
            vh = vw / aspect
        return vw, vh

    def step(self, now: float, aspect: float | None = None) -> None:
        """Tiến camera ảo tới thời điểm `now`. Gọi mỗi lần giao diện làm mới."""
        with self._lock:
            if self.state == FocusState.IDLE or self.est is None:
                return
            dt = now - self._t_cam
            self._t_cam = now
            if dt <= 0:
                return
            dt = min(dt, 0.1)
            fw, fh = float(self.fw), float(self.fh)
            aspect = aspect if aspect and aspect > 0 else fw / fh

            if self.state == FocusState.LOST:
                ref_c, ref_v, ref_a = np.array([fw / 2, fh / 2]), np.zeros(2), np.zeros(2)
                ref_zoom = fh
            else:
                et = now - self.t_obs
                ref_c = self.est.extrapolate(et)
                ref_v = self.est.velocity_at(et)
                ref_a = self.est.acc * math.exp(-et / 0.3)         # gia tốc phai dần khi dữ liệu cũ
                want = self._want_zoom_h()
                if self.state == FocusState.ACQUIRE:               # zoom dần theo số lần quét đã xác nhận
                    p = _smoothstep(self.acq_hits / self.acquire_frames)
                    ref_zoom = fh + (want - fh) * p
                else:
                    ref_zoom = want

            vw, vh = self._vp_size(aspect)
            ref_c = np.array([min(max(ref_c[0], vw / 2), fw - vw / 2),
                              min(max(ref_c[1], vh / 2), fh - vh / 2)])

            n = max(1, int(math.ceil(dt / 0.02)))
            h = dt / n
            w1 = 2.0 / max(self.smooth_time, 0.05)
            w2 = 2.0 / max(self.zoom_smooth_time, 0.05)
            dz = 0.03 * vh                                          # vùng chết nhỏ: bỏ qua rung lắc của box
            for _ in range(n):
                e = ref_c - self.cam_c
                e = e - np.clip(e, -dz, dz)
                acc = w1 * w1 * e + 2 * w1 * (ref_v - self.cam_v) + ref_a   # (ref_v - cam_v) = vận tốc tương đối
                self.cam_v = self.cam_v + acc * h
                self.cam_c = self.cam_c + self.cam_v * h
                za = w2 * w2 * (ref_zoom - self.zoom_h) - 2 * w2 * self.zoom_v
                self.zoom_v += za * h
                self.zoom_h += self.zoom_v * h
            lo = fh / self.max_zoom
            if self.zoom_h > fh:
                self.zoom_h, self.zoom_v = fh, min(self.zoom_v, 0.0)
            elif self.zoom_h < lo:
                self.zoom_h, self.zoom_v = lo, max(self.zoom_v, 0.0)

    @property
    def view_active(self) -> bool:
        """Có đang hiển thị khung nhìn phóng to không (LOST thì tới khi zoom xa xong)."""
        with self._lock:
            if self.state in (FocusState.ACQUIRE, FocusState.TRACK):
                return True
            if self.state == FocusState.LOST:
                return self.zoom_h < self.fh * 0.985
            return False

    def viewport(self, aspect: float | None = None):
        """Khung nhìn (x1,y1,x2,y2) theo toạ độ ảnh gốc, đúng tỉ lệ `aspect`, nằm trọn trong khung hình.

        Dùng stable_center (chống dao động) khi đang TRACK/ACQUIRE để tâm cắt không nhảy lung tung.
        """
        with self._lock:
            if not self.view_active:
                return None
            fw, fh = float(self.fw), float(self.fh)
            aspect = aspect if aspect and aspect > 0 else fw / fh
            vw, vh = self._vp_size(aspect)
            # Ưu tiên stable_center khi đang bám (tránh dao động nhỏ làm rung viewport)
            if self.state in (FocusState.ACQUIRE, FocusState.TRACK) and self.stable_center is not None:
                cx_ref = float(self.stable_center[0])
                cy_ref = float(self.stable_center[1])
                # Blending nhẹ cam_c vào để viewport vẫn mượt
                cx = cx_ref * 0.7 + float(self.cam_c[0]) * 0.3
                cy = cy_ref * 0.7 + float(self.cam_c[1]) * 0.3
            else:
                cx = float(self.cam_c[0])
                cy = float(self.cam_c[1])
            cx = min(max(cx, vw / 2), fw - vw / 2)
            cy = min(max(cy, vh / 2), fh - vh / 2)
            return (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)

    @property
    def zoom_factor(self) -> float:
        return self.fh / max(self.zoom_h, 1.0)