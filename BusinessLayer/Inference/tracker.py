"""ByteTrack gọn nhẹ (numpy + scipy), không phụ thuộc thư viện tracking bên ngoài.

Ý tưởng ByteTrack: ghép nối 2 vòng
  vòng 1: track hiện có  <-> box điểm CAO      (IoU)
  vòng 2: track chưa ghép <-> box điểm THẤP     (cứu các box bị che khuất nhẹ)
Track mới chỉ "chính thức" sau khi được ghép thêm 1 lần nữa (trừ khung đầu tiên) -> giảm track ma.
Track giữ thêm thông tin định danh (tên/người lạ) để nhận diện mặt không phải làm lại mỗi khung.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState:
    NEW, TRACKED, LOST, REMOVED = range(4)


# ---------------------------------------------------------------------- Kalman (x, y, a, h, vx, vy, va, vh)
class KalmanFilterXYAH:
    def __init__(self):
        ndim = 4
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = 1.0
        self._update = np.eye(ndim, 2 * ndim)
        self._wp, self._wv = 1.0 / 20, 1.0 / 160

    def initiate(self, m: np.ndarray):
        mean = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-2, self._wp * h,
               self._wv * h, self._wv * h, 1e-5, self._wv * h]
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def _project(self, mean, cov):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-1, self._wp * h]
        return self._update @ mean, self._update @ cov @ self._update.T + np.diag(np.square(std))

    def update(self, mean, cov, meas):
        pm, pc = self._project(mean, cov)
        gain = np.linalg.solve(pc, (cov @ self._update.T).T).T
        new_mean = mean + gain @ (meas - pm)
        new_cov = cov - gain @ pc @ gain.T
        return new_mean, new_cov


def _tlbr_to_xyah(b) -> np.ndarray:
    w, h = b[2] - b[0], max(b[3] - b[1], 1e-6)
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h], dtype=np.float64)


def _xyah_to_tlbr(m) -> np.ndarray:
    h = m[3]
    w = m[2] * h
    return np.array([m[0] - w / 2, m[1] - h / 2, m[0] + w / 2, m[1] + h / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _assign(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr, mc = {r for r, _ in matches}, {c for _, c in matches}
    return (matches,
            [i for i in range(cost.shape[0]) if i not in mr],
            [j for j in range(cost.shape[1]) if j not in mc])


# ---------------------------------------------------------------------- Track
class Track:
    def __init__(self, track_id: int, tlbr, score: float):
        self.track_id = track_id
        self.score = float(score)
        self._init_tlbr = np.asarray(tlbr, dtype=np.float64)
        self.mean = None
        self.cov = None
        self.state = TrackState.NEW
        self.is_activated = False
        self.frame_id = 0
        self.hits = 1
        # --- định danh khuôn mặt (do pipeline điền)
        self.identity: str | None = None      # None = chưa kết luận | tên | "Unknown"
        self.identity_score = 0.0
        self.known_votes: dict[str, int] = {}
        self.unknown_votes = 0
        self.face_box = None
        self.face_time = 0.0
        self.last_face_try = 0.0

    @property
    def tlbr(self) -> np.ndarray:
        return self._init_tlbr if self.mean is None else _xyah_to_tlbr(self.mean[:4])

    def activate(self, kf: KalmanFilterXYAH, frame_id: int) -> None:
        self.mean, self.cov = kf.initiate(_tlbr_to_xyah(self._init_tlbr))
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id

    def predict(self, kf: KalmanFilterXYAH) -> None:
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0.0
        self.mean, self.cov = kf.predict(mean, self.cov)

    def update(self, kf: KalmanFilterXYAH, tlbr, score: float, frame_id: int) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, _tlbr_to_xyah(tlbr))
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = float(score)
        self.hits += 1


# ---------------------------------------------------------------------- Tracker
class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, new_track_thresh=0.5,
                 match_thresh=0.8, track_buffer=30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.kf = KalmanFilterXYAH()
        self.reset()

    def reset(self) -> None:
        self.frame_id = 0
        self._next_id = 1
        self.tracked: list[Track] = []
        self.lost: list[Track] = []

    def update(self, dets: np.ndarray) -> list[Track]:
        """dets: (N,5) [x1,y1,x2,y2,score]. Trả về các track chính thức khớp được trong khung này."""
        self.frame_id += 1
        fid = self.frame_id
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        scores = dets[:, 4]
        high = dets[scores >= self.high_thresh]
        low = dets[(scores >= self.low_thresh) & (scores < self.high_thresh)]

        confirmed = [t for t in self.tracked if t.is_activated]
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        pool = confirmed + self.lost
        for t in pool:
            t.predict(self.kf)

        activated, refound, lost_now, removed = [], [], [], []

        # --- vòng 1: pool <-> box điểm cao
        pool_boxes = np.array([t.tlbr for t in pool]).reshape(-1, 4)
        matches, u_trk, u_det = _assign(1.0 - iou_matrix(pool_boxes, high[:, :4]), self.match_thresh)
        for ti, di in matches:
            t = pool[ti]
            was_tracked = t.state == TrackState.TRACKED
            t.update(self.kf, high[di, :4], high[di, 4], fid)
            (activated if was_tracked else refound).append(t)

        # --- vòng 2: track còn lại (đang TRACKED) <-> box điểm thấp
        rest = [pool[i] for i in u_trk if pool[i].state == TrackState.TRACKED]
        rest_boxes = np.array([t.tlbr for t in rest]).reshape(-1, 4)
        matches2, u_trk2, _ = _assign(1.0 - iou_matrix(rest_boxes, low[:, :4]), 0.5)
        for ti, di in matches2:
            rest[ti].update(self.kf, low[di, :4], low[di, 4], fid)
            activated.append(rest[ti])
        for i in u_trk2:
            rest[i].state = TrackState.LOST
            lost_now.append(rest[i])

        # --- track chưa chính thức <-> box cao chưa ghép
        remain = high[u_det]
        unc_boxes = np.array([t.tlbr for t in unconfirmed]).reshape(-1, 4)
        matches3, u_unc, u_det3 = _assign(1.0 - iou_matrix(unc_boxes, remain[:, :4]), 0.7)
        for ti, di in matches3:
            unconfirmed[ti].update(self.kf, remain[di, :4], remain[di, 4], fid)
            activated.append(unconfirmed[ti])
        for i in u_unc:
            unconfirmed[i].state = TrackState.REMOVED
            removed.append(unconfirmed[i])

        # --- khởi tạo track mới
        for di in u_det3:
            if remain[di, 4] < self.new_track_thresh:
                continue
            t = Track(self._next_id, remain[di, :4], remain[di, 4])
            self._next_id += 1
            t.activate(self.kf, fid)
            activated.append(t)

        # --- dọn track mất quá lâu
        for t in self.lost:
            if fid - t.frame_id > self.track_buffer:
                t.state = TrackState.REMOVED
                removed.append(t)

        removed_ids = {t.track_id for t in removed}
        tracked_now = [t for t in self.tracked if t.state == TrackState.TRACKED]
        seen = {t.track_id for t in tracked_now}
        for t in activated + refound:
            if t.track_id not in seen:
                tracked_now.append(t)
                seen.add(t.track_id)
        self.tracked = [t for t in tracked_now if t.track_id not in removed_ids]
        refound_ids = {t.track_id for t in self.tracked}
        lost = [t for t in self.lost if t.track_id not in refound_ids and t.track_id not in removed_ids]
        lost_ids = {t.track_id for t in lost}
        lost.extend(t for t in lost_now if t.track_id not in lost_ids and t.track_id not in removed_ids)
        self.lost = lost

        return [t for t in self.tracked if t.is_activated and t.frame_id == fid]
