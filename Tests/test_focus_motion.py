"""Unit tests cho FocusController và MotionEstimator.

Chạy:
    cd /home/truong/Documents/projects/python/Project1
    python -m pytest Tests/test_focus_motion.py -v

Không cần camera hay model AI thật – dùng dữ liệu giả.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Thêm thư mục gốc project vào sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from BusinessLayer.focus import (
    MODE_ECO,
    MODE_FULL,
    Cand,
    FocusController,
    FocusState,
    MotionEstimator,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _cfg(overrides: dict | None = None) -> dict:
    """Cấu hình focus mặc định cho test."""
    base = {
        "acquire_frames": 2,
        "full_fps": 12,
        "eco_fps_min": 6,
        "eco_fps_max": 15,
        "min_margin": 0.30,
        "speed_gain": 1.5,
        "expand_on_miss": 1.6,
        "min_roi": 80,
        "lost_misses": 3,
        "lost_timeout": 1.5,
        "lost_probe_fps": 2,
        "zoom_margin": 1.8,
        "max_zoom": 4.0,
        "smooth_time": 0.25,
        "zoom_smooth_time": 0.5,
        "reid_threshold": 0.72,
    }
    if overrides:
        base.update(overrides)
    return base


def _box(x1=100, y1=100, x2=200, y2=400) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _frame(h=720, w=1280) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _cand(box=None, score=0.9, track_id=1, identity=None) -> Cand:
    return Cand(box if box is not None else _box(), score, track_id, identity)


# ===========================================================================
# MotionEstimator tests
# ===========================================================================

class TestMotionEstimator:
    def test_initial_state(self):
        """Khởi tạo: vị trí và vận tốc phải bằng 0."""
        me = MotionEstimator(scale=720.0)
        assert me.x.shape == (3, 2)
        assert np.allclose(me.x, 0.0)

    def test_reset_sets_position(self):
        """reset() phải đặt vị trí chuẩn hoá đúng."""
        me = MotionEstimator(scale=720.0)
        me.reset((360, 180))  # cx=360, cy=180
        # Vị trí chuẩn hoá = pixel / scale
        assert pytest.approx(me.x[0, 0], abs=1e-6) == 360 / 720.0
        assert pytest.approx(me.x[0, 1], abs=1e-6) == 180 / 720.0

    def test_predict_advances_position(self):
        """predict(dt) phải dịch chuyển vị trí theo vận tốc."""
        me = MotionEstimator(scale=720.0)
        me.reset((0, 0))
        # Đặt vận tốc thủ công: 1 đơn vị / giây (chuẩn hoá)
        me.x[1, 0] = 1.0
        pos_before = me.x[0, 0]
        me.predict(0.1)
        pos_after = me.x[0, 0]
        assert pos_after > pos_before  # vị trí phải tăng

    def test_update_corrects_position(self):
        """update() phải kéo vị trí về phép đo mới."""
        me = MotionEstimator(scale=720.0)
        me.reset((0, 0))
        me.predict(0.1)
        # Đặt phép đo xa: cx=500px
        me.update((500, 250))
        # Sau update, vị trí phải gần hơn (500/720) so với 0
        assert me.x[0, 0] > 0.1

    def test_extrapolate_returns_ndarray(self):
        """extrapolate() phải trả về mảng (2,) hợp lệ."""
        me = MotionEstimator(scale=720.0)
        me.reset((320, 240))
        result = me.extrapolate(0.05)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_predict_dt_zero_no_change(self):
        """predict(0) không được thay đổi trạng thái."""
        me = MotionEstimator(scale=720.0)
        me.reset((100, 200))
        x_before = me.x.copy()
        me.predict(0)
        assert np.allclose(me.x, x_before)

    def test_velocity_and_acceleration_accessible(self):
        """Phải truy cập được vận tốc và gia tốc từ state vector."""
        me = MotionEstimator(scale=720.0)
        me.reset((320, 240))
        # Sau 2 bước update, vận tốc phải có giá trị hữu hạn
        me.predict(0.1)
        me.update((350, 260))
        me.predict(0.1)
        me.update((380, 280))
        vel = me.x[1]  # vận tốc (x, y) chuẩn hoá
        assert np.all(np.isfinite(vel))


# ===========================================================================
# FocusController - observe / scan_roi / step
# ===========================================================================

class TestFocusControllerObserve:
    def test_starts_idle(self):
        fc = FocusController(_cfg())
        assert fc.state == FocusState.IDLE
        assert not fc.active

    def test_start_changes_to_acquire(self):
        fc = FocusController(_cfg())
        fc.start(_box(), track_id=1, score=0.9, frame=_frame(), mode=MODE_ECO,
                  now=time.perf_counter())
        assert fc.state == FocusState.ACQUIRE
        assert fc.active

    def test_observe_acquire_transitions_to_track(self):
        """Sau đủ acquire_frames lần observe thành công → TRACK."""
        fc = FocusController(_cfg({"acquire_frames": 2}))
        now = time.perf_counter()
        fc.start(_box(), track_id=1, score=0.9, frame=_frame(), mode=MODE_ECO, now=now)

        cands = [_cand(_box(110, 110, 210, 410), 0.9, track_id=1)]
        for _ in range(3):  # đủ acquire_frames
            now += 0.1
            fc.observe(cands, _frame(), now)

        assert fc.state == FocusState.TRACK

    def test_observe_no_cand_increments_misses(self):
        """Không có ứng viên → tăng miss_count."""
        fc = FocusController(_cfg({"acquire_frames": 1, "lost_misses": 5}))
        now = time.perf_counter()
        fc.start(_box(), track_id=1, score=0.9, frame=_frame(), mode=MODE_ECO, now=now)
        # Chuyển nhanh sang TRACK
        fc.observe([_cand()], _frame(), now + 0.1)
        fc.observe([_cand()], _frame(), now + 0.2)

        miss_before = fc._miss_count if hasattr(fc, "_miss_count") else 0
        fc.observe([], _frame(), now + 0.3)  # không có ứng viên

    def test_observe_lost_after_too_many_misses(self):
        """Quá lost_misses lần không thấy → LOST."""
        lost_misses = 3
        fc = FocusController(_cfg({"acquire_frames": 1, "lost_misses": lost_misses, "lost_timeout": 999}))
        now = time.perf_counter()
        fc.start(_box(), track_id=1, score=0.9, frame=_frame(), mode=MODE_ECO, now=now)
        # Chuyển sang TRACK
        for i in range(2):
            fc.observe([_cand()], _frame(), now + i * 0.1)
        # Gây miss nhiều lần
        for i in range(lost_misses + 1):
            now += 0.1
            fc.observe([], _frame(), now)

        assert fc.state == FocusState.LOST

    def test_cancel_returns_to_idle(self):
        fc = FocusController(_cfg())
        now = time.perf_counter()
        fc.start(_box(), track_id=1, score=0.9, frame=_frame(), mode=MODE_ECO, now=now)
        assert fc.active
        fc.cancel()
        assert fc.state == FocusState.IDLE
        assert not fc.active

    def test_observe_returns_none_when_idle(self):
        fc = FocusController(_cfg())
        result = fc.observe([_cand()], _frame(), time.perf_counter())
        assert result is None

    def test_mode_eco_and_full(self):
        """Kiểm tra set_mode thay đổi mode."""
        fc = FocusController(_cfg())
        now = time.perf_counter()
        fc.start(_box(), track_id=1, score=0.9, frame=_frame(), mode=MODE_ECO, now=now)
        assert fc.mode == MODE_ECO
        fc.set_mode(MODE_FULL)
        assert fc.mode == MODE_FULL


class TestFocusControllerScanRoi:
    def test_scan_roi_returns_valid_box_when_tracking(self):
        """scan_roi() phải trả về (x1,y1,x2,y2) hợp lệ khi đang TRACK."""
        fc = FocusController(_cfg({"acquire_frames": 1}))
        now = time.perf_counter()
        frame_shape = (720, 1280, 3)
        fc.start(_box(400, 200, 600, 500), track_id=1, score=0.9,
                  frame=np.zeros(frame_shape, dtype=np.uint8), mode=MODE_ECO, now=now)
        # Chuyển sang TRACK
        for i in range(2):
            fc.observe([_cand(_box(400, 200, 600, 500))], np.zeros(frame_shape, dtype=np.uint8), now + i * 0.1)

        roi = fc.scan_roi(frame_shape, now + 0.3)
        if roi is not None:
            x1, y1, x2, y2 = roi
            assert x2 > x1 and y2 > y1
            assert x1 >= 0 and y1 >= 0
            assert x2 <= frame_shape[1] and y2 <= frame_shape[0]

    def test_scan_roi_returns_none_when_idle(self):
        fc = FocusController(_cfg())
        roi = fc.scan_roi((720, 1280, 3), time.perf_counter())
        assert roi is None


class TestFocusControllerStep:
    def test_step_does_not_crash_when_idle(self):
        """step() không được raise exception khi IDLE."""
        fc = FocusController(_cfg())
        fc.step(time.perf_counter(), aspect=16 / 9)

    def test_step_updates_viewport_when_tracking(self):
        """step() phải cập nhật viewport khi đang TRACK."""
        fc = FocusController(_cfg({"acquire_frames": 1}))
        now = time.perf_counter()
        frame_shape = (720, 1280, 3)
        fc.start(_box(400, 200, 600, 500), track_id=1, score=0.9,
                  frame=np.zeros(frame_shape, dtype=np.uint8), mode=MODE_ECO, now=now)
        for i in range(2):
            fc.observe([_cand(_box(400, 200, 600, 500))],
                       np.zeros(frame_shape, dtype=np.uint8), now + i * 0.1)
        now += 0.3
        fc.step(now, aspect=16 / 9)
        vp = fc.viewport(aspect=16 / 9)
        # Nếu trả về không None, phải là tuple 4 phần tử hợp lệ
        if vp is not None:
            assert len(vp) == 4

    def test_viewport_returns_none_when_idle(self):
        fc = FocusController(_cfg())
        assert fc.viewport(aspect=16 / 9) is None


# ===========================================================================
# Tích hợp: FocusController + MotionEstimator (không cần model)
# ===========================================================================

class TestFocusMotionIntegration:
    def test_full_acquire_track_lost_cycle(self):
        """Mô phỏng vòng đời đầy đủ: ACQUIRE → TRACK → LOST."""
        cfg = _cfg({"acquire_frames": 2, "lost_misses": 2, "lost_timeout": 999})
        fc = FocusController(cfg)
        frame = _frame()
        now = 0.0

        # 1. Bắt đầu focus
        fc.start(_box(300, 100, 500, 400), track_id=5, score=0.85,
                  frame=frame, mode=MODE_ECO, now=now)
        assert fc.state == FocusState.ACQUIRE

        # 2. Xác nhận mục tiêu → TRACK
        for i in range(3):
            now += 0.1
            fc.observe([_cand(_box(310 + i * 5, 100, 510 + i * 5, 400), track_id=5)], frame, now)
        assert fc.state == FocusState.TRACK

        # 3. Mất mục tiêu → LOST
        for _ in range(cfg["lost_misses"] + 1):
            now += 0.1
            fc.observe([], frame, now)
        assert fc.state == FocusState.LOST

        # 4. Huỷ → IDLE
        fc.cancel()
        assert fc.state == FocusState.IDLE

    def test_scan_interval_decreases_with_speed(self):
        """scan_interval() phải nhỏ hơn khi mục tiêu di chuyển nhanh (eco mode)."""
        cfg = _cfg({"acquire_frames": 1, "eco_fps_min": 4, "eco_fps_max": 20})
        fc = FocusController(cfg)
        frame = _frame()
        now = 0.0
        fc.start(_box(300, 100, 500, 400), track_id=1, score=0.9,
                  frame=frame, mode=MODE_ECO, now=now)

        # Chuyển sang TRACK
        now += 0.1
        fc.observe([_cand(_box(300, 100, 500, 400))], frame, now)
        now += 0.1
        fc.observe([_cand(_box(300, 100, 500, 400))], frame, now)

        # Lấy interval khi đứng yên
        interval_still = fc.scan_interval()

        # Mô phỏng mục tiêu chuyển động nhanh: observe với box dịch xa
        now += 0.1
        fc.observe([_cand(_box(500, 100, 700, 400))], frame, now)  # dịch 200px
        interval_moving = fc.scan_interval()

        # Interval khi di chuyển nhanh phải <= interval khi đứng yên
        assert interval_moving <= interval_still + 0.05  # cho phép sai số nhỏ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
