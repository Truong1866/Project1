"""Pipeline hợp nhất: Camera/Video -> Phát hiện chuyển động (có lọc) -> YOLO người -> Khuôn mặt -> Sự kiện.

Mô hình luồng (mỗi nguồn có luồng đọc riêng trong InputManager):

    [Input threads] --frame mới nhất--> [Motion thread] --mở cổng--> hàng đợi --> [AI thread]
           |                                                                         |
           +--------------------------- UI đọc khung + kết quả (không chờ AI) -------+

  * Luồng chuyển động: rẻ (vài ms/khung), chạy cho mọi nguồn, KHÔNG BAO GIỜ bị AI chặn.
  * Luồng AI: 1 luồng duy nhất (1 iGPU) xử lý lần lượt các nguồn đang "mở cổng", tối đa 1 việc chờ / nguồn.
  * Cổng AI của mỗi nguồn: IDLE --(chuyển động hợp lệ liên tục)--> ACTIVE --(hết người một lúc)--> IDLE.
    Trong ACTIVE mà người đứng yên (không còn chuyển động), AI vẫn chạy ở tần suất thấp để theo dõi tiếp.

FOCUS (BusinessLayer/focus.py): khi một nguồn đang focus (ACQUIRE/TRACK), FocusController quyết định nhịp quét và vùng quét,
cổng chuyển động bị bỏ qua. Chế độ eco chỉ quét ROI quanh mục tiêu (không chạy ByteTrack, không nhận diện mặt người khác);
chế độ full vẫn quét toàn khung + ByteTrack như bình thường. Khi mất mục tiêu (LOST) nguồn trở về trạng thái bình thường
+ quét thăm dò thưa cho tới khi tìm lại được hoặc người dùng huỷ.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from BusinessLayer.focus import MODE_ECO, MODE_FULL, Cand, FocusController, FocusState
from BusinessLayer.Inference.tracker import ByteTracker, Track
from BusinessLayer.input_manager import InputManager, SourceKind, classify_source
from BusinessLayer.motion_detector import MotionDetector
from DataLayer.vector_db import UNKNOWN, FaceDatabase
from Utils.config import Config
from Utils.event_bus import PERSON_KNOWN, PERSON_UNKNOWN, EventBus
from Utils.logger import get_logger

log = get_logger("Pipeline")

IDLE, ACTIVE = "idle", "active"


@dataclass
class TrackView:
    track_id: int
    box: tuple
    label: str
    kind: str                 # known | unknown | pending
    score: float = 0.0
    face_box: tuple | None = None
    focused: bool = False     # đây là mục tiêu đang được focus


@dataclass
class SourceView:
    """Ảnh chụp trạng thái 1 nguồn để UI vẽ (UI chỉ đọc, không đụng vào luồng xử lý)."""
    source_id: str
    name: str
    kind: str
    status: str
    frame_id: int
    frame: np.ndarray | None
    fps: float
    progress: float
    gate: str
    ai_ms: float
    tracks: list = field(default_factory=list)
    motion_boxes: list = field(default_factory=list)
    motion_rejected: list = field(default_factory=list)
    # --- focus
    focus_state: str = "idle"          # idle | acquire | track | lost
    focus_mode: str = MODE_ECO
    focus_label: str = ""
    viewport: tuple | None = None      # (x1,y1,x2,y2) khung nhìn phóng to theo toạ độ ảnh gốc; None = toàn khung


class SourceContext:
    def __init__(self, sid: str, name: str, kind: str, inp: InputManager,
                 motion: MotionDetector, tracker: ByteTracker, focus: FocusController):
        self.id, self.name, self.kind = sid, name, kind
        self.input, self.motion, self.tracker, self.focus = inp, motion, tracker, focus
        # --- cổng AI
        self.gate = IDLE
        self.motion_streak = 0
        self.last_motion_time = 0.0
        self.last_ai_time = 0.0
        self.last_person_time = 0.0
        self.person_seen = False
        self.refractory_until = 0.0
        self.ai_pending = False
        self.motion_stale = False      # bộ phát hiện chuyển động đã bị bỏ qua một lúc (đang focus) -> cần reset
        # --- bộ đếm khung đã xử lý
        self.last_motion_fid = -1
        self.last_motion_ts = 0.0
        # --- dữ liệu cho UI (gán nguyên khối => đọc không cần khoá)
        self.tracks_view: list[TrackView] = []
        self.motion_boxes: list = []
        self.motion_rejected: list = []
        self.ai_ms = 0.0
        self.cooldown: dict = {}


class SmartVisionPipeline:
    def __init__(self, cfg: Config, db: FaceDatabase | None = None, bus: EventBus | None = None,
                 event_repo=None, engine_factory=None):
        self.cfg = cfg
        self.db = db or FaceDatabase(threshold=cfg.get("face.threshold", 0.45))
        self.bus = bus or EventBus()
        self.event_repo = event_repo
        self._engine_factory = engine_factory or self._default_engine_factory

        det = cfg.section("detection")
        self.ai_fps = float(det.get("ai_fps", 10))
        self.idle_ai_fps = float(det.get("idle_ai_fps", 2))
        self.no_person_timeout = float(det.get("no_person_timeout", 1.5))
        self.motion_hold = float(det.get("motion_hold", 1.0))
        self.person_conf = float(det.get("person_conf", 0.30))

        self.motion_cfg = cfg.section("motion")
        self.motion_fps = float(self.motion_cfg.get("fps", 15))
        self.motion_confirm = int(self.motion_cfg.get("confirm_frames", 3))
        self.tracker_cfg = cfg.section("tracker")
        self.face_cfg = cfg.section("face")
        self.event_cfg = cfg.section("events")

        self.focus_cfg = cfg.section("focus")
        mode = str(self.focus_cfg.get("default_mode", MODE_ECO))
        self.focus_mode = mode if mode in (MODE_ECO, MODE_FULL) else MODE_ECO
        self.lost_probe_fps = max(0.2, float(self.focus_cfg.get("lost_probe_fps", 2)))
        self._focus_sid: str | None = None

        self.engine = None
        self.ai_ready = threading.Event()
        self.ai_status = "Đang nạp mô hình AI..."
        self.ai_device = "-"

        self._sources: dict[str, SourceContext] = {}
        self._lock = threading.RLock()
        self._counter = 0
        self._running = False
        self._queue: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []

        if event_repo is not None:
            self.bus.subscribe(PERSON_KNOWN, self._persist_event)
            self.bus.subscribe(PERSON_UNKNOWN, self._persist_event)

    # ================================================================== vòng đời
    def _default_engine_factory(self):
        from BusinessLayer.Inference.inference_engine import InferenceEngine
        c = self.cfg
        fc = self.face_cfg
        roi_path = c.get("models.yolo_roi_path")
        return InferenceEngine(
            c.path("models.yolo_path"), c.path("models.face_dir"),
            device=c.get("models.device", "GPU"), imgsz=int(c.get("models.imgsz", 640)),
            person_conf=self.person_conf, person_iou=float(c.get("detection.person_iou", 0.45)),
            face_det_size=tuple(fc.get("det_size", (320, 320))), face_det_thresh=float(fc.get("det_thresh", 0.5)),
            head_ratio=float(fc.get("head_ratio", 0.6)), min_face_px=int(fc.get("min_face_px", 36)),
            face_device_type=str(fc.get("device_type", "GPU_FP16")),
            yolo_roi_path=c.path("models.yolo_roi_path") if roi_path else None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for target, name in ((self._motion_loop, "MotionLoop"), (self._ai_loop, "AILoop")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("Pipeline đã chạy (luồng chuyển động + luồng AI).")

    def stop(self) -> None:
        self._running = False
        with self._lock:
            ctxs = list(self._sources.values())
            self._sources.clear()
        for ctx in ctxs:
            ctx.input.stop(wait=False)
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    # ================================================================== quản lý nguồn
    def add_source(self, source, kind: str | None = None, name: str | None = None, loop: bool = False) -> str:
        kind = kind or classify_source(source)
        with self._lock:
            self._counter += 1
            sid = f"src{self._counter}"
        if not name:
            name = Path(str(source)).name if kind == SourceKind.FILE else f"{kind} {self._counter}"
        mc = self.motion_cfg
        motion = MotionDetector(
            width=mc.get("width", 320), min_area_ratio=mc.get("min_area_ratio", 0.003),
            min_solidity=mc.get("min_solidity", 0.30), max_aspect=mc.get("max_aspect", 4.0),
            global_change_ratio=mc.get("global_change_ratio", 0.45), history=mc.get("history", 300),
            var_threshold=mc.get("var_threshold", 32), warmup_frames=mc.get("warmup_frames", 20),
            clutter_limit=mc.get("clutter_limit", 0.6), clutter_gain=mc.get("clutter_gain", 0.5),
            clutter_decay=mc.get("clutter_decay", 0.9995))
        tc = self.tracker_cfg
        tracker = ByteTracker(tc.get("high_thresh", 0.5), tc.get("low_thresh", 0.1),
                              tc.get("new_track_thresh", 0.5), tc.get("match_thresh", 0.8),
                              tc.get("track_buffer", 30))
        inp = InputManager(source, kind=kind, name=name, loop=loop)
        ctx = SourceContext(sid, name, kind, inp, motion, tracker, FocusController(self.focus_cfg))
        with self._lock:
            self._sources[sid] = ctx
        inp.start()
        return sid

    def remove_source(self, sid: str) -> None:
        with self._lock:
            ctx = self._sources.pop(sid, None)
            if sid == self._focus_sid:
                self._focus_sid = None
        if ctx:
            ctx.focus.cancel()
            ctx.input.stop(wait=False)
            log.info("Đã gỡ nguồn %s (%s)", sid, ctx.name)

    def list_sources(self) -> list[tuple[str, str, str]]:
        with self._lock:
            return [(c.id, c.name, c.kind) for c in self._sources.values()]

    def get_view(self, sid: str, aspect: float | None = None) -> SourceView | None:
        """aspect: tỉ lệ rộng/cao của widget sẽ hiển thị khung nhìn focus (camera ảo cần để tính viewport)."""
        ctx = self._sources.get(sid)
        if ctx is None:
            return None
        fid, frame = ctx.input.get_frame()
        fc = ctx.focus
        viewport, fstate, flabel = None, "idle", ""
        if fc.active:
            fc.step(time.perf_counter(), aspect)         # camera ảo chạy theo nhịp UI -> chuyển động mượt
            viewport = fc.viewport(aspect)
            fstate, flabel = fc.state, fc.label
        return SourceView(
            source_id=sid, name=ctx.name, kind=ctx.kind, status=ctx.input.status, frame_id=fid, frame=frame,
            fps=ctx.input.measured_fps, progress=ctx.input.progress, gate=ctx.gate, ai_ms=ctx.ai_ms,
            tracks=ctx.tracks_view, motion_boxes=ctx.motion_boxes, motion_rejected=ctx.motion_rejected,
            focus_state=fstate, focus_mode=fc.mode, focus_label=flabel, viewport=viewport)

    # ================================================================== cài đặt trực tiếp
    def apply_settings(self, **kw) -> None:
        if "ai_fps" in kw:
            self.ai_fps = max(1.0, float(kw["ai_fps"]))
        if "min_area_ratio" in kw:
            with self._lock:
                for c in self._sources.values():
                    c.motion.min_area_ratio = float(kw["min_area_ratio"])
            self.motion_cfg["min_area_ratio"] = float(kw["min_area_ratio"])
        if "person_conf" in kw:
            self.person_conf = float(kw["person_conf"])
            if self.engine is not None:
                self.engine.set_person_conf(self.person_conf)
        if "face_threshold" in kw:
            self.db.threshold = float(kw["face_threshold"])
        # --- focus
        if kw.get("focus_mode") in (MODE_ECO, MODE_FULL):
            mode = kw["focus_mode"]
            self.focus_mode = mode
            with self._lock:
                ctxs = list(self._sources.values())
            for c in ctxs:
                if c.focus.active and c.focus.mode != mode:
                    c.focus.set_mode(mode)
                    c.tracker.reset()                 # eco không cập nhật tracker -> bắt đầu lại cho sạch
                    c.motion_stale = True
                    c.last_ai_time = 0.0
        if "focus_zoom" in kw:
            v = float(kw["focus_zoom"])
            self.focus_cfg["zoom_margin"] = v
            with self._lock:
                for c in self._sources.values():
                    c.focus.zoom_margin = v
        if "focus_smooth" in kw:
            v = max(0.05, float(kw["focus_smooth"]))
            self.focus_cfg["smooth_time"], self.focus_cfg["zoom_smooth_time"] = v, v * 2
            with self._lock:
                for c in self._sources.values():
                    c.focus.smooth_time, c.focus.zoom_smooth_time = v, v * 2

    def register_face(self, sid: str, name: str) -> tuple[bool, str]:
        ctx = self._sources.get(sid)
        if ctx is None:
            return False, "Chưa chọn ô video nào."
        if not self.ai_ready.is_set():
            return False, "Mô hình AI chưa sẵn sàng."
        _, frame = ctx.input.get_frame()
        if frame is None:
            return False, "Nguồn chưa có khung hình."
        face = self.engine.largest_face(frame)
        if face is None:
            return False, "Không thấy khuôn mặt rõ nét trong khung hình."
        self.db.add_person(name, face["embedding"])
        return True, f"Đã lưu khuôn mặt của {name}."

    # ================================================================== FOCUS: API cho giao diện
    def list_targets(self) -> list[dict]:
        """Các mục tiêu (người) đang có trên màn hình, mọi nguồn."""
        with self._lock:
            ctxs = list(self._sources.values())
        out = []
        for ctx in ctxs:
            fc = ctx.focus
            fid = fc.ftrack.track_id if (fc.active and fc.ftrack is not None and fc.state != FocusState.LOST) else None
            for t in list(ctx.tracks_view):
                out.append({"sid": ctx.id, "source_name": ctx.name, "track_id": t.track_id, "label": t.label,
                            "kind": t.kind, "score": t.score, "focused": fid is not None and t.track_id == fid})
        return out

    def focus_info(self) -> dict | None:
        sid = self._focus_sid
        ctx = self._sources.get(sid) if sid else None
        if ctx is None or not ctx.focus.active:
            return None
        fc = ctx.focus
        return {"sid": sid, "mode": fc.mode, "state": fc.state, "label": fc.label, "zoom": fc.zoom_factor}

    def start_focus(self, sid: str, track_id: int) -> tuple[bool, str]:
        ctx = self._sources.get(sid)
        if ctx is None:
            return False, "Nguồn không còn tồn tại."
        if not self.ai_ready.is_set():
            return False, "Mô hình AI chưa sẵn sàng."
        tv = next((t for t in list(ctx.tracks_view) if t.track_id == track_id), None)
        if tv is None:
            return False, "Mục tiêu không còn trong khung hình."
        _, frame = ctx.input.get_frame()
        if frame is None:
            return False, "Nguồn chưa có khung hình."
        if self._focus_sid and self._focus_sid != sid:
            self.stop_focus()                                  # chỉ focus một mục tiêu tại một thời điểm
        src = next((t for t in list(ctx.tracker.tracked) if t.track_id == track_id), None)
        now = time.perf_counter()
        ctx.focus.start(tv.box, track_id, src.score if src is not None else 0.9, frame, self.focus_mode, now, src)
        ctx.gate = ACTIVE                                      # focus tự điều khiển nhịp quét, bỏ qua cổng chuyển động
        ctx.last_ai_time = 0.0
        self._focus_sid = sid
        return True, f"Đang focus: {ctx.focus.label}"

    def stop_focus(self) -> None:
        sid, self._focus_sid = self._focus_sid, None
        ctx = self._sources.get(sid) if sid else None
        if ctx is None:
            return
        ctx.focus.cancel()
        self._focus_to_normal(ctx, time.perf_counter())

    def _focus_to_normal(self, ctx: SourceContext, now: float) -> None:
        """Về trạng thái bình thường: tracker/motion làm lại từ đầu, cổng AI chạy tiếp tới khi hết người."""
        ctx.tracker.reset()
        ctx.tracks_view = []
        ctx.motion_stale = True
        ctx.gate = ACTIVE
        ctx.person_seen = True
        ctx.last_person_time = now
        ctx.last_ai_time = 0.0

    # ================================================================== luồng chuyển động
    def _motion_loop(self) -> None:
        interval = 1.0 / self.motion_fps
        while self._running:
            did_work = False
            now = time.perf_counter()
            for ctx in list(self._sources.values()):
                fc = ctx.focus
                focusing = fc.active and fc.state != FocusState.LOST
                fid, frame = ctx.input.get_frame()
                gap = 0.0 if focusing else interval           # focus: nhịp quét do FocusController quyết định
                if frame is None or fid == ctx.last_motion_fid or now - ctx.last_motion_ts < gap:
                    continue
                ctx.last_motion_fid, ctx.last_motion_ts = fid, now
                did_work = True
                try:
                    if focusing:
                        ctx.gate = ACTIVE                     # bỏ qua phát hiện chuyển động khi đang bám mục tiêu
                        ctx.motion_boxes, ctx.motion_rejected = [], []
                        ctx.motion_stale = True
                    else:
                        if ctx.motion_stale:
                            ctx.motion.reset()
                            ctx.motion_stale = False
                        res = ctx.motion.update(frame)
                        ctx.motion_boxes, ctx.motion_rejected = res.boxes, res.rejected
                        self._update_gate(ctx, res.valid, now)
                    if self._ai_due(ctx, now):
                        ctx.ai_pending = True
                        ctx.last_ai_time = now
                        roi = fc.scan_roi(frame.shape, now) if focusing else None
                        self._queue.put((ctx, frame, roi, now))
                except Exception:
                    log.exception("Lỗi luồng chuyển động (%s)", ctx.name)
            if not did_work:
                time.sleep(0.004)

    def _update_gate(self, ctx: SourceContext, valid: bool, now: float) -> None:
        if valid:
            ctx.motion_streak += 1
            ctx.last_motion_time = now
        else:
            ctx.motion_streak = 0
        if ctx.gate == IDLE and ctx.motion_streak >= self.motion_confirm and now >= ctx.refractory_until:
            ctx.gate = ACTIVE
            ctx.last_person_time = now
            ctx.person_seen = False
            ctx.last_ai_time = 0.0
            ctx.tracker.reset()
            ctx.motion.begin_window()

    def _ai_due(self, ctx: SourceContext, now: float) -> bool:
        if ctx.ai_pending or not self.ai_ready.is_set():
            return False
        fc = ctx.focus
        if fc.active:
            if fc.state != FocusState.LOST:                    # ACQUIRE/TRACK: nhịp quét do focus quyết định
                return now - ctx.last_ai_time >= fc.scan_interval()
            if ctx.gate != ACTIVE:                             # LOST + cổng đóng: quét thăm dò thưa để tìm lại mục tiêu
                return now - ctx.last_ai_time >= 1.0 / self.lost_probe_fps
        if ctx.gate != ACTIVE:
            return False
        moving = (now - ctx.last_motion_time) <= self.motion_hold
        fps = self.ai_fps if moving else self.idle_ai_fps
        return now - ctx.last_ai_time >= 1.0 / fps

    def _close_gate(self, ctx: SourceContext, now: float) -> None:
        if not ctx.person_seen:
            ctx.motion.report_false_alarm()   # chuyển động nhưng không có người -> học là nhiễu
            ctx.refractory_until = now + 0.5
        ctx.gate = IDLE
        ctx.motion_streak = 0
        ctx.tracker.reset()
        ctx.tracks_view = []

    # ================================================================== luồng AI
    def _ai_loop(self) -> None:
        try:
            self.engine = self._engine_factory()
            self.ai_device = f"YOLO {getattr(self.engine, 'device', '-')} | Mặt {getattr(self.engine, 'face_device', '-')}"
            self.ai_status = f"AI sẵn sàng ({self.ai_device})"
            self.ai_ready.set()
        except Exception as e:
            log.exception("Không nạp được mô hình AI")
            self.ai_status = f"Lỗi nạp mô hình: {e}"
            return

        while self._running:
            try:
                ctx, frame, roi, ts = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if ctx.id in self._sources:
                    self._run_ai(ctx, frame, roi, ts)
            except Exception:
                log.exception("Lỗi AI (%s)", ctx.name)
            finally:
                ctx.ai_pending = False

    def _run_ai(self, ctx: SourceContext, frame: np.ndarray, roi=None, ts: float | None = None) -> None:
        ts = ts if ts is not None else time.perf_counter()
        fc = ctx.focus
        if fc.active and fc.mode == MODE_ECO and fc.state == FocusState.TRACK:
            self._run_focus_eco(ctx, frame, roi, ts)
        else:
            self._run_normal_ai(ctx, frame, ts)

    # ------------------------------------------------------------------ eco: chỉ quét vùng quanh mục tiêu
    def _run_focus_eco(self, ctx: SourceContext, frame: np.ndarray, roi, ts: float) -> None:
        t0 = time.perf_counter()
        fc = ctx.focus
        if roi is not None:
            persons = self.engine.detect_persons_roi(frame, roi)    # chỉ quét ROI, không đụng phần còn lại của khung
        else:
            persons = self.engine.detect_persons(frame)
        cands = [Cand(np.asarray(p["box"], dtype=np.float64), float(p["confidence"])) for p in persons]
        match = fc.observe(cands, frame, ts)                         # ghép đúng mục tiêu, cập nhật Kalman + ROI kế tiếp
        now = time.perf_counter()
        ft = fc.ftrack
        if match is not None and ft is not None:
            ctx.person_seen = True
            ctx.last_person_time = now
            if self._need_face(ft, now):                             # nhận diện mặt chỉ cho mục tiêu (nếu chưa biết là ai)
                ft.last_face_try = now
                self._apply_face(ctx, ft, self.engine.process_faces(frame, ft.tlbr), frame)
            ctx.tracks_view = [self._to_view(ft, now, focused=True)]
        ctx.ai_ms = (time.perf_counter() - t0) * 1000
        if fc.state == FocusState.LOST:                              # mất mục tiêu -> về trạng thái bình thường
            log.info("[%s] Mất mục tiêu focus, quay về chế độ bình thường.", ctx.name)
            self._focus_to_normal(ctx, now)

    # ------------------------------------------------------------------ bình thường (và full / ACQUIRE / LOST)
    def _run_normal_ai(self, ctx: SourceContext, frame: np.ndarray, ts: float) -> None:
        t0 = time.perf_counter()
        now = time.perf_counter()

        persons = self.engine.detect_persons(frame)
        dets = np.array([[*p["box"], p["confidence"]] for p in persons], dtype=np.float32).reshape(-1, 5)
        tracks = ctx.tracker.update(dets)

        # --- nhận diện mặt: chỉ cho track còn cần, tối đa N mặt / lượt, ưu tiên track lâu chưa thử
        fc_face = self.face_cfg
        todo = [t for t in tracks if self._need_face(t, now)]
        todo.sort(key=lambda t: t.last_face_try)
        for t in todo[: int(fc_face.get("max_per_run", 2))]:
            t.last_face_try = now
            faces = self.engine.process_faces(frame, t.tlbr)
            self._apply_face(ctx, t, faces, frame)

        # --- focus: ghép mục tiêu trong số các track vừa quét (ACQUIRE / TRACK-full / LOST tìm lại)
        fc = ctx.focus
        focus_tid = None
        if fc.active:
            cands = [Cand(np.asarray(t.tlbr, dtype=np.float64).copy(), float(t.score), t.track_id, t.identity)
                     for t in tracks]
            if fc.observe(cands, frame, ts) is not None and fc.ftrack is not None:
                focus_tid = fc.ftrack.track_id

        if tracks:
            ctx.person_seen = True
            ctx.last_person_time = now
            ctx.motion.report_person([tuple(t.tlbr) for t in tracks])

        ctx.tracks_view = [self._to_view(t, now, focused=(focus_tid is not None and t.track_id == focus_tid))
                           for t in tracks]
        ctx.ai_ms = (time.perf_counter() - t0) * 1000

        focusing = fc.active and fc.state in (FocusState.ACQUIRE, FocusState.TRACK)
        if ctx.gate == ACTIVE and not focusing and not tracks and now - ctx.last_person_time > self.no_person_timeout:
            self._close_gate(ctx, now)

    # ------------------------------------------------------------------ nhận diện
    def _need_face(self, t: Track, now: float) -> bool:
        fc = self.face_cfg
        if t.identity not in (None, UNKNOWN):
            return False  # đã chốt là người quen
        gap = fc.get("interval_pending", 0.3) if t.identity is None else fc.get("interval_unknown", 1.5)
        return now - t.last_face_try >= gap

    def _apply_face(self, ctx: SourceContext, t: Track, faces: list, frame: np.ndarray) -> None:
        if not faces:
            return
        fc = self.face_cfg
        face = max(faces, key=lambda f: (f["abs_box"][2] - f["abs_box"][0]) * (f["abs_box"][3] - f["abs_box"][1]))
        name, score = self.db.identify(face["embedding"])
        t.face_box, t.face_time = tuple(face["abs_box"]), time.perf_counter()
        if name != UNKNOWN:
            t.known_votes[name] = t.known_votes.get(name, 0) + 1
            if t.known_votes[name] >= int(fc.get("confirm_known", 2)) and t.identity != name:
                t.identity, t.identity_score = name, score
                self._emit(ctx, t, PERSON_KNOWN, frame)
        else:
            t.unknown_votes += 1
            if t.identity is None and t.unknown_votes >= int(fc.get("confirm_unknown", 3)):
                t.identity, t.identity_score = UNKNOWN, score
                self._emit(ctx, t, PERSON_UNKNOWN, frame)

    @staticmethod
    def _to_view(t: Track, now: float, focused: bool = False) -> TrackView:
        box = tuple(int(v) for v in t.tlbr)
        face = t.face_box if (t.face_box and now - t.face_time < 1.0) else None
        if t.identity is None:
            return TrackView(t.track_id, box, f"Người #{t.track_id}", "pending", face_box=face, focused=focused)
        if t.identity == UNKNOWN:
            return TrackView(t.track_id, box, "NGƯỜI LẠ", "unknown", t.identity_score, face, focused)
        return TrackView(t.track_id, box, t.identity, "known", t.identity_score, face, focused)

    # ------------------------------------------------------------------ sự kiện
    def _emit(self, ctx: SourceContext, t: Track, kind: str, frame: np.ndarray) -> None:
        now = time.time()
        key = (kind, t.identity)
        if now - ctx.cooldown.get(key, 0) < float(self.event_cfg.get("cooldown_sec", 10)):
            return
        ctx.cooldown[key] = now
        snapshot = self._save_snapshot(ctx, t, kind, frame, now)
        self.bus.publish(kind, source_id=ctx.id, source_name=ctx.name, track_id=t.track_id,
                         name=t.identity, score=float(t.identity_score), snapshot=snapshot)

    def _save_snapshot(self, ctx, t, kind, frame, ts) -> str | None:
        if not self.event_cfg.get("save_snapshot", True):
            return None
        try:
            folder = self.cfg.path("events.snapshot_dir", "Data/snapshots")
            folder.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(v) for v in t.tlbr)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                return None
            path = folder / f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))}_{kind}_{ctx.id}_{t.track_id}.jpg"
            cv2.imwrite(str(path), crop)
            return str(path)
        except Exception:
            log.exception("Không lưu được snapshot")
            return None

    def _persist_event(self, ev: dict) -> None:
        self.event_repo.add(ev["ts"], ev["topic"], ev.get("source_id"), ev.get("source_name"),
                            ev.get("name"), ev.get("track_id"), ev.get("score"), ev.get("snapshot"))