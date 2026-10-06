"""Cửa sổ chính. Mọi cài đặt nằm ở bảng nổi góc trên bên trái; vùng giữa là lưới video.

FOCUS:
  * Nút "🎯 Focus" trên thanh trên: nếu chỉ có 1 người trên màn hình -> focus ngay; nhiều người -> vào chế độ chọn
    (nhấp vào người trong video, hoặc chọn một dòng trong "Mục tiêu trên màn hình"). Bấm lại / phím Esc để huỷ.
  * Giữ Ctrl rồi nhấp vào người cũng focus ngay mà không cần bấm nút.
  * Chế độ focus (Tiết kiệm / Toàn bộ) đổi trong ⚙ Cài đặt > Focus.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter, QVBoxLayout, QWidget

from BusinessLayer.input_manager import SourceKind
from PresentLayer.Component.event_list import EventList
from PresentLayer.Component.settings_panel import MODE_FILE, MODE_STREAM, MODE_WEBCAM, SettingsPanel
from PresentLayer.Component.target_list import TargetList
from PresentLayer.Component.video_grid import VideoGrid

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".wmv"}
FOCUS_STATE_TEXT = {"acquire": "đang xác định mục tiêu...", "track": "đang bám theo",
                    "lost": "mất mục tiêu, đang tìm lại..."}

STYLE = """
QWidget { background:#0f1218; color:#e5e8ee; font-size:13px; }
QLabel, QRadioButton, QCheckBox { background:transparent; }
#TopBar { background:#151922; border-bottom:1px solid #262c38; }
QPushButton { background:#1d2330; border:1px solid #2b3342; border-radius:6px; padding:6px 12px; }
QPushButton:hover { background:#252c3b; }
QPushButton:checked { background:#25365a; border-color:#4f8cff; }
QPushButton:disabled { color:#5b6577; }
#FocusBtn:checked { background:#0c4a6e; border-color:#38bdf8; }
QToolButton { background:transparent; border:none; color:#9aa4b5; padding:4px 8px; border-radius:4px; }
QToolButton:hover { background:#252c3b; color:white; }
#SettingsPanel { background:#151922; border:1px solid #2f3748; border-radius:12px; }
#SettingsPanel QWidget { background:transparent; }
QLineEdit, QSpinBox, QDoubleSpinBox { background:#0f1218; border:1px solid #2b3342; border-radius:5px; padding:5px 7px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color:#4f8cff; }
QTabWidget::pane { border:1px solid #2b3342; border-radius:8px; top:-1px; }
QTabBar::tab { padding:7px 14px; background:transparent; color:#9aa4b5; }
QTabBar::tab:selected { color:white; border-bottom:2px solid #4f8cff; }
QListWidget, QTableWidget { background:#0f1218; border:1px solid #2b3342; border-radius:6px; }
QListWidget::item:selected, QTableWidget::item:selected { background:#25365a; }
QHeaderView::section { background:#151922; color:#9aa4b5; border:none; padding:5px; }
QSplitter::handle { background:#262c38; width:1px; height:1px; }
"""


class _Bridge(QObject):
    """Chuyển sự kiện từ luồng AI sang luồng giao diện (Qt tự xếp hàng tín hiệu liên luồng)."""
    event = Signal(dict)


class MainWindow(QMainWindow):
    def __init__(self, cfg, pipeline, bus, db, initial_sources: list[str] | None = None):
        super().__init__()
        self.cfg, self.pipeline, self.db = cfg, pipeline, db
        self._webcam_id: str | None = None
        self._armed = False                      # đang chờ người dùng chọn mục tiêu để focus
        self._focus_msg = ""

        self.setWindowTitle(cfg.get("app.title", "Smart Vision"))
        self.resize(1360, 800)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- thanh trên: nút cài đặt nằm góc trái, nút Focus ngay cạnh
        self.topbar = QFrame()
        self.topbar.setObjectName("TopBar")
        self.topbar.setFixedHeight(46)
        tb = QHBoxLayout(self.topbar)
        tb.setContentsMargins(10, 0, 12, 0)
        self.btn_settings = QPushButton("⚙  Cài đặt")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setChecked(True)
        self.btn_settings.toggled.connect(self._toggle_settings)
        self.btn_focus = QPushButton("🎯  Focus")
        self.btn_focus.setObjectName("FocusBtn")
        self.btn_focus.setCheckable(True)
        self.btn_focus.setToolTip("Bật focus vào một người (Esc để huỷ). Giữ Ctrl + nhấp vào người cũng focus ngay.")
        self.btn_focus.clicked.connect(self._on_focus_button)
        self.lbl_focus = QLabel("")
        self.lbl_focus.setStyleSheet("color:#7dd3fc;")
        self.lbl_ai = QLabel(pipeline.ai_status)
        self.lbl_ai.setStyleSheet("color:#9aa4b5;")
        self.btn_events = QPushButton("Sự kiện")
        self.btn_events.setCheckable(True)
        self.btn_events.setChecked(True)
        tb.addWidget(self.btn_settings)
        tb.addSpacing(6)
        tb.addWidget(self.btn_focus)
        tb.addSpacing(10)
        tb.addWidget(self.lbl_focus)
        tb.addStretch()
        tb.addWidget(self.lbl_ai)
        tb.addSpacing(12)
        tb.addWidget(self.btn_events)
        root.addWidget(self.topbar)

        # ---- thân: lưới video | (mục tiêu trên màn hình + danh sách sự kiện)
        self.grid = VideoGrid(pipeline, per_page=int(cfg.get("ui.tiles_per_page", 4)))
        self.targets = TargetList()
        self.events = EventList()
        self.side = QSplitter(Qt.Orientation.Vertical)
        self.side.addWidget(self.targets)
        self.side.addWidget(self.events)
        self.side.setSizes([300, 420])
        self.side.setChildrenCollapsible(False)
        self.side.setMinimumWidth(300)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.grid)
        split.addWidget(self.side)
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 340])
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        self.btn_events.toggled.connect(self.side.setVisible)

        # ---- bảng cài đặt nổi (góc trên trái)
        defaults = {
            "min_area_ratio": cfg.get("motion.min_area_ratio", 0.003),
            "ai_fps": cfg.get("detection.ai_fps", 10),
            "person_conf": cfg.get("detection.person_conf", 0.30),
            "face_threshold": cfg.get("face.threshold", 0.45),
            "focus_mode": pipeline.focus_mode,
            "focus_zoom": cfg.get("focus.zoom_margin", 1.8),
            "focus_smooth": cfg.get("focus.smooth_time", 0.25),
        }
        self.settings = SettingsPanel(defaults, parent=central)
        self.settings.set_people(db.list_people())
        self._wire_settings()
        self.grid.removeRequested.connect(self._remove_source)

        # ---- focus
        self.grid.targetPicked.connect(self._start_focus)
        self.grid.focusButtonClicked.connect(self._on_tile_focus_button)
        self.grid.focusCancelRequested.connect(self._cancel_focus)
        self.targets.focusRequested.connect(self._start_focus)
        self.targets.cancelRequested.connect(self._cancel_focus)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._cancel_focus)

        # ---- sự kiện AI -> UI
        self._bridge = _Bridge()
        self._bridge.event.connect(self.events.add_event)
        bus.subscribe("*", self._bridge.event.emit)

        # ---- vòng làm mới giao diện (độc lập với AI)
        self._tick_n = 0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(cfg.get("ui.refresh_ms", 16)))

        for src in initial_sources or []:
            self._add_files([src]) if Path(src).suffix.lower() in VIDEO_EXT else self._add_stream("", src)
        self._place_settings()

    # ================================================================== bố cục
    def _place_settings(self) -> None:
        self.settings.adjustSize()
        self.settings.move(12, self.topbar.height() + 8)
        self.settings.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_settings()

    def _toggle_settings(self, on: bool) -> None:
        self.settings.setVisible(on)
        if on:
            self._place_settings()

    def _tick(self) -> None:
        self._sync_focus_ui()                       # trước refresh: bố cục đã đúng trước khi vẽ
        self.grid.refresh()
        self._tick_n += 1
        if self._tick_n % 30 == 0:
            self.lbl_ai.setText(self.pipeline.ai_status)
        if self._tick_n % 8 == 0:
            self.targets.set_targets(self.pipeline.list_targets())

    # ================================================================== FOCUS
    def _on_focus_button(self, checked: bool) -> None:
        if self.pipeline.focus_info() is not None:          # đang focus -> bấm lại là huỷ
            self._cancel_focus()
            return
        if not checked:                                     # đang chờ chọn -> bỏ chờ
            self._arm(False)
            return
        targets = self.pipeline.list_targets()
        if len(targets) == 1:                               # chỉ có 1 người -> dùng luôn
            self._start_focus(targets[0]["sid"], targets[0]["track_id"])
        else:
            self._arm(True, "Nhấp vào người cần focus (hoặc chọn trong danh sách mục tiêu)"
                      if targets else "Chưa thấy ai trên màn hình, khi có người hãy nhấp vào họ để focus")

    def _on_tile_focus_button(self, sid: str) -> None:
        info = self.pipeline.focus_info()
        if info is not None and info["sid"] == sid:
            self._cancel_focus()
            return
        mine = [t for t in self.pipeline.list_targets() if t["sid"] == sid]
        if len(mine) == 1:
            self._start_focus(sid, mine[0]["track_id"])
        else:
            self._arm(True, "Nhấp vào người cần focus trong ô này" if mine
                      else "Chưa thấy ai trong ô này, khi có người hãy nhấp vào họ để focus")

    def _arm(self, on: bool, msg: str = "") -> None:
        self._armed = on
        self._focus_msg = msg if on else ""
        self.grid.set_pick_mode(on)
        self._sync_focus_ui(force=True)

    def _start_focus(self, sid: str, track_id: int) -> None:
        ok, msg = self.pipeline.start_focus(sid, int(track_id))
        self._arm(False)
        if not ok:
            self._focus_msg = msg
            self.lbl_focus.setText(msg)
            QTimer.singleShot(4000, self._clear_focus_msg)
        self._sync_focus_ui(force=True)

    def _clear_focus_msg(self) -> None:
        if not self._armed:
            self._focus_msg = ""

    def _cancel_focus(self) -> None:
        if self.pipeline.focus_info() is not None:
            self.pipeline.stop_focus()
        self._arm(False)

    def _sync_focus_ui(self, force: bool = False) -> None:
        info = self.pipeline.focus_info()
        if info is not None:
            self.grid.set_focus(info["sid"], info["mode"])
            self.btn_focus.setChecked(True)
            self.btn_focus.setText("🎯  Huỷ focus")
            mode = "Tiết kiệm" if info["mode"] == "eco" else "Toàn bộ"
            self.lbl_focus.setText(f"{info['label']} · {FOCUS_STATE_TEXT.get(info['state'], '')} · {mode}")
            self.targets.set_focus_active(True)
        else:
            self.grid.set_focus(None, self.pipeline.focus_mode)
            self.btn_focus.setChecked(self._armed)
            self.btn_focus.setText("🎯  Focus")
            if self._armed or self._focus_msg:
                self.lbl_focus.setText(self._focus_msg)
            else:
                self.lbl_focus.setText("")
            self.targets.set_focus_active(False)

    # ================================================================== nối tín hiệu
    def _wire_settings(self) -> None:
        s = self.settings
        s.modeChanged.connect(self._on_mode_changed)
        s.webcamIndexChanged.connect(self._on_webcam_index)
        s.addStreamRequested.connect(self._add_stream)
        s.addFilesRequested.connect(self._add_files)
        s.removeSourceRequested.connect(self._remove_source)
        s.settingChanged.connect(self._on_setting)
        s.registerFaceRequested.connect(self._register_face)
        s.deleteFaceRequested.connect(self._delete_face)

    def _sync_sources(self) -> None:
        infos = self.pipeline.list_sources()
        self.grid.set_sources(infos)
        self.settings.set_sources(infos)

    # ------------------------------------------------------------------ webcam: chọn là bật, đổi nguồn là tắt
    def _on_mode_changed(self, mode: str) -> None:
        if mode == MODE_WEBCAM:
            self._start_webcam()
        else:
            self._stop_webcam()

    def _start_webcam(self) -> None:
        if self._webcam_id is None:
            idx = self.settings.webcam_index()
            self._webcam_id = self.pipeline.add_source(idx, kind=SourceKind.WEBCAM, name=f"Camera máy tính {idx}")
            self._sync_sources()

    def _stop_webcam(self) -> None:
        if self._webcam_id is not None:
            self.pipeline.remove_source(self._webcam_id)
            self._webcam_id = None
            self._sync_sources()

    def _on_webcam_index(self, idx: int) -> None:
        if self._webcam_id is not None:  # đang bật -> khởi động lại với camera mới
            self._stop_webcam()
            self._start_webcam()

    # ------------------------------------------------------------------ camera mạng / file
    def _add_stream(self, name: str, url: str) -> None:
        self._stop_webcam()
        self.settings.set_mode(MODE_STREAM)
        self.pipeline.add_source(url, kind=SourceKind.STREAM, name=name or url.split("@")[-1])
        self._sync_sources()

    def _add_files(self, paths: list[str]) -> None:
        paths = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXT]
        if not paths:
            return
        self._stop_webcam()
        self.settings.set_mode(MODE_FILE)
        for p in paths:
            self.pipeline.add_source(p, kind=SourceKind.FILE, name=Path(p).name)  # tự chạy + tự phát hiện
        self._sync_sources()

    def _remove_source(self, sid: str) -> None:
        if sid == self._webcam_id:
            self._webcam_id = None
        self.pipeline.remove_source(sid)
        self._sync_sources()

    # ------------------------------------------------------------------ thông số
    def _on_setting(self, key: str, value) -> None:
        if key == "show_motion":
            self.grid.show_motion = bool(value)
        else:
            self.pipeline.apply_settings(**{key: value})

    # ------------------------------------------------------------------ khuôn mặt
    def _register_face(self, name: str) -> None:
        sid = self.grid.selected_id or next(iter(self.grid.visible_ids()), None)
        ok, msg = self.pipeline.register_face(sid, name) if sid else (False, "Chưa có ô video nào.")
        self.settings.set_register_result(ok, msg)
        self.settings.set_people(self.db.list_people())

    def _delete_face(self, name: str) -> None:
        self.db.remove_person(name)
        self.settings.set_people(self.db.list_people())

    # ================================================================== kéo thả file
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(Path(u.toLocalFile()).suffix.lower() in VIDEO_EXT for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._add_files([u.toLocalFile() for u in e.mimeData().urls()])
        e.acceptProposedAction()

    def closeEvent(self, e):
        self.timer.stop()
        self.pipeline.stop()
        super().closeEvent(e)