"""Khu vực video chính: 1 nguồn -> toàn khung; >=2 nguồn -> chia 4 phần (2x2); >4 nguồn -> phân trang, mỗi trang 4 ô.

FOCUS:
  * eco  : ô của nguồn đang focus được phóng to chiếm cả vùng (các ô khác tạm ẩn); ô tự zoom vào mục tiêu.
  * full : ô nguồn chính (cả khung + box) nằm bên trái, FocusPane (mục tiêu phóng to, không box) nằm ngay bên phải.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from PresentLayer.Component.focus_pane import FocusPane
from PresentLayer.Component.video_tile import EmptyTile, VideoTile


class VideoGrid(QWidget):
    removeRequested = Signal(str)
    selectionChanged = Signal(str)
    targetPicked = Signal(str, int)         # source_id, track_id  (nhấp chọn mục tiêu để focus)
    focusButtonClicked = Signal(str)        # nút 🎯 trên một ô
    focusCancelRequested = Signal()

    def __init__(self, pipeline, per_page: int = 4, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.per_page = per_page
        self.show_motion = False
        self.selected_id: str | None = None
        self.pick_mode = False
        self.focus_sid: str | None = None
        self.focus_mode = "eco"

        self._order: list[str] = []
        self._tiles: dict[str, VideoTile] = {}
        self._empty = [EmptyTile(self) for _ in range(per_page)]
        self._page = 0
        self._focus: str | None = None
        self._focus_pane = FocusPane(self)
        self._focus_pane.hide()
        self._focus_pane.cancelRequested.connect(self.focusCancelRequested)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        host = QWidget(self)
        self._grid = QGridLayout(host)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._grid.setSpacing(6)
        root.addWidget(host, 1)

        self._nav = QWidget(self)
        nav = QHBoxLayout(self._nav)
        nav.setContentsMargins(6, 0, 6, 6)
        self._btn_prev = QPushButton("◀")
        self._btn_next = QPushButton("▶")
        self._lbl_page = QLabel("")
        self._lbl_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for b in (self._btn_prev, self._btn_next):
            b.setFixedWidth(44)
        self._btn_prev.clicked.connect(lambda: self.set_page(self._page - 1))
        self._btn_next.clicked.connect(lambda: self.set_page(self._page + 1))
        nav.addStretch()
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._lbl_page)
        nav.addWidget(self._btn_next)
        nav.addStretch()
        root.addWidget(self._nav)
        self._relayout()

    # ------------------------------------------------------------------ nguồn
    def set_sources(self, infos: list[tuple[str, str, str]]) -> None:
        ids = [i for i, _, _ in infos]
        for sid in [s for s in self._tiles if s not in ids]:
            tile = self._tiles.pop(sid)
            self._grid.removeWidget(tile)
            tile.deleteLater()
        for sid in ids:
            if sid not in self._tiles:
                tile = VideoTile(sid, self)
                tile.closeRequested.connect(self.removeRequested)
                tile.clicked.connect(self._on_clicked)
                tile.doubleClicked.connect(self._on_double_clicked)
                tile.targetClicked.connect(self.targetPicked)
                tile.focusButtonClicked.connect(self.focusButtonClicked)
                tile.set_pick_mode(self.pick_mode)
                self._tiles[sid] = tile
        self._order = ids
        if self.selected_id not in ids:
            self.selected_id = None
        if self._focus not in ids:
            self._focus = None
        if self.focus_sid not in ids:
            self.focus_sid = None
        self._page = max(0, min(self._page, self.page_count - 1))
        self._relayout()

    @property
    def page_count(self) -> int:
        return max(1, math.ceil(len(self._order) / self.per_page))

    def set_page(self, page: int) -> None:
        self._page = max(0, min(page, self.page_count - 1))
        self._relayout()

    def visible_ids(self) -> list[str]:
        if self.focus_sid in self._tiles:
            return [self.focus_sid]
        if self._focus:
            return [self._focus]
        if len(self._order) <= 1:
            return list(self._order)
        start = self._page * self.per_page
        return self._order[start:start + self.per_page]

    # ------------------------------------------------------------------ focus
    def set_focus(self, sid: str | None, mode: str = "eco") -> None:
        """sid=None: tắt bố cục focus. Gọi mỗi nhịp cũng được (chỉ dựng lại bố cục khi có thay đổi)."""
        if sid not in self._tiles:
            sid = None
        if (sid, mode) == (self.focus_sid, self.focus_mode):
            return
        self.focus_sid, self.focus_mode = sid, mode
        self._relayout()

    def set_pick_mode(self, on: bool) -> None:
        self.pick_mode = on
        for t in self._tiles.values():
            t.set_pick_mode(on)

    # ------------------------------------------------------------------ tương tác
    def _on_clicked(self, sid: str) -> None:
        self.selected_id = sid
        for s, t in self._tiles.items():
            t.selected = (s == sid)
            t.update()
        self.selectionChanged.emit(sid)

    def _on_double_clicked(self, sid: str) -> None:
        self._focus = None if self._focus == sid else sid   # nhấp đúp: phóng to / thu nhỏ
        self._relayout()

    # ------------------------------------------------------------------ bố cục
    def _relayout(self) -> None:
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.hide()
        fsid = self.focus_sid if self.focus_sid in self._tiles else None
        if fsid:
            self._layout_focus(fsid)
            return

        ids = self.visible_ids()
        total = len(self._order)
        single = bool(self._focus) or total <= 1
        cols = rows = 1 if single else 2

        for n, sid in enumerate(ids):
            self._grid.addWidget(self._tiles[sid], n // cols, n % cols)
            self._tiles[sid].show()
        if single and not ids:
            self._empty[0].set_text("Chưa có nguồn video\nKéo thả file .mp4 vào đây hoặc mở ⚙ Cài đặt để chọn camera")
            self._grid.addWidget(self._empty[0], 0, 0)
            self._empty[0].show()
        elif not single:
            for n in range(len(ids), rows * cols):
                self._empty[n].set_text("Ô trống\nKéo thả video vào đây để thêm")
                self._grid.addWidget(self._empty[n], n // cols, n % cols)
                self._empty[n].show()
        for r in range(2):
            self._grid.setRowStretch(r, 1 if r < rows else 0)
            self._grid.setColumnStretch(r, 1 if r < cols else 0)

        for sid, t in self._tiles.items():
            t.selected = (sid == self.selected_id)
        pages = self.page_count
        show_nav = pages > 1 and not self._focus
        self._nav.setVisible(show_nav)
        self._lbl_page.setText(f"Trang {self._page + 1} / {pages}")
        self._btn_prev.setEnabled(self._page > 0)
        self._btn_next.setEnabled(self._page < pages - 1)

    def _layout_focus(self, sid: str) -> None:
        tile = self._tiles[sid]
        self._grid.addWidget(tile, 0, 0)
        tile.show()
        if self.focus_mode == "full":
            self._grid.addWidget(self._focus_pane, 0, 1)
            self._focus_pane.show()
            self._grid.setColumnStretch(0, 3)
            self._grid.setColumnStretch(1, 2)
        else:
            self._grid.setColumnStretch(0, 1)
            self._grid.setColumnStretch(1, 0)
        self._grid.setRowStretch(0, 1)
        self._grid.setRowStretch(1, 0)
        for s, t in self._tiles.items():
            t.selected = (s == self.selected_id)
        self._nav.setVisible(False)

    # ------------------------------------------------------------------ làm mới từ timer
    def refresh(self) -> None:
        for sid in self.visible_ids():
            tile = self._tiles.get(sid)
            if tile is None:
                continue
            aspect = None
            if sid == self.focus_sid:        # camera ảo cần tỉ lệ của widget sẽ hiển thị khung nhìn
                target = self._focus_pane if self.focus_mode == "full" else tile
                aspect = max(0.2, target.width() / max(1, target.height()))
            view = self.pipeline.get_view(sid, aspect)
            if view is None:
                continue
            tile.update_view(view, self.show_motion)
            if sid == self.focus_sid and self.focus_mode == "full":
                self._focus_pane.update_view(view)