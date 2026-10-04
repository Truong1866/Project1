"""Khu vực video chính: 1 nguồn -> toàn khung; >=2 nguồn -> chia 4 phần (2x2); >4 nguồn -> phân trang, mỗi trang 4 ô."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from PresentLayer.Component.video_tile import EmptyTile, VideoTile


class VideoGrid(QWidget):
    removeRequested = Signal(str)
    selectionChanged = Signal(str)

    def __init__(self, pipeline, per_page: int = 4, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.per_page = per_page
        self.show_motion = False
        self.selected_id: str | None = None

        self._order: list[str] = []
        self._tiles: dict[str, VideoTile] = {}
        self._empty = [EmptyTile(self) for _ in range(per_page)]
        self._page = 0
        self._focus: str | None = None

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
                self._tiles[sid] = tile
        self._order = ids
        if self.selected_id not in ids:
            self.selected_id = None
        if self._focus not in ids:
            self._focus = None
        self._page = max(0, min(self._page, self.page_count - 1))
        self._relayout()

    @property
    def page_count(self) -> int:
        return max(1, math.ceil(len(self._order) / self.per_page))

    def set_page(self, page: int) -> None:
        self._page = max(0, min(page, self.page_count - 1))
        self._relayout()

    def visible_ids(self) -> list[str]:
        if self._focus:
            return [self._focus]
        if len(self._order) <= 1:
            return list(self._order)
        start = self._page * self.per_page
        return self._order[start:start + self.per_page]

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

    # ------------------------------------------------------------------ làm mới từ timer
    def refresh(self) -> None:
        for sid in self.visible_ids():
            view = self.pipeline.get_view(sid)
            if view is not None and sid in self._tiles:
                self._tiles[sid].update_view(view, self.show_motion)