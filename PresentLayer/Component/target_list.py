"""Danh sách MỤC TIÊU đang có trên màn hình (cùng kiểu bảng với EventList) - nhấp một dòng để focus.

Dữ liệu lấy từ pipeline.list_targets(): mỗi dict gồm sid, source_name, track_id, label, kind, focused.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

_KIND_TEXT = {"known": "Người quen", "unknown": "Người lạ", "pending": "Đang xác định"}
_KIND_COLOR = {"known": QColor("#4ade80"), "unknown": QColor("#f87171"), "pending": QColor("#fbbf24")}
_FOCUS_COLOR = QColor("#38bdf8")


class TargetList(QWidget):
    focusRequested = Signal(str, int)     # source_id, track_id
    cancelRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sig = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Mục tiêu trên màn hình")
        title.setStyleSheet("font-size:14px;font-weight:600;")
        hint = QLabel("Nhấp một dòng để focus. Hoặc nhấp trực tiếp vào người trong video (Ctrl+nhấp).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8a94a6;font-size:12px;")
        lay.addWidget(title)
        lay.addWidget(hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Nguồn", "Mục tiêu", "Trạng thái"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._on_clicked)
        lay.addWidget(self.table, 1)

        self.btn_cancel = QPushButton("Huỷ focus")
        self.btn_cancel.clicked.connect(self.cancelRequested)
        self.btn_cancel.hide()
        lay.addWidget(self.btn_cancel)

    def set_targets(self, targets: list[dict]) -> None:
        sig = tuple((t["sid"], t["track_id"], t["label"], t["kind"], t["focused"]) for t in targets)
        if sig == self._sig:
            return                                   # không đổi -> không dựng lại (giữ nguyên dòng đang chọn)
        self._sig = sig
        self.table.setRowCount(0)
        for t in targets:
            r = self.table.rowCount()
            self.table.insertRow(r)
            focused = t["focused"]
            cells = [t["source_name"], ("🎯 " if focused else "") + t["label"], _KIND_TEXT.get(t["kind"], t["kind"])]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, (t["sid"], int(t["track_id"])))
                if c == 2:
                    item.setForeground(_KIND_COLOR.get(t["kind"], QColor("white")))
                if focused:
                    item.setBackground(QColor(14, 116, 144, 90))
                    if c == 1:
                        item.setForeground(_FOCUS_COLOR)
                self.table.setItem(r, c, item)

    def set_focus_active(self, active: bool) -> None:
        self.btn_cancel.setVisible(active)

    def _on_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        data = item.data(Qt.ItemDataRole.UserRole) if item else None
        if data:
            self.focusRequested.emit(data[0], int(data[1]))