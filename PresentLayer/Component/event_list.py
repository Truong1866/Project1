"""Bảng danh sách sự kiện (người quen / người lạ), mới nhất ở trên cùng."""
from __future__ import annotations

import time

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from Utils.event_bus import PERSON_KNOWN, PERSON_UNKNOWN

_TEXT = {PERSON_KNOWN: "Người quen", PERSON_UNKNOWN: "Người lạ"}
_COLOR = {PERSON_KNOWN: QColor("#4ade80"), PERSON_UNKNOWN: QColor("#f87171")}


class EventList(QWidget):
    MAX_ROWS = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Sự kiện")
        title.setStyleSheet("font-size:14px;font-weight:600;")
        lay.addWidget(title)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Giờ", "Nguồn", "Sự kiện", "Tên"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table)

    def add_event(self, ev: dict) -> None:
        topic = ev.get("topic")
        row = [time.strftime("%H:%M:%S", time.localtime(ev.get("ts", time.time()))),
               ev.get("source_name", ""), _TEXT.get(topic, topic),
               ev.get("name") if topic == PERSON_KNOWN else "-"]
        self.table.insertRow(0)
        for col, text in enumerate(row):
            item = QTableWidgetItem(str(text))
            if col == 2 and topic in _COLOR:
                item.setForeground(_COLOR[topic])
            self.table.setItem(0, col, item)
        while self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(self.table.rowCount() - 1)