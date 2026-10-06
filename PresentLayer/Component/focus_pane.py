"""Ô FOCUS (chế độ "Toàn bộ"): hiển thị phần khung hình phóng to quanh mục tiêu, nằm cạnh nguồn chính.

Không vẽ box nào lên hình (yêu cầu của chế độ focus). Chỉ có chip trạng thái nhỏ + nút huỷ.
Khung nhìn (`view.viewport`) do camera ảo trong FocusController tính, đã đúng tỉ lệ của ô này.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QToolButton, QWidget

from PresentLayer.Component.video_tile import C_BG, C_FOCUS, FOCUS_STATE_TEXT


class FocusPane(QWidget):
    cancelRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = None
        self._image: QImage | None = None
        self._frame = None
        self._last_fid = -1
        self.setMinimumSize(160, 160)

        self.btn_cancel = QToolButton(self)
        self.btn_cancel.setText("✕")
        self.btn_cancel.setToolTip("Huỷ focus")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setStyleSheet(
            "QToolButton{background:rgba(0,0,0,120);color:#e5e8ee;border:none;border-radius:4px;padding:2px 6px;}"
            "QToolButton:hover{background:#ef4444;color:white;}")
        self.btn_cancel.clicked.connect(self.cancelRequested)

    def update_view(self, view) -> None:
        self._view = view
        if view.frame is not None and view.frame_id != self._last_fid:
            self._last_fid = view.frame_id
            self._frame = view.frame               # giữ tham chiếu cho vùng nhớ của QImage
            h, w = view.frame.shape[:2]
            self._image = QImage(view.frame.data, w, h, view.frame.strides[0], QImage.Format.Format_BGR888)
        self.update()

    def resizeEvent(self, e):
        self.btn_cancel.move(self.width() - self.btn_cancel.sizeHint().width() - 6, 4)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), C_BG)
        v = self._view
        if self._image is not None and v is not None:
            iw, ih = self._image.width(), self._image.height()
            vp = v.viewport
            sx, sy, sx2, sy2 = vp if vp is not None else (0.0, 0.0, float(iw), float(ih))
            sw, sh = sx2 - sx, sy2 - sy
            s = min(W / sw, H / sh)
            dw, dh = sw * s, sh * s
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF((W - dw) / 2, (H - dh) / 2, dw, dh), self._image, QRectF(sx, sy, sw, sh))
            if v.focus_state == "lost":
                p.fillRect(self.rect(), QColor(0, 0, 0, 140))
                p.setPen(QColor("#fbbf24"))
                p.setFont(QFont(self.font().family(), 11, QFont.Weight.Bold))
                p.drawText(self.rect().adjusted(16, 0, -16, 0),
                           Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                           "Mất mục tiêu\nĐang tìm lại...")
        elif v is None or self._image is None:
            p.setPen(QColor("#6b7686"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Đang chờ khung hình...")

        if v is not None:
            f = QFont(self.font().family(), 9)
            f.setBold(True)
            p.setFont(f)
            text = f"🎯 {v.focus_label} · {FOCUS_STATE_TEXT.get(v.focus_state, '')}"
            text = QFontMetrics(f).elidedText(text, Qt.TextElideMode.ElideRight, max(40, W - 60))
            w = QFontMetrics(f).horizontalAdvance(text) + 12
            h = QFontMetrics(f).height() + 4
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor("#0369a1")))
            p.drawRoundedRect(QRectF(8, 6, w, h), 4, 4)
            p.setPen(QColor("white"))
            p.drawText(QRectF(8, 6, w, h), Qt.AlignmentFlag.AlignCenter, text)

        p.setPen(QPen(C_FOCUS, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self.rect().adjusted(1, 1, -2, -2))
        p.end()