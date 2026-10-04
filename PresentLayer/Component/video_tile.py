"""Một ô video: vẽ khung hình + box người + nhãn + trạng thái bằng QPainter (hỗ trợ tiếng Việt, không tốn CPU vẽ lên ảnh)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QToolButton, QWidget

from BusinessLayer.input_manager import SourceKind, Status

C_BG = QColor("#07090c")
C_KNOWN = QColor("#22c55e")
C_UNKNOWN = QColor("#ef4444")
C_PENDING = QColor("#f59e0b")
C_MOTION = QColor("#facc15")
C_REJECTED = QColor("#6b7280")
C_ACCENT = QColor("#4f8cff")

KIND_TEXT = {SourceKind.WEBCAM: "Camera máy tính", SourceKind.STREAM: "Camera mạng", SourceKind.FILE: "Video"}
STATUS_TEXT = {
    Status.CONNECTING: "Đang kết nối...",
    Status.RECONNECTING: "Mất tín hiệu, đang kết nối lại...",
    Status.ERROR: "Không mở được nguồn này",
    Status.STOPPED: "Đã dừng",
}
KIND_COLOR = {"known": C_KNOWN, "unknown": C_UNKNOWN, "pending": C_PENDING}


class VideoTile(QWidget):
    closeRequested = Signal(str)
    clicked = Signal(str)
    doubleClicked = Signal(str)

    def __init__(self, source_id: str, parent=None):
        super().__init__(parent)
        self.source_id = source_id
        self.selected = False
        self._view = None
        self._image: QImage | None = None
        self._frame = None            # giữ tham chiếu để vùng nhớ của QImage không bị giải phóng
        self._last_fid = -1
        self._show_motion = False
        self.setMinimumSize(200, 140)
        self.setMouseTracking(True)

        self.btn_close = QToolButton(self)
        self.btn_close.setText("✕")
        self.btn_close.setToolTip("Gỡ nguồn này")
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet(
            "QToolButton{background:rgba(0,0,0,120);color:#e5e8ee;border:none;border-radius:4px;padding:2px 6px;}"
            "QToolButton:hover{background:#ef4444;color:white;}")
        self.btn_close.clicked.connect(lambda: self.closeRequested.emit(self.source_id))

    # ------------------------------------------------------------------ dữ liệu
    def update_view(self, view, show_motion: bool) -> None:
        self._view = view
        self._show_motion = show_motion
        if view.frame is not None and view.frame_id != self._last_fid:
            self._last_fid = view.frame_id
            self._frame = view.frame
            h, w = view.frame.shape[:2]
            self._image = QImage(view.frame.data, w, h, view.frame.strides[0], QImage.Format.Format_BGR888)
        self.update()

    # ------------------------------------------------------------------ sự kiện chuột
    def resizeEvent(self, e):
        self.btn_close.move(self.width() - self.btn_close.sizeHint().width() - 6, 4)

    def mousePressEvent(self, e):
        self.clicked.emit(self.source_id)

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit(self.source_id)

    # ------------------------------------------------------------------ vẽ
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), C_BG)
        v = self._view

        ox = oy = 0.0
        s = 1.0
        if self._image is not None:
            iw, ih = self._image.width(), self._image.height()
            s = min(W / iw, H / ih)
            dw, dh = iw * s, ih * s
            ox, oy = (W - dw) / 2, (H - dh) / 2
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF(ox, oy, dw, dh), self._image)

        if v is not None:
            if self._show_motion:
                self._draw_motion(p, v, ox, oy, s)
            for t in v.tracks:
                self._draw_track(p, t, ox, oy, s)
            self._draw_header(p, v, W)
            self._draw_footer(p, v, W, H)
            if self._image is None or v.status in STATUS_TEXT:
                msg = STATUS_TEXT.get(v.status, "Đang chờ khung hình...")
                p.setPen(QColor("#9aa4b5"))
                p.setFont(QFont(self.font().family(), 11))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, msg)

        if self.selected:
            p.setPen(QPen(C_ACCENT, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect().adjusted(1, 1, -2, -2))

    def _chip(self, p: QPainter, x: float, y: float, text: str, bg: QColor, fg=QColor("white"), size=9) -> float:
        f = QFont(self.font().family(), size)
        f.setBold(True)
        p.setFont(f)
        fm = QFontMetrics(f)
        w, h = fm.horizontalAdvance(text) + 12, fm.height() + 4
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(QRectF(x, y, w, h), 4, 4)
        p.setPen(fg)
        p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)
        return w

    def _draw_track(self, p: QPainter, t, ox, oy, s) -> None:
        color = KIND_COLOR.get(t.kind, C_PENDING)
        x1, y1, x2, y2 = t.box
        r = QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)
        p.setPen(QPen(color, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        if t.face_box:
            fx1, fy1, fx2, fy2 = t.face_box
            p.setPen(QPen(color, 2, Qt.PenStyle.DotLine))
            p.drawRect(QRectF(ox + fx1 * s, oy + fy1 * s, (fx2 - fx1) * s, (fy2 - fy1) * s))
        text = t.label if t.kind != "known" or not t.score else f"{t.label}  {t.score:.2f}"
        fm = QFontMetrics(QFont(self.font().family(), 9, QFont.Weight.Bold))
        ch = fm.height() + 4
        y = r.top() - ch - 1 if r.top() - ch - 1 > 0 else r.top() + 1
        self._chip(p, r.left(), y, text, color)

    def _draw_motion(self, p: QPainter, v, ox, oy, s) -> None:
        for boxes, color, style in ((v.motion_rejected, C_REJECTED, Qt.PenStyle.DotLine),
                                    (v.motion_boxes, C_MOTION, Qt.PenStyle.DashLine)):
            p.setPen(QPen(color, 1.5, style))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for x1, y1, x2, y2 in boxes:
                p.drawRect(QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s))

    def _draw_header(self, p: QPainter, v, W) -> None:
        p.fillRect(QRectF(0, 0, W, 30), QColor(0, 0, 0, 150))
        f = QFont(self.font().family(), 10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#f1f4f9"))
        fps = f"   {v.fps:.0f} fps" if v.fps and v.status == Status.RUNNING else ""
        text = f"{v.name}{fps}"
        fm = QFontMetrics(f)
        text = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(W - 120))
        p.drawText(QRectF(10, 0, W - 50, 30), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        # chip loại nguồn nằm trước nút đóng
        kind = KIND_TEXT.get(v.kind, v.kind)
        f2 = QFont(self.font().family(), 8)
        w = QFontMetrics(f2).horizontalAdvance(kind) + 12
        p.setFont(f2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawRoundedRect(QRectF(W - 36 - w, 6, w, 18), 4, 4)
        p.setPen(QColor("#c8cfdb"))
        p.drawText(QRectF(W - 36 - w, 6, w, 18), Qt.AlignmentFlag.AlignCenter, kind)

    def _draw_footer(self, p: QPainter, v, W, H) -> None:
        if v.status == Status.ENDED:
            self._chip(p, 8, H - 26, "Đã phát xong", QColor("#374151"))
        elif v.gate == "active":
            ms = f"  {v.ai_ms:.0f} ms" if v.ai_ms else ""
            self._chip(p, 8, H - 26, f"Đang phân tích{ms}", QColor("#2563eb"))
        else:
            self._chip(p, 8, H - 26, "Chờ chuyển động", QColor(55, 65, 81, 200))
        if v.kind == SourceKind.FILE:
            p.fillRect(QRectF(0, H - 3, W, 3), QColor(255, 255, 255, 40))
            p.fillRect(QRectF(0, H - 3, W * float(v.progress), 3), C_ACCENT)


class EmptyTile(QWidget):
    """Ô trống: gợi ý kéo thả video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = "Ô trống\nKéo thả video vào đây để thêm"
        self.setMinimumSize(200, 140)

    def set_text(self, text: str) -> None:
        self.text = text
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0e13"))
        pen = QPen(QColor("#2b3342"), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(8, 8, -8, -8), 10, 10)
        p.setPen(QColor("#6b7686"))
        p.setFont(QFont(self.font().family(), 10))
        p.drawText(self.rect().adjusted(20, 0, -20, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.text)
