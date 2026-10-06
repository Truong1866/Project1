"""Một ô video: vẽ khung hình + box người + nhãn + trạng thái bằng QPainter (hỗ trợ tiếng Việt, không tốn CPU vẽ lên ảnh).

Hỗ trợ FOCUS:
  * Chế độ tiết kiệm: ô tự phóng to vào `view.viewport` (khung nhìn do camera ảo tính) và KHÔNG vẽ box.
  * Chế độ toàn bộ: ô vẫn hiện cả khung + box, riêng mục tiêu được tô nổi bật (khung hình phóng to nằm ở FocusPane).
  * Nhấp vào người (khi đang ở chế độ chọn mục tiêu, hoặc giữ Ctrl) -> phát tín hiệu targetClicked.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
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
C_FOCUS = QColor("#38bdf8")

KIND_TEXT = {SourceKind.WEBCAM: "Camera máy tính", SourceKind.STREAM: "Camera mạng", SourceKind.FILE: "Video"}
STATUS_TEXT = {
    Status.CONNECTING: "Đang kết nối...",
    Status.RECONNECTING: "Mất tín hiệu, đang kết nối lại...",
    Status.ERROR: "Không mở được nguồn này",
    Status.STOPPED: "Đã dừng",
}
KIND_COLOR = {"known": C_KNOWN, "unknown": C_UNKNOWN, "pending": C_PENDING}
FOCUS_STATE_TEXT = {"acquire": "đang xác định", "track": "đang bám theo", "lost": "mất mục tiêu, đang tìm lại"}
FOCUS_MODE_TEXT = {"eco": "Tiết kiệm", "full": "Toàn bộ"}


class VideoTile(QWidget):
    closeRequested = Signal(str)
    clicked = Signal(str)
    doubleClicked = Signal(str)
    targetClicked = Signal(str, int)      # source_id, track_id
    focusButtonClicked = Signal(str)      # nút 🎯 trên ô: bật focus nhanh / huỷ focus

    def __init__(self, source_id: str, parent=None):
        super().__init__(parent)
        self.source_id = source_id
        self.selected = False
        self.pick_mode = False
        self._hover_tid: int | None = None
        self._view = None
        self._image: QImage | None = None
        self._frame = None            # giữ tham chiếu để vùng nhớ của QImage không bị giải phóng
        self._last_fid = -1
        self._show_motion = False
        self._xf = None               # (ox, oy, scale, src_x, src_y): ánh xạ toạ độ ảnh gốc -> toạ độ widget
        self.setMinimumSize(200, 140)
        self.setMouseTracking(True)

        btn_css = ("QToolButton{background:rgba(0,0,0,120);color:#e5e8ee;border:none;border-radius:4px;padding:2px 6px;}"
                   "QToolButton:hover{background:%s;color:white;}")
        self.btn_close = QToolButton(self)
        self.btn_close.setText("✕")
        self.btn_close.setToolTip("Gỡ nguồn này")
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet(btn_css % "#ef4444")
        self.btn_close.clicked.connect(lambda: self.closeRequested.emit(self.source_id))

        self.btn_focus = QToolButton(self)
        self.btn_focus.setText("🎯")
        self.btn_focus.setToolTip("Bật focus vào một người trong ô này")
        self.btn_focus.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_focus.setStyleSheet(btn_css % "#0ea5e9")
        self.btn_focus.clicked.connect(lambda: self.focusButtonClicked.emit(self.source_id))

    # ------------------------------------------------------------------ dữ liệu
    def update_view(self, view, show_motion: bool) -> None:
        self._view = view
        self._show_motion = show_motion
        if view.frame is not None and view.frame_id != self._last_fid:
            self._last_fid = view.frame_id
            self._frame = view.frame
            h, w = view.frame.shape[:2]
            self._image = QImage(view.frame.data, w, h, view.frame.strides[0], QImage.Format.Format_BGR888)
        focusing = view.focus_state != "idle"
        self.btn_focus.setText("⏹" if focusing else "🎯")
        self.btn_focus.setToolTip("Huỷ focus" if focusing else "Bật focus vào một người trong ô này")
        self.update()

    def set_pick_mode(self, on: bool) -> None:
        self.pick_mode = on
        self._hover_tid = None
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    # ------------------------------------------------------------------ ánh xạ toạ độ
    def _to_frame(self, wx: float, wy: float):
        if self._xf is None:
            return None
        ox, oy, s, sx, sy = self._xf
        return sx + (wx - ox) / s, sy + (wy - oy) / s

    def _r(self, x1, y1, x2, y2) -> QRectF:
        ox, oy, s, sx, sy = self._xf
        return QRectF(ox + (x1 - sx) * s, oy + (y1 - sy) * s, (x2 - x1) * s, (y2 - y1) * s)

    def _boxes_hidden(self) -> bool:
        """Chế độ tiết kiệm đang focus: màn chính không vẽ box."""
        v = self._view
        return bool(v is not None and v.focus_mode == "eco" and v.focus_state in ("acquire", "track"))

    def _track_at(self, wx: float, wy: float):
        v = self._view
        pt = self._to_frame(wx, wy)
        if v is None or pt is None or self._boxes_hidden():
            return None
        fx, fy = pt
        best, best_area = None, float("inf")
        for t in v.tracks:
            x1, y1, x2, y2 = t.box
            if x1 - 8 <= fx <= x2 + 8 and y1 - 8 <= fy <= y2 + 8:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = t, area
        return best

    # ------------------------------------------------------------------ sự kiện chuột
    def resizeEvent(self, e):
        x = self.width() - 6
        x -= self.btn_close.sizeHint().width()
        self.btn_close.move(x, 4)
        x -= self.btn_focus.sizeHint().width() + 4
        self.btn_focus.move(x, 4)

    def mousePressEvent(self, e):
        self.clicked.emit(self.source_id)
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if e.button() == Qt.MouseButton.LeftButton and (self.pick_mode or ctrl):
            t = self._track_at(e.position().x(), e.position().y())
            if t is not None:
                self.targetClicked.emit(self.source_id, int(t.track_id))

    def mouseMoveEvent(self, e):
        if self.pick_mode:
            t = self._track_at(e.position().x(), e.position().y())
            tid = t.track_id if t is not None else None
            if tid != self._hover_tid:
                self._hover_tid = tid
                self.update()

    def leaveEvent(self, e):
        if self._hover_tid is not None:
            self._hover_tid = None
            self.update()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit(self.source_id)

    # ------------------------------------------------------------------ vẽ
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), C_BG)
        v = self._view

        self._xf = None
        if self._image is not None:
            iw, ih = self._image.width(), self._image.height()
            sx, sy, sx2, sy2 = 0.0, 0.0, float(iw), float(ih)
            if v is not None and v.focus_mode == "eco" and v.focus_state != "idle" and v.viewport is not None:
                sx, sy, sx2, sy2 = v.viewport                      # camera ảo: chỉ vẽ phần được phóng to
            sw, sh = sx2 - sx, sy2 - sy
            s = min(W / sw, H / sh)
            dw, dh = sw * s, sh * s
            ox, oy = (W - dw) / 2, (H - dh) / 2
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF(ox, oy, dw, dh), self._image, QRectF(sx, sy, sw, sh))
            self._xf = (ox, oy, s, sx, sy)

        if v is not None:
            if self._xf is not None and not self._boxes_hidden():
                if self._show_motion:
                    self._draw_motion(p, v)
                for t in v.tracks:
                    self._draw_track(p, t, hover=(self.pick_mode and t.track_id == self._hover_tid))
            self._draw_header(p, v, W)
            self._draw_footer(p, v, W, H)
            if v.focus_state != "idle":
                self._draw_focus_chip(p, v)
            if self._image is None or v.status in STATUS_TEXT:
                msg = STATUS_TEXT.get(v.status, "Đang chờ khung hình...")
                p.setPen(QColor("#9aa4b5"))
                p.setFont(QFont(self.font().family(), 11))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, msg)
        if self.pick_mode:
            self._chip(p, max(8, W / 2 - 110), H - 54, "Nhấp vào người cần focus", C_FOCUS, QColor("#06222f"), 10)

        if self.selected:
            p.setPen(QPen(C_ACCENT, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect().adjusted(1, 1, -2, -2))
        p.end()

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

    def _draw_track(self, p: QPainter, t, hover: bool = False) -> None:
        focused = bool(getattr(t, "focused", False))
        color = C_FOCUS if focused else KIND_COLOR.get(t.kind, C_PENDING)
        x1, y1, x2, y2 = t.box
        r = self._r(x1, y1, x2, y2)
        p.setPen(QPen(color, 4 if (focused or hover) else 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        if t.face_box and not focused:
            fx1, fy1, fx2, fy2 = t.face_box
            p.setPen(QPen(color, 2, Qt.PenStyle.DotLine))
            p.drawRect(self._r(fx1, fy1, fx2, fy2))
        text = t.label if t.kind != "known" or not t.score else f"{t.label}  {t.score:.2f}"
        if focused:
            text = "🎯 " + text
        if hover:
            text += "  — nhấp để focus"
        fm = QFontMetrics(QFont(self.font().family(), 9, QFont.Weight.Bold))
        ch = fm.height() + 4
        y = r.top() - ch - 1 if r.top() - ch - 1 > 0 else r.top() + 1
        self._chip(p, r.left(), y, text, color)

    def _draw_motion(self, p: QPainter, v) -> None:
        for boxes, color, style in ((v.motion_rejected, C_REJECTED, Qt.PenStyle.DotLine),
                                    (v.motion_boxes, C_MOTION, Qt.PenStyle.DashLine)):
            p.setPen(QPen(color, 1.5, style))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for x1, y1, x2, y2 in boxes:
                p.drawRect(self._r(x1, y1, x2, y2))

    def _draw_header(self, p: QPainter, v, W) -> None:
        p.fillRect(QRectF(0, 0, W, 30), QColor(0, 0, 0, 150))
        f = QFont(self.font().family(), 10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#f1f4f9"))
        fps = f"   {v.fps:.0f} fps" if v.fps and v.status == Status.RUNNING else ""
        text = f"{v.name}{fps}"
        fm = QFontMetrics(f)
        text = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(W - 150))
        p.drawText(QRectF(10, 0, W - 80, 30), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        # chip loại nguồn nằm trước các nút
        kind = KIND_TEXT.get(v.kind, v.kind)
        f2 = QFont(self.font().family(), 8)
        w = QFontMetrics(f2).horizontalAdvance(kind) + 12
        p.setFont(f2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        x0 = W - 74 - w
        p.drawRoundedRect(QRectF(x0, 6, w, 18), 4, 4)
        p.setPen(QColor("#c8cfdb"))
        p.drawText(QRectF(x0, 6, w, 18), Qt.AlignmentFlag.AlignCenter, kind)

    def _draw_focus_chip(self, p: QPainter, v) -> None:
        state = FOCUS_STATE_TEXT.get(v.focus_state, v.focus_state)
        zoom = ""
        if v.focus_mode == "eco" and v.viewport is not None and self._image is not None:
            zoom = f" · ×{self._image.height() / max(1.0, v.viewport[3] - v.viewport[1]):.1f}"
        text = f"🎯 {v.focus_label} · {state} · {FOCUS_MODE_TEXT.get(v.focus_mode, '')}{zoom}"
        bg = QColor("#b45309") if v.focus_state == "lost" else QColor("#0369a1")
        self._chip(p, 8, 36, text, bg)

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
        p.end()