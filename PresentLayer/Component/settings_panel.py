"""Bảng cài đặt nổi ở GÓC TRÊN BÊN TRÁI: nguồn video, thông số phát hiện, đăng ký khuôn mặt."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QRadioButton, QSpinBox,
                               QStackedWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from BusinessLayer.input_manager import SourceKind

MODE_WEBCAM, MODE_STREAM, MODE_FILE = SourceKind.WEBCAM, SourceKind.STREAM, SourceKind.FILE
_PAGE = {None: 0, MODE_WEBCAM: 1, MODE_STREAM: 2, MODE_FILE: 3}
_KIND_LABEL = {SourceKind.WEBCAM: "camera máy tính", SourceKind.STREAM: "camera mạng", SourceKind.FILE: "video"}
VIDEO_FILTER = "Video (*.mp4 *.avi *.mkv *.mov *.m4v *.webm *.wmv);;Tất cả (*.*)"


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color:#8a94a6;font-size:12px;")
    return lb


class SettingsPanel(QFrame):
    modeChanged = Signal(str)                 # 'webcam' | 'stream' | 'file'
    webcamIndexChanged = Signal(int)
    addStreamRequested = Signal(str, str)     # tên, url
    addFilesRequested = Signal(list)
    removeSourceRequested = Signal(str)
    settingChanged = Signal(str, object)
    registerFaceRequested = Signal(str)
    deleteFaceRequested = Signal(str)

    def __init__(self, defaults: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPanel")
        self.setFixedWidth(380)
        self._silent = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("Cài đặt")
        title.setStyleSheet("font-size:15px;font-weight:600;")
        close = QToolButton()
        close.setText("✕")
        close.clicked.connect(self.hide)
        head.addWidget(title)
        head.addStretch()
        head.addWidget(close)
        root.addLayout(head)

        tabs = QTabWidget()
        tabs.addTab(self._build_sources(), "Nguồn")
        tabs.addTab(self._build_detection(defaults), "Phát hiện")
        tabs.addTab(self._build_faces(), "Khuôn mặt")
        root.addWidget(tabs)

    # ================================================================== tab Nguồn
    def _build_sources(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 10, 8, 8)

        self.rb = {
            MODE_WEBCAM: QRadioButton("Camera máy tính"),
            MODE_STREAM: QRadioButton("Camera ngoài (RTSP / HTTP)"),
            MODE_FILE: QRadioButton("Video (MP4)"),
        }
        self._group = QButtonGroup(self)
        for mode, rb in self.rb.items():
            self._group.addButton(rb)
            rb.toggled.connect(lambda checked, m=mode: self._on_mode_toggled(m, checked))
            lay.addWidget(rb)

        self.stack = QStackedWidget()
        # 0: gợi ý ban đầu
        self.stack.addWidget(_note("Chọn loại nguồn ở trên. Có thể thêm nhiều camera và nhiều video cùng lúc."))
        # 1: webcam
        p = QWidget()
        pl = QVBoxLayout(p)
        pl.setContentsMargins(0, 6, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Chỉ số camera"))
        self.spin_cam = QSpinBox()
        self.spin_cam.setRange(0, 9)
        self.spin_cam.valueChanged.connect(self.webcamIndexChanged)
        row.addWidget(self.spin_cam)
        row.addStretch()
        pl.addLayout(row)
        pl.addWidget(_note("Camera tự bật khi chọn mục này và tự tắt khi bạn chuyển sang nguồn khác."))
        self.stack.addWidget(p)
        # 2: camera mạng
        p = QWidget()
        pl = QVBoxLayout(p)
        pl.setContentsMargins(0, 6, 0, 0)
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Tên camera (không bắt buộc)")
        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("rtsp://user:pass@192.168.1.10:554/stream")
        self.edit_url.returnPressed.connect(self._emit_add_stream)
        btn = QPushButton("Thêm camera")
        btn.clicked.connect(self._emit_add_stream)
        for x in (self.edit_name, self.edit_url, btn):
            pl.addWidget(x)
        self.stack.addWidget(p)
        # 3: file
        p = QWidget()
        pl = QVBoxLayout(p)
        pl.setContentsMargins(0, 6, 0, 0)
        btn = QPushButton("Chọn file video...")
        btn.clicked.connect(self._pick_files)
        pl.addWidget(btn)
        pl.addWidget(_note("Hoặc kéo thả file .mp4 vào cửa sổ. Video tự chạy và tự phát hiện người."))
        self.stack.addWidget(p)
        lay.addWidget(self.stack)

        lay.addWidget(QLabel("Nguồn đang chạy"))
        self.list_sources = QListWidget()
        self.list_sources.setFixedHeight(130)
        lay.addWidget(self.list_sources)
        btn = QPushButton("Gỡ nguồn đã chọn")
        btn.clicked.connect(self._emit_remove)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _on_mode_toggled(self, mode: str, checked: bool) -> None:
        if not checked:
            return
        self.stack.setCurrentIndex(_PAGE[mode])
        if not self._silent:
            self.modeChanged.emit(mode)

    def set_mode(self, mode: str | None, silent: bool = True) -> None:
        """Đổi radio bằng code (ví dụ khi kéo thả file) mà không phát lại tín hiệu modeChanged."""
        self._silent = silent
        try:
            if mode in self.rb:
                self.rb[mode].setChecked(True)
            self.stack.setCurrentIndex(_PAGE.get(mode, 0))
        finally:
            self._silent = False

    def set_sources(self, infos: list[tuple[str, str, str]]) -> None:
        self.list_sources.clear()
        for sid, name, kind in infos:
            item = QListWidgetItem(f"{name}   ({_KIND_LABEL.get(kind, kind)})")
            item.setData(Qt.ItemDataRole.UserRole, sid)
            self.list_sources.addItem(item)

    def webcam_index(self) -> int:
        return self.spin_cam.value()

    def _emit_add_stream(self) -> None:
        url = self.edit_url.text().strip()
        if url:
            self.addStreamRequested.emit(self.edit_name.text().strip(), url)
            self.edit_url.clear()
            self.edit_name.clear()

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Chọn video", "", VIDEO_FILTER)
        if paths:
            self.addFilesRequested.emit(paths)

    def _emit_remove(self) -> None:
        item = self.list_sources.currentItem()
        if item:
            self.removeSourceRequested.emit(item.data(Qt.ItemDataRole.UserRole))

    # ================================================================== tab Phát hiện
    def _build_detection(self, d: dict) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(8, 12, 8, 8)
        form.setVerticalSpacing(10)

        self.spin_area = QDoubleSpinBox()
        self.spin_area.setRange(0.05, 10.0)
        self.spin_area.setSingleStep(0.05)
        self.spin_area.setDecimals(2)
        self.spin_area.setSuffix(" %")
        self.spin_area.setValue(float(d.get("min_area_ratio", 0.003)) * 100)
        self.spin_area.valueChanged.connect(lambda v: self.settingChanged.emit("min_area_ratio", v / 100.0))

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 30)
        self.spin_fps.setValue(int(d.get("ai_fps", 10)))
        self.spin_fps.valueChanged.connect(lambda v: self.settingChanged.emit("ai_fps", v))

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.10, 0.95)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(float(d.get("person_conf", 0.30)))
        self.spin_conf.valueChanged.connect(lambda v: self.settingChanged.emit("person_conf", v))

        self.spin_face = QDoubleSpinBox()
        self.spin_face.setRange(0.20, 0.90)
        self.spin_face.setSingleStep(0.01)
        self.spin_face.setValue(float(d.get("face_threshold", 0.45)))
        self.spin_face.valueChanged.connect(lambda v: self.settingChanged.emit("face_threshold", v))

        self.chk_motion = QCheckBox("Hiện vùng chuyển động (gỡ lỗi)")
        self.chk_motion.toggled.connect(lambda v: self.settingChanged.emit("show_motion", v))

        form.addRow("Vùng chuyển động tối thiểu", self.spin_area)
        form.addRow("AI chạy tối đa (lần/giây)", self.spin_fps)
        form.addRow("Ngưỡng nhận diện người", self.spin_conf)
        form.addRow("Ngưỡng khuôn mặt (cosine)", self.spin_face)
        form.addRow(self.chk_motion)
        form.addRow(_note(
            "Vùng chuyển động nhỏ hơn mức tối thiểu bị bỏ qua. Tăng giá trị nếu cây, côn trùng vẫn làm AI chạy. "
            "Bật gỡ lỗi để xem khung vàng (hợp lệ) và khung xám (đã bị lọc)."))
        return w

    # ================================================================== tab Khuôn mặt
    def _build_faces(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.addWidget(_note("Bấm chọn một ô video có người nhìn thẳng vào camera, nhập tên rồi đăng ký."))
        self.edit_person = QLineEdit()
        self.edit_person.setPlaceholderText("Tên người quen")
        lay.addWidget(self.edit_person)
        btn = QPushButton("Đăng ký từ ô đang chọn")
        btn.clicked.connect(self._emit_register)
        lay.addWidget(btn)
        self.lbl_result = _note("")
        lay.addWidget(self.lbl_result)

        lay.addWidget(QLabel("Người quen đã lưu"))
        self.list_people = QListWidget()
        self.list_people.setFixedHeight(120)
        lay.addWidget(self.list_people)
        btn = QPushButton("Xóa người đã chọn")
        btn.clicked.connect(self._emit_delete)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _emit_register(self) -> None:
        name = self.edit_person.text().strip()
        if not name:
            self.set_register_result(False, "Hãy nhập tên trước.")
            return
        self.registerFaceRequested.emit(name)

    def _emit_delete(self) -> None:
        item = self.list_people.currentItem()
        if item:
            self.deleteFaceRequested.emit(item.data(Qt.ItemDataRole.UserRole))

    def set_register_result(self, ok: bool, msg: str) -> None:
        self.lbl_result.setText(msg)
        self.lbl_result.setStyleSheet(f"color:{'#22c55e' if ok else '#f87171'};font-size:12px;")

    def set_people(self, people: list[tuple[str, int]]) -> None:
        self.list_people.clear()
        for name, n in people:
            item = QListWidgetItem(f"{name}   ({n} mẫu)")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list_people.addItem(item)
