"""Tab "Focus" trong bảng cài đặt: chọn chế độ focus + vài thông số camera ảo.

Dùng trong SettingsPanel.__init__ chỉ với một dòng:
    tabs.addTab(build_focus_tab(self, defaults), "Focus")
Tín hiệu phát ra qua panel.settingChanged(key, value): focus_mode | focus_zoom | focus_smooth.
"""
from __future__ import annotations

from PySide6.QtWidgets import QButtonGroup, QDoubleSpinBox, QFormLayout, QLabel, QRadioButton, QVBoxLayout, QWidget

FOCUS_ECO, FOCUS_FULL = "eco", "full"


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color:#8a94a6;font-size:12px;")
    return lb


def build_focus_tab(panel, d: dict) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 10, 8, 8)
    lay.addWidget(_note("Chế độ dùng khi bấm 🎯 Focus (đổi được cả khi đang focus)."))

    panel.rb_focus = {
        FOCUS_ECO: QRadioButton("Tiết kiệm"),
        FOCUS_FULL: QRadioButton("Toàn bộ"),
    }
    panel._focus_group = QButtonGroup(panel)
    current = d.get("focus_mode", FOCUS_ECO)
    for mode, rb in panel.rb_focus.items():
        panel._focus_group.addButton(rb)
        rb.setChecked(mode == current)
        rb.toggled.connect(lambda checked, m=mode: checked and panel.settingChanged.emit("focus_mode", m))
        lay.addWidget(rb)
    lay.addWidget(_note(
        "• Tiết kiệm: chỉ quét vùng nhỏ quanh mục tiêu (thu hẹp theo độ dời), ô chính tự phóng to, không vẽ box.\n"
        "• Toàn bộ: vẫn quét cả khung như bình thường, mục tiêu phóng to hiện ở ô riêng bên cạnh nguồn chính."))

    form = QFormLayout()
    form.setVerticalSpacing(10)
    spin_zoom = QDoubleSpinBox()
    spin_zoom.setRange(1.2, 4.0)
    spin_zoom.setSingleStep(0.1)
    spin_zoom.setValue(float(d.get("focus_zoom", 1.8)))
    spin_zoom.valueChanged.connect(lambda v: panel.settingChanged.emit("focus_zoom", v))
    spin_smooth = QDoubleSpinBox()
    spin_smooth.setRange(0.10, 1.00)
    spin_smooth.setSingleStep(0.05)
    spin_smooth.setSuffix(" s")
    spin_smooth.setValue(float(d.get("focus_smooth", 0.25)))
    spin_smooth.valueChanged.connect(lambda v: panel.settingChanged.emit("focus_smooth", v))
    form.addRow("Khung nhìn / chiều cao người", spin_zoom)
    form.addRow("Độ mượt camera", spin_smooth)
    lay.addLayout(form)
    lay.addWidget(_note("Khung nhìn nhỏ = phóng to nhiều hơn. Độ mượt lớn = camera chậm, êm hơn; nhỏ = bám sát hơn."))
    lay.addStretch()
    panel.spin_focus_zoom, panel.spin_focus_smooth = spin_zoom, spin_smooth
    return w