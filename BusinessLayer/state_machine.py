"""State machine cho SmartVisionPipeline.

Định nghĩa toàn bộ các trạng thái của ứng dụng và các chuyển trạng thái:

  ┌─────────────────────────────────────────────────────────┐
  │                    AI GATE STATE                        │
  │   IDLE ──(motion_streak >= confirm)──► ACTIVE           │
  │   ACTIVE ──(no person timeout)──────► IDLE              │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │                   FOCUS STATE                           │
  │   IDLE ──(start_focus called)───► ACQUIRE               │
  │   ACQUIRE ──(confirmed N frames)─► TRACK                │
  │   TRACK ──(lost N misses/timeout)─► LOST                │
  │   LOST ──(re-detected)────────────► TRACK               │
  │   LOST ──(stop_focus / timeout)───► IDLE                │
  └─────────────────────────────────────────────────────────┘

Pipeline dùng GateState + FocusState kết hợp để quyết định:
  - Có chạy AI không (_ai_due)
  - Chạy AI theo luồng nào (run_for_state trong engine)
  - Có update motion detector không
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Trạng thái cổng AI (AI Gate)
# ============================================================

class GateState(str, Enum):
    """Kiểm soát việc có được phép đẩy frame vào hàng đợi AI không."""
    IDLE = "idle"       # Đang chờ chuyển động hợp lệ
    ACTIVE = "active"   # Đang chạy AI (có người / đang focus)


# ============================================================
# Trạng thái Focus
# ============================================================

class FocusState(str, Enum):
    """Trạng thái bám theo một mục tiêu cụ thể."""
    IDLE = "idle"           # Không focus
    ACQUIRE = "acquire"     # Đang xác nhận mục tiêu (zoom dần)
    TRACK = "track"         # Đang bám chắc mục tiêu
    LOST = "lost"           # Mất mục tiêu, đang thăm dò tìm lại


# ============================================================
# Trạng thái tổng hợp của một nguồn (source)
# ============================================================

@dataclass
class AppState:
    """Snapshot trạng thái đầy đủ của một nguồn tại một thời điểm.

    Pipeline và engine dùng object này để quyết định luồng xử lý.
    """
    gate: GateState = GateState.IDLE
    focus: FocusState = FocusState.IDLE

    # --- thông tin bổ sung để engine ra quyết định ---
    focus_mode: str = "eco"         # "eco" | "full"
    is_moving: bool = False         # motion detector đang báo có chuyển động
    has_person: bool = False        # AI đã thấy người trong khoảng thời gian gần

    @property
    def focusing(self) -> bool:
        """Đang focus và chưa mất mục tiêu."""
        return self.focus in (FocusState.ACQUIRE, FocusState.TRACK)

    @property
    def ai_active(self) -> bool:
        """AI gate đang mở."""
        return self.gate == GateState.ACTIVE

    def to_dict(self) -> dict:
        return {
            "gate": self.gate.value,
            "focus": self.focus.value,
            "focus_mode": self.focus_mode,
            "is_moving": self.is_moving,
            "has_person": self.has_person,
        }


# ============================================================
# Định nghĩa các luồng xử lý AI tương ứng với trạng thái
# ============================================================

class AIRunMode(str, Enum):
    """Luồng AI engine sẽ thực thi.

    Pipeline xây dựng AppState rồi gọi AIRunMode.from_state(state)
    để lấy mode, sau đó truyền vào engine.run_for_state().
    """
    NORMAL = "normal"           # Detect + ByteTrack + nhận diện mặt đầy đủ
    FOCUS_ECO = "focus_eco"     # YOLO nhỏ toàn khung, không ByteTrack, mặt tần suất thấp
    FOCUS_FULL = "focus_full"   # YOLO 640 + ByteTrack, mặt bình thường (chế độ full)
    PROBE = "probe"             # Thăm dò tìm lại mục tiêu khi LOST (nhẹ, tần suất thấp)
    SKIP = "skip"               # Không chạy AI (gate đóng, không đủ điều kiện)

    @classmethod
    def from_state(cls, state: AppState) -> "AIRunMode":
        """Ánh xạ AppState → AIRunMode.

        Quy tắc (ưu tiên từ trên xuống):
        1. Gate đóng + không focus → SKIP
        2. Focus LOST → PROBE  (gate có thể đóng, vẫn thăm dò)
        3. Focus ACQUIRE hoặc TRACK + eco → FOCUS_ECO
        4. Focus ACQUIRE hoặc TRACK + full → FOCUS_FULL
        5. Gate mở (bình thường) → NORMAL
        """
        if state.focus == FocusState.LOST:
            return cls.PROBE

        if not state.ai_active:
            return cls.SKIP

        if state.focusing:
            return cls.FOCUS_ECO if state.focus_mode == "eco" else cls.FOCUS_FULL

        return cls.NORMAL


# ============================================================
# Bộ chuyển trạng thái (Transition helpers)
# ============================================================

class GateTransition:
    """Tập hợp các hàm chuyển trạng thái của AI Gate.

    Pipeline gọi các method này thay vì tự thao tác biến nội bộ,
    giúp tập trung logic chuyển trạng thái vào một chỗ.
    """

    @staticmethod
    def try_open(gate: GateState, streak: int, confirm: int,
                 refractory_ok: bool) -> GateState:
        """IDLE → ACTIVE nếu đủ điều kiện chuyển động liên tục."""
        if gate == GateState.IDLE and streak >= confirm and refractory_ok:
            return GateState.ACTIVE
        return gate

    @staticmethod
    def close(gate: GateState) -> GateState:
        """ACTIVE → IDLE (hết người hoặc báo động giả)."""
        return GateState.IDLE

    @staticmethod
    def force_open() -> GateState:
        """Buộc mở gate (dùng khi bắt đầu focus hoặc chuyển về normal sau focus)."""
        return GateState.ACTIVE


class FocusTransition:
    """Tập hợp các hàm chuyển trạng thái của Focus."""

    @staticmethod
    def start() -> FocusState:
        """Người dùng bắt đầu focus → ACQUIRE."""
        return FocusState.ACQUIRE

    @staticmethod
    def confirm(current: FocusState) -> FocusState:
        """Xác nhận xong mục tiêu ACQUIRE → TRACK."""
        if current == FocusState.ACQUIRE:
            return FocusState.TRACK
        return current

    @staticmethod
    def lose(current: FocusState) -> FocusState:
        """ACQUIRE/TRACK → LOST khi mất mục tiêu."""
        if current in (FocusState.ACQUIRE, FocusState.TRACK):
            return FocusState.LOST
        return current

    @staticmethod
    def recover() -> FocusState:
        """LOST → TRACK khi tìm lại được mục tiêu."""
        return FocusState.TRACK

    @staticmethod
    def cancel() -> FocusState:
        """Huỷ focus (người dùng hoặc timeout) → IDLE."""
        return FocusState.IDLE


# ============================================================
# Utility: mô tả trạng thái cho logging / UI
# ============================================================

_GATE_LABELS = {
    GateState.IDLE: "⏸ Chờ chuyển động",
    GateState.ACTIVE: "▶ Đang quét",
}

_FOCUS_LABELS = {
    FocusState.IDLE: "",
    FocusState.ACQUIRE: "🔍 Xác nhận mục tiêu",
    FocusState.TRACK: "🎯 Đang bám",
    FocusState.LOST: "❓ Mất mục tiêu",
}

_RUN_MODE_LABELS = {
    AIRunMode.NORMAL: "Normal",
    AIRunMode.FOCUS_ECO: "Focus-Eco",
    AIRunMode.FOCUS_FULL: "Focus-Full",
    AIRunMode.PROBE: "Probe",
    AIRunMode.SKIP: "Skip",
}


def describe_state(state: AppState) -> str:
    """Trả về chuỗi mô tả ngắn gọn để log / debug."""
    gate_lbl = _GATE_LABELS.get(state.gate, state.gate.value)
    focus_lbl = _FOCUS_LABELS.get(state.focus, "")
    mode = AIRunMode.from_state(state)
    mode_lbl = _RUN_MODE_LABELS.get(mode, mode.value)
    parts = [gate_lbl, f"[{mode_lbl}]"]
    if focus_lbl:
        parts.append(focus_lbl)
    return " | ".join(parts)
