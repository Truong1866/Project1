from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

import numpy as np

from Utils.logger import get_logger

log = get_logger("FaceDB")

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "known_faces.pkl"
UNKNOWN = "Unknown"


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceDatabase:
    MAX_SAMPLES = 8

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, threshold: float = 0.45):
        self.db_path = Path(db_path)
        self.threshold = threshold
        self._lock = threading.RLock()
        self.known_faces: dict[str, list[np.ndarray]] = {}
        self._names: list[str] = []
        self._owners = np.zeros(0, dtype=np.int64)
        self._matrix = np.zeros((0, 512), dtype=np.float32)
        self._load_db()

    # ------------------------------------------------------------------ lưu / tải
    def _load_db(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    raw = pickle.load(f)
                for name, value in raw.items():
                    arr = np.asarray(value, dtype=np.float32)
                    samples = [arr] if arr.ndim == 1 else list(arr)
                    self.known_faces[name] = [_normalize(s) for s in samples][-self.MAX_SAMPLES:]
                log.info("Đã tải %d người quen từ CSDL.", len(self.known_faces))
            except Exception:
                log.exception("Không đọc được %s, bắt đầu với CSDL trống", self.db_path)
        else:
            log.info("CSDL khuôn mặt trống. Hãy đăng ký người quen!")
        self._rebuild()

    def save_db(self) -> None:
        with self._lock:
            os.makedirs(self.db_path.parent, exist_ok=True)
            tmp = self.db_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.known_faces, f)
            os.replace(tmp, self.db_path)  # ghi nguyên tử, không hỏng file nếu tắt đột ngột

    def _rebuild(self) -> None:
        names, owners, rows = [], [], []
        for idx, (name, samples) in enumerate(self.known_faces.items()):
            names.append(name)
            for s in samples:
                owners.append(idx)
                rows.append(s)
        self._names = names
        self._owners = np.asarray(owners, dtype=np.int64)
        self._matrix = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 512), np.float32)

    # ------------------------------------------------------------------ quản lý
    def add_person(self, name: str, embedding) -> None:
        with self._lock:
            samples = self.known_faces.setdefault(name, [])
            samples.append(_normalize(embedding))
            del samples[:-self.MAX_SAMPLES]
            self._rebuild()
            self.save_db()
        log.info("Đã lưu khuôn mặt của: %s (%d mẫu)", name, len(self.known_faces[name]))

    def remove_person(self, name: str) -> bool:
        with self._lock:
            if name not in self.known_faces:
                return False
            del self.known_faces[name]
            self._rebuild()
            self.save_db()
            return True

    def list_people(self) -> list[tuple[str, int]]:
        with self._lock:
            return [(n, len(s)) for n, s in self.known_faces.items()]

    # ------------------------------------------------------------------ nhận diện
    def identify(self, embedding) -> tuple[str, float]:
        """Trả về (tên hoặc 'Unknown', điểm cosine cao nhất)."""
        with self._lock:
            if self._matrix.shape[0] == 0:
                return UNKNOWN, 0.0
            sims = self._matrix @ _normalize(embedding)           # (số_mẫu,)
            best = np.full(len(self._names), -1.0, dtype=np.float32)
            np.maximum.at(best, self._owners, sims)               # mẫu tốt nhất của từng người
            i = int(best.argmax())
            score = float(best[i])
            return (self._names[i], score) if score >= self.threshold else (UNKNOWN, score)

    def recognize(self, embedding) -> str:
        """API cũ: chỉ trả về tên."""
        return self.identify(embedding)[0]
