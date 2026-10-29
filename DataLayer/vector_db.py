from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from Utils.logger import get_logger

log = get_logger("FaceDB")

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "known_faces.json"
UNKNOWN = "Unknown"


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceDatabase:
    MAX_SAMPLES = 8

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, threshold: float = 0.45):
        self.db_path = Path(db_path)
        # Đảm bảo db_path kết thúc bằng .json (hỗ trợ truyền vào .pkl cũ)
        if self.db_path.suffix == ".pkl":
            self.db_path = self.db_path.with_suffix(".json")
        self.threshold = threshold
        self._lock = threading.RLock()
        self.known_faces: dict[str, list[np.ndarray]] = {}
        self._names: list[str] = []
        self._owners = np.zeros(0, dtype=np.int64)
        self._matrix = np.zeros((0, 512), dtype=np.float32)
        self._load_db()

    # ------------------------------------------------------------------ lưu / tải
    def _npz_path(self) -> Path:
        return self.db_path.with_suffix(".npz")

    def _load_db(self) -> None:
        json_path = self.db_path
        npz_path = self._npz_path()

        # --- Nạp từ định dạng mới (JSON + NPZ) ---
        if json_path.exists() and npz_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                npz = np.load(str(npz_path))
                for i, name in enumerate(meta.get("names", [])):
                    key = f"s{i}"
                    if key in npz:
                        arr = npz[key].astype(np.float32)
                        samples = [arr[j] for j in range(arr.shape[0])] if arr.ndim == 2 else [arr]
                        self.known_faces[name] = [_normalize(s) for s in samples][-self.MAX_SAMPLES:]
                log.info("Đã tải %d người quen từ CSDL (JSON+NPZ).", len(self.known_faces))
                self._rebuild()
                return
            except Exception:
                log.exception("Không đọc được %s + %s, thử tải từ pickle...", json_path, npz_path)
                self.known_faces.clear()

        # --- Tương thích ngược: tải từ pickle rồi chuyển sang định dạng mới ---
        pkl_path = self.db_path.with_suffix(".pkl")
        if pkl_path.exists():
            try:
                import pickle  # noqa: PLC0415
                with open(pkl_path, "rb") as f:
                    raw = pickle.load(f)  # noqa: S301 — chỉ dùng để di chuyển dữ liệu cũ
                for name, value in raw.items():
                    arr = np.asarray(value, dtype=np.float32)
                    samples = [arr] if arr.ndim == 1 else list(arr)
                    self.known_faces[name] = [_normalize(s) for s in samples][-self.MAX_SAMPLES:]
                log.info(
                    "Đã tải %d người quen từ pickle cũ. Đang chuyển sang JSON+NPZ...",
                    len(self.known_faces),
                )
                self._rebuild()
                self.save_db()  # lưu ngay sang định dạng mới
                log.info(
                    "Chuyển đổi xong. File pickle gốc được giữ lại tại '%s' (có thể xoá thủ công).",
                    pkl_path,
                )
                return
            except Exception:
                log.exception("Không đọc được pickle '%s', bắt đầu với CSDL trống", pkl_path)
                self.known_faces.clear()
        else:
            log.info("CSDL khuôn mặt trống. Hãy đăng ký người quen!")

        self._rebuild()

    def save_db(self) -> None:
        with self._lock:
            os.makedirs(self.db_path.parent, exist_ok=True)
            names = list(self.known_faces.keys())
            counts = [len(self.known_faces[n]) for n in names]
            meta = {"names": names, "counts": counts}

            # Ghi JSON (index)
            tmp_json = self.db_path.with_suffix(".json.tmp")
            tmp_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_json, self.db_path)

            # Ghi NPZ (embeddings)
            npz_data = {}
            for i, name in enumerate(names):
                samples = self.known_faces[name]
                npz_data[f"s{i}"] = np.stack(samples, axis=0) if samples else np.zeros((0, 512), np.float32)
            tmp_npz = self._npz_path().with_suffix(".npz.tmp")
            np.savez(str(tmp_npz), **npz_data)
            os.replace(tmp_npz, self._npz_path())

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
