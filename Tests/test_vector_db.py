"""Unit tests cho FaceDatabase (JSON, NPZ và tương thích ngược PKL).

Chạy:
    .venv/bin/python -m unittest Tests/test_vector_db.py -v
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataLayer.vector_db import UNKNOWN, FaceDatabase


class TestFaceDatabase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_facedb_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_from_legacy_pickle(self):
        """Kiểm tra FaceDatabase tải được dữ liệu từ file .pkl cũ và tự chuyển đổi sang JSON+NPZ."""
        pkl_path = Path(self.test_dir) / "known_faces.pkl"
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        # Tạo file pickle cũ với 2 người
        legacy_data = {
            "Alice": [emb1, emb2],
            "Bob": np.random.randn(512).astype(np.float32),
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(legacy_data, f)

        # Khởi tạo FaceDatabase trỏ vào file .pkl
        db = FaceDatabase(pkl_path)

        # Kiểm tra đã tải đủ người và số mẫu
        people = dict(db.list_people())
        self.assertIn("Alice", people)
        self.assertEqual(people["Alice"], 2)
        self.assertIn("Bob", people)
        self.assertEqual(people["Bob"], 1)

        # Kiểm tra các file định dạng mới đã được tạo tự động
        json_path = pkl_path.with_suffix(".json")
        npz_path = pkl_path.with_suffix(".npz")
        self.assertTrue(json_path.exists())
        self.assertTrue(npz_path.exists())

        # Khởi tạo instance mới đọc từ file JSON+NPZ vừa tạo
        db2 = FaceDatabase(json_path)
        people2 = dict(db2.list_people())
        self.assertEqual(people, people2)

    def test_identify_person(self):
        """Kiểm tra nhận diện khuôn mặt khớp chính xác và trả về điểm cosine."""
        json_path = Path(self.test_dir) / "db.json"
        db = FaceDatabase(json_path, threshold=0.60)

        emb = np.random.randn(512).astype(np.float32)
        db.add_person("Charlie", emb)

        # Nhận diện chính vector đó
        name, score = db.identify(emb)
        self.assertEqual(name, "Charlie")
        self.assertGreater(score, 0.99)

        # Vector hoàn toàn ngẫu nhiên/ngược chiều
        other_emb = -emb
        name_unk, score_unk = db.identify(other_emb)
        self.assertEqual(name_unk, UNKNOWN)

    def test_add_and_remove_person(self):
        """Kiểm tra thêm và xoá người, dữ liệu được ghi đè an toàn."""
        json_path = Path(self.test_dir) / "db.json"
        db = FaceDatabase(json_path)

        for i in range(3):
            db.add_person("UserA", np.random.randn(512).astype(np.float32))

        self.assertEqual(dict(db.list_people())["UserA"], 3)

        # Xoá
        self.assertTrue(db.remove_person("UserA"))
        self.assertNotIn("UserA", dict(db.list_people()))

        # Reload lại từ đĩa để chắc chắn file đã lưu đúng
        db_reload = FaceDatabase(json_path)
        self.assertNotIn("UserA", dict(db_reload.list_people()))

    def test_max_samples_limit(self):
        """Kiểm tra không vượt quá MAX_SAMPLES (8 mẫu) mỗi người."""
        json_path = Path(self.test_dir) / "db.json"
        db = FaceDatabase(json_path)

        for _ in range(12):
            db.add_person("FrequentUser", np.random.randn(512).astype(np.float32))

        self.assertEqual(dict(db.list_people())["FrequentUser"], FaceDatabase.MAX_SAMPLES)


if __name__ == "__main__":
    unittest.main()

