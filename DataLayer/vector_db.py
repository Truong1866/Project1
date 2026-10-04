from pathlib import Path
import numpy as np
import os
import pickle

current_file_path = Path(__file__).resolve()
data_dir = current_file_path.parent.parent / 'DataLayer'

class FaceDatabase:
    def __init__(self, db_path=data_dir / "known_faces.pkl", threshold=0.45):
        self.db_path = db_path
        self.threshold = threshold  # Ngưỡng độ giống nhau (0.45 là chuẩn cho InsightFace)
        self.known_faces = {}
        self._load_db()

    def _load_db(self):
        """Tải dữ liệu khuôn mặt từ ổ cứng nếu có."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.known_faces = pickle.load(f)
            print(f"[FaceDB] Đã tải {len(self.known_faces)} người quen từ CSDL.")
        else:
            print("[FaceDB] CSDL trống. Hãy thêm người quen mới!")

    def save_db(self):
        """Lưu dữ liệu xuống ổ cứng."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.known_faces, f)

    def add_person(self, name, embedding):
        """Thêm một người quen vào cơ sở dữ liệu."""
        self.known_faces[name] = embedding
        self.save_db()
        print(f"[FaceDB] Đã lưu khuôn mặt của: {name}")

    def recognize(self, embedding):
        """So sánh vector truyền vào với CSDL để tìm người giống nhất."""
        if not self.known_faces:
            return "Unknown"  # Không có ai trong CSDL thì mặc định là Lạ

        best_match_name = "Unknown"
        highest_similarity = -1.0

        for name, known_emb in self.known_faces.items():
            # Tính độ tương đồng Cosine (Cosine Similarity)
            # Công thức: Tích vô hướng / (Độ dài vector 1 * Độ dài vector 2)
            dot_product = np.dot(embedding, known_emb)
            norm_1 = np.linalg.norm(embedding)
            norm_2 = np.linalg.norm(known_emb)
            similarity = dot_product / (norm_1 * norm_2)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = name

        # Nếu độ giống nhau vượt qua ngưỡng cho phép, kết luận là Người quen
        if highest_similarity >= self.threshold:
            return best_match_name
        else:
            return "Unknown"