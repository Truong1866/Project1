from pathlib import Path
import cv2
from BusinessLayer.Inference.inference_engine import InferenceEngine
from DataLayer.vector_db import FaceDatabase


def register():
    name = input("Nhập tên người cần thêm (VD: Truong): ")

    print("Đang khởi động AI...")
    # Đảm bảo đường dẫn khớp với main.py của bạn
    current_file_path = Path(__file__).resolve()
    models_dir = current_file_path.parent.parent / 'Models'
    engine = InferenceEngine(models_dir / "yolov8n.pt", models_dir /"face_models/")
    db = FaceDatabase()

    cam = cv2.VideoCapture(0)
    print("Hãy nhìn thẳng vào camera và nhấn phím 's' để chụp/lưu. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cam.read()
        if not ret: break

        # Hiển thị camera để bạn căn chỉnh
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 1. Tìm người
            persons = engine.detect_persons(frame)
            if not persons:
                print("Không thấy ai trong khung hình. Thử lại!")
                continue

            # 2. Lấy người bự nhất (phòng khi có nhiều người)
            main_person = persons[0]

            # 3. Trích xuất khuôn mặt
            faces = engine.process_faces(frame, main_person["box"])
            if not faces:
                print("Không thấy rõ khuôn mặt. Thử lại!")
                continue

            # 4. Lưu vector đầu tiên tìm được
            face_vector = faces[0]["embedding"]
            db.add_person(name, face_vector)
            print("==> Đăng ký thành công! Nhấn 'q' để thoát.")

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register()