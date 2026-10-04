import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from BusinessLayer.camera_manager import CameraStream
from BusinessLayer.Inference.inference_engine import InferenceEngine
from DataLayer.vector_db import FaceDatabase


class SmartVisionPipeline:
    def __init__(self, yolo_path, face_dir, camera_sources):
        self.engine = InferenceEngine(yolo_path, face_dir)
        self.db = FaceDatabase()
        self.cameras = []

        self.ai_interval = 1.0 / 10.0  # 10 FPS cho AI
        self.last_ai_time = {}
        self.ai_cache = {}
        self.ai_latency = {}
        self.ai_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=len(camera_sources))

        # --- BIẾN LƯU TRỮ KHUNG HÌNH CHO WEB ---
        self.latest_frames = {}
        self.running = False

        for i, source in enumerate(camera_sources):
            cam_id = f"Cam_{i + 1}"
            cam = CameraStream(camera_id=cam_id, source=source)
            self.cameras.append(cam)

            self.last_ai_time[cam_id] = 0.0
            self.ai_cache[cam_id] = []
            self.ai_latency[cam_id] = 0.0
            self.ai_tasks[cam_id] = None
            self.latest_frames[cam_id] = None

    def start(self):
        """Khởi động toàn bộ camera và chạy vòng lặp xử lý ngầm."""
        self.running = True
        for cam in self.cameras:
            cam.start()

        # Tách vòng lặp chính ra một thread ngầm để không block FastAPI
        self.main_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.main_thread.start()
        print("=> AI Pipeline đang chạy ngầm...")

    def _run_loop(self):
        while self.running:
            loop_start = time.perf_counter()

            for cam in self.cameras:
                frame, has_motion = cam.read()
                if frame is not None:
                    self._process_and_encode(cam.camera_id, frame, has_motion)

            # Nghỉ ngắn để nhường CPU nếu xử lý quá nhanh
            elapsed = time.perf_counter() - loop_start
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)

    def _process_and_encode(self, camera_id, frame, has_motion):
        current_time = time.perf_counter()

        # 1. Thu thập kết quả AI ngầm
        if self.ai_tasks[camera_id] is not None and self.ai_tasks[camera_id].done():
            self.ai_cache[camera_id], self.ai_latency[camera_id] = self.ai_tasks[camera_id].result()
            self.ai_tasks[camera_id] = None

        # 2. Kích hoạt AI nếu đủ điều kiện
        if has_motion and (current_time - self.last_ai_time[camera_id] >= self.ai_interval) and self.ai_tasks[
            camera_id] is None:
            self.last_ai_time[camera_id] = current_time
            self.ai_tasks[camera_id] = self.executor.submit(self._run_ai, frame.copy())
        elif not has_motion and self.ai_tasks[camera_id] is None:
            self.ai_cache[camera_id] = []

        # 3. Vẽ Bounding Boxes
        for draw_data in self.ai_cache[camera_id]:
            color = draw_data["color"]
            if "body_box" in draw_data:
                bx1, by1, bx2, by2 = draw_data["body_box"]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            if "face_box" in draw_data:
                fx1, fy1, fx2, fy2 = draw_data["face_box"]
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                cv2.putText(frame, draw_data["label"], (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"AI Latency: {self.ai_latency[camera_id]:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # 4. CHUYỂN ĐỔI ẢNH SANG JPEG (Chuẩn bị xuất lên Web)
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            self.latest_frames[camera_id] = buffer.tobytes()

    def _run_ai(self, frame):
        start_time = time.perf_counter()
        draw_list = []
        persons = self.engine.detect_persons(frame)
        for p in persons:
            person_data = {"body_box": p["box"], "color": (150, 150, 150)}
            faces = self.engine.process_faces(frame, p["box"])
            for face in faces:
                name = self.db.recognize(face["embedding"])
                person_data.update({
                    "face_box": face["abs_box"],
                    "label": name if name != "Unknown" else "NGUOI LA!",
                    "color": (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                })
            draw_list.append(person_data)
        return draw_list, (time.perf_counter() - start_time) * 1000

    def get_frame(self, camera_id):
        """API lấy frame JPEG mới nhất cho FastAPI."""
        return self.latest_frames.get(camera_id)

    def stop(self):
        self.running = False
        for cam in self.cameras: cam.stop()
        self.executor.shutdown(wait=False)0