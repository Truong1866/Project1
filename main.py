from BusinessLayer.pipeline import SmartVisionPipeline

if __name__ == '__main__':
    # 1. Cấu hình đường dẫn model (từ bước setup)
    YOLO_PATH = "Models/yolov8n.pt"
    FACE_DIR = "Models/face_models/"
    CAMERA_SOURCES = [0]

    # Khởi tạo và chạy
    pipeline = SmartVisionPipeline(
        yolo_path=YOLO_PATH,
        face_dir=FACE_DIR,
        camera_sources=CAMERA_SOURCES
    )

    pipeline.run()