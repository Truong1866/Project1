from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import time
from BusinessLayer.pipeline import SmartVisionPipeline

app = FastAPI(title="Smart Vision Dashboard")
templates = Jinja2Templates(directory="PresentLayer/templates")

# Khởi tạo Pipeline ở mức Global
YOLO_PATH = "Models/yolov8n.pt"
FACE_DIR = "Models/face_models/"
CAMERA_SOURCES = [0] # Thêm RTSP URL vào đây nếu có 2-3 camera

pipeline = SmartVisionPipeline(YOLO_PATH, FACE_DIR, CAMERA_SOURCES)

@app.on_event("startup")
def startup_event():
    pipeline.start()

@app.on_event("shutdown")
def shutdown_event():
    pipeline.stop()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Trả về trang Web và danh sách tên các camera
    cam_ids = [cam.camera_id for cam in pipeline.cameras]

    # Chỉ định rõ ràng request, name và context cho phiên bản FastAPI mới
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"cameras": cam_ids}
    )

def generate_video_stream(camera_id: str):
    """Hàm Generator liên tục phát ảnh JPEG cho trình duyệt (MJPEG)."""
    while True:
        frame_bytes = pipeline.get_frame(camera_id)
        if frame_bytes:
            # Định dạng Multipart cho stream video trên trình duyệt
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # ~30 FPS

@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str):
    return StreamingResponse(generate_video_stream(camera_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")