import openvino as ov
from ultralytics import YOLO

core = ov.Core()
devices = core.available_devices

print("=== THÔNG TIN PHẦN CỨNG HỖ TRỢ AI ===")
for device in devices:
    name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"- Thiết bị khả dụng: {device} ({name})")

print("\n=== THÔNG TIN YOLO ===")
model = YOLO('Model/yolov8n.pt')
print("Khởi tạo YOLOv8 thành công!")