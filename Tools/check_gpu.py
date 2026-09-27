import openvino as ov
import os


def diagnose_hardware():
    print("=== CHẨN ĐOÁN THIẾT BỊ OPENVINO ===")
    core = ov.Core()

    # 1. Liệt kê tất cả các thiết bị OpenVINO nhận diện được
    devices = core.available_devices
    print(f"Các thiết bị được OpenVINO nhận diện: {devices}")

    has_gpu = False
    for device in devices:
        try:
            name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f" - {device}: {name}")
            if 'GPU' in device:
                has_gpu = True
        except Exception as e:
            print(f" - Lỗi khi đọc thông tin thiết bị {device}: {e}")

    print("\n=== KIỂM TRA QUYỀN TRUY CẬP (LINUX) ===")
    # Trên Linux, GPU tích hợp Intel thường nằm ở /dev/dri/renderD128
    gpu_path = "/dev/dri/renderD128"

    if os.name == 'posix':  # Chỉ chạy trên Linux/Mac
        if os.path.exists("/dev/dri"):
            print("Tìm thấy thư mục đồ họa /dev/dri:")
            os.system("ls -l /dev/dri")

            if os.path.exists(gpu_path):
                # Kiểm tra quyền đọc/ghi vào thiết bị
                r_ok = os.access(gpu_path, os.R_OK)
                w_ok = os.access(gpu_path, os.W_OK)
                print(f"\nPhân quyền truy cập {gpu_path} cho user hiện tại:")
                print(f" - Quyền đọc (Read): {'CÓ' if r_ok else 'KHÔNG'}")
                print(f" - Quyền ghi (Write): {'CÓ' if w_ok else 'KHÔNG'}")

                if not (r_ok and w_ok):
                    print("\n[CẢNH BÁO]: User của bạn không có quyền đọc/ghi vào GPU!")
                    print("=> Đó là lý do OpenVINO phải nạp mô hình vào CPU.")
            else:
                print(f"\n[CẢNH BÁO]: Không tìm thấy thiết bị {gpu_path}. Có thể driver Intel chưa được cài.")
        else:
            print("\n[CẢNH BÁO]: Không tìm thấy thư mục /dev/dri. Kernel Linux có thể chưa nhận iGPU.")
    else:
        print("Hệ điều hành Windows: Không kiểm tra quyền qua /dev/dri.")


if __name__ == '__main__':
    diagnose_hardware()