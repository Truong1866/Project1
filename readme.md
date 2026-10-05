# Hệ Thống Camera Giám Sát Thông Minh (Smart Camera Surveillance)

--------------------
## ⚠️ Cảnh Báo:
Repository này được phát triển, thử nghiệm và tối ưu hóa dựa trên cấu hình máy tính cá nhân sử dụng iGPU Intel Iris Graphic Xe (không hỗ trợ cuda). Đối với các cấu hình:
+ Có iGPU không giống với repository.
+ GPU hỗ trợ cuda.

Sẽ có khả năng không hoạt động khi đó hãy xem các file [yolo_native.py](BusinessLayer/inference/yolo_native.py) và [face_recognizer.py](BusinessLayer/Inference/face_recognizer.py) để chỉnh sửa cho hợp lý.

___
## Mục Lục
+ [Giới thiệu](#Giới-thiệu)
+ [Tính năng chính](#tính-năng-chính)
+ [Yêu cầu hệ thống](#Yêu-cầu-hệ-thống)
+ [Hướng dẫn cài đặt và khởi chạy](#Hướng-dẫn-cài-đặt-và-khởi-chạy)
+ [Cấu trúc thư mục](#Cấu-trúc-thư-mục)
+ [Giấy phép (License)](#Giấy-phép-(License))

---
## Giới thiệu

Đây là một dự án phần mềm giám sát video thông minh được viết bằng Python. Dự án tích hợp các công nghệ Computer Vision tiên tiến để phân tích luồng video từ camera trong thời gian thực. Hệ thống được tổ chức theo kiến trúc đa tầng (Layered Architecture) bao gồm Business Layer, Data Layer, và Present Layer giúp mã nguồn dễ dàng mở rộng và bảo trì.

---
##  Tính năng chính

### Dựa trên kiến trúc mã nguồn, hệ thống cung cấp các tính năng:
>Nhận diện đối tượng (Object Detection): Tích hợp YOLO (hỗ trợ cả OpenVINO).

>Nhận diện khuôn mặt (Face Recognition): Phát hiện và định danh các khuôn mặt quen thuộc.

>Phát hiện chuyển động (Motion Detection) & Theo dõi (Tracking): Theo dõi các chuyển động bất thường trong khung hình.

>Thông báo tự động (Discord Notifier): Gửi cảnh báo thời gian thực qua Discord khi phát hiện sự kiện.

>Lưu trữ dữ liệu: Quản lý lịch sử sự kiện bằng SQLite và Vector DB.

>Giao diện người dùng: Cung cấp giao diện trực quan với Video Grid, Settings Panel và Event List.

---
## Yêu cầu hệ thống

OS: Windows / Linux
Python: Phiên bản 3.9 - 3.11 (Khuyến nghị 3.11)  
Git: Để clone repository  
Phần cứng: Khuyến nghị có GPU hoặc iGPU mạnh để xử lý inference các mô hình AI.

## Hướng dẫn cài đặt và khởi chạy
Dự án hiện không cung cấp các bản build release (như .exe). Để chạy phần mềm, bạn cần thiết lập môi trường và chạy trực tiếp từ mã nguồn theo các bước sau:

---
__Bước 1:__ Cài đặt cần thiết để có thể cho phép iGPU chạy được các model AI.

__LINUX__
+ Ubuntu/ Debian:
```bash
# Cập nhật danh sách package
sudo apt update

# Cài đặt các thư viện cần thiết
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free clinfo

# Thêm user vào nhóm render và video
sudo usermod -aG render $USER
```
+ Arch Linux / Manjaro / EndeavourOS
```bash
# Cài đặt OpenCL, Level Zero và Driver Media cho Intel
sudo pacman -S intel-compute-runtime oneapi-level-zero-gpu intel-media-driver clinfo

# Thêm user vào nhóm render và video
sudo usermod -aG render,video $USER
```
Sau khi chạy các lệnh trên thực hiện logout và chạy lệnh sau:
```bash
clinfo | grep "Device Name"
```
Nếu hiển thị tên igpu thì đã hoàn thành.

---------------------------
__WINDOW__
+ Cài đặt: Truy cập trang chủ Intel và tải/cài đặt Intel® Arc™ & Iris® Xe Graphics - Windows Driver mới nhất.
+ Bản driver này đã tích hợp sẵn cả OpenCL và Level Zero Compute Engine.
+ Sau khi cài đặt và khởi động lại, OpenVINO trong code Python của bạn (khi set thiết bị là "GPU") sẽ tự động nhận diện và offload khối lượng tính toán AI xuống iGPU mà không cần cấu hình gì thêm.

--------------------------
__Bước 2:__ Clone repository về máy
Mở terminal/command prompt và chạy lệnh:
```script
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

-------------------------------
__Bước 3:__ Tạo môi trường ảo (Virtual Environment) - Khuyến nghị
+ Tạo môi trường ảo có tên 'venv'
```script
python -m venv venv
```
+ Kích hoạt môi trường (Windows)
```script
venv\Scripts\activate
```
Kích hoạt môi trường (Linux/macOS)
```script
source venv/bin/activate
```

----------------
__Bước 4:__ Cài đặt các thư viện phụ thuộc
```script
pip install -r requirements.txt
```

-------------------
__Bước 5:__ Thiết lập cấu hình và mô hình AI
+ Đảm bảo bạn đã cấu hình các thông số cần thiết trong file config.yaml (ví dụ: Discord Webhook, nguồn camera...).  
+ Chạy script setup model nếu cần thiết (tải trọng số YOLO/Face recognition):  
```script
Tools/setup_models.py
```

-----------------
__Bước 6:__ Khởi chạy ứng dụng
```script
python main.py
```

---
## Cấu trúc thư mục
__Project__
   ├── BusinessLayer/: Chứa logic nghiệp vụ cốt lõi (Inference, Camera Manager, Pipeline, Notifier...).
   ├── DataLayer/: Xử lý lưu trữ và truy xuất dữ liệu (SQLite, Vector DB, Repositories).  
   ├──PresentLayer/: Giao diện người dùng (UI Components, Main Window).  
   ├──Tools/: Các công cụ hỗ trợ (Benchmark, Đăng ký khuôn mặt mới...).  
   ├──Utils/: Các hàm và class tiện ích (Event Bus, Logger, Config...).  
   ├──config.yaml: File cấu hình hệ thống.  
   └──main.py: Điểm khởi chạy của ứng dụng.  

---
## Giấy phép (License)

Dự án này được phân phối dưới giấy phép MIT License.  
Bạn có thể tự do sử dụng, sao chép, sửa đổi, gộp, xuất bản, phân phối, cấp phép lại và/hoặc bán các bản sao của Phần mềm, với điều kiện là thông báo bản quyền ở trên và thông báo cho phép này phải được bao gồm trong tất cả các bản sao hoặc các phần quan trọng của Phần mềm.  
Chi tiết vui lòng xem file LICENSE (nếu có) hoặc tham khảo MIT License.