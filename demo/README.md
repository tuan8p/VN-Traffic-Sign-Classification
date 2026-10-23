# Module Web Demo: Nhận Diện & Đánh Giá Biển Báo Giao Thông Việt Nam

Ứng dụng web trực quan (xây dựng trên nền tảng **Streamlit**) phục vụ kiểm thử, so sánh đối đầu thực nghiệm về độ chính xác và độ trễ suy luận giữa **3 pipeline mô hình** (Deep Learning, SVM, Gradient Boosting) trên bộ dữ liệu chuẩn hóa 52 lớp biển báo giao thông Việt Nam.

---

## 1. Tính năng nổi bật & Giao diện ứng dụng

### 1.1. Thu thập & Nạp dữ liệu đầu vào (Input Acquisition)
- **Cố định linh hoạt (Sticky Layout):** Khối nhập ảnh và thanh điều khiển được ghim cố định (`sticky`) ở cột bên trái, luôn bám theo màn hình khi cuộn trang để thuận tiện theo dõi cùng lúc với các biểu đồ và bảng kết quả so sánh.
- **Nguồn ảnh đa dạng:**
  - **Tải / Kéo thả ảnh:** Hỗ trợ nhận diện hầu hết các định dạng ảnh phổ biến (`PNG`, `JPG`, `JPEG`, `WEBP`, `BMP`, `TIFF`, `TIF`, `GIF`, `AVIF`, `ICO`).
  - **Thư viện ảnh mẫu (Preset Gallery):** Tự động quét và nạp các mẫu ảnh test thực tế từ thư mục `demo/sample_images/`, hiển thị kèm mã và tên tiếng Việt để phục vụ demo nhanh.
- **Điều chỉnh số lượng dự đoán (Top-K):** Tích hợp thanh trượt trực tiếp tại màn hình chính, cho phép chọn linh hoạt từ 1 đến 5 ứng viên có xác suất cao nhất.

### 1.2. Động cơ suy luận hợp nhất 3 Pipeline (Multi-Pipeline Inference)
Chạy suy luận song song đối đầu trên cùng một ảnh đầu vào:
1. **Deep Learning:** Mô hình mạng tích chập **EfficientNetV2-B0** (`checkpoint_best.pt`), thực hiện chuẩn hóa tensor RGB $224 \times 224$ và tính xác suất qua hàm Softmax.
2. **Bộ phân loại SVM:** Mô hình **Support Vector Machine (SVC)** (`svm.joblib`), trích xuất vector đặc trưng kết hợp 2020 chiều (HOG + LBP + HSV Color Histogram), tính điểm phân loại qua cơ chế biểu quyết cặp One-vs-One (OvO).
3. **Gradient Boosting:** Mô hình **LightGBM** (`boosting.joblib`), sử dụng vector đặc trưng 2020 chiều qua bộ chuẩn hóa `scaler.joblib`.
- **Ánh xạ nhãn tiếng Việt tự động:** Đọc và đồng bộ danh mục 52 lớp biển báo trực tiếp từ `class_table.csv`, hiển thị trực quan cả mã biển chuẩn (ví dụ: `P.131a`, `W.224`) kèm tên đầy đủ tiếng Việt (ví dụ: `Cấm đỗ xe`, `Đường người đi bộ cắt ngang`).

### 1.3. Trực quan hóa & So sánh hiệu năng
- **Thẻ kết quả gióng hàng đồng nhất:** 3 thẻ thông tin tương ứng với 3 mô hình được căn gióng hàng thẳng tuyệt đối, hiển thị:
  - Tên mô hình & kiến trúc.
  - Trạng thái thực thi (`Mô hình thực thi` hoặc `Mô phỏng dự phòng`).
  - Mã biển báo và tên tiếng Việt.
  - Độ tin cậy (Confidence %).
  - Thời gian suy luận (Latency tính bằng mili-giây trên CPU).
- **Biểu đồ so sánh Top-K:** Biểu đồ cột nhóm (Grouped Bar Chart qua Plotly) biểu diễn phân phối xác suất phân vân giữa các mô hình.
- **Biểu đồ đo lường độ trễ (Latency Profiling):** Biểu đồ thanh ngang so sánh trực quan tốc độ phản hồi thực tế (ms) giữa các kiến trúc.
- **Bảng chi tiết dự đoán:** Bảng tổng hợp kèm thanh tiến trình (`ProgressColumn`) biểu diễn chi tiết độ tin cậy của từng ứng viên.

---

## 2. Cấu trúc thư mục mô hình (`saved_models/`)

Để ứng dụng nạp trọng số thực thi thành công, thư mục `saved_models/` cần đảm bảo các tệp sau:

```text
saved_models/
├── checkpoint_best.pt       # Trọng số mô hình Deep Learning (EfficientNetV2-B0)
├── svm.joblib               # Mô hình SVM (Pipeline SVC OvO)
├── boosting.joblib          # Mô hình Gradient Boosting (LightGBM)
├── scaler.joblib            # StandardScaler chuẩn hóa 2020 đặc trưng (HOG + LBP + Color Hist)
└── class_table.csv          # Bảng từ điển 52 nhãn (cột sign_code và name_vi)
```

---

## 3. Hướng dẫn Cài đặt & Khởi chạy

### Yêu cầu tiên quyết
- Python 3.9+
- Môi trường ảo (`venv` hoặc `conda`)

### Bước 1: Khởi tạo và kích hoạt môi trường ảo
```bash
# Di chuyển vào thư mục gốc của repository
cd VN-Traffic-Sign-Classification

# Tạo môi trường ảo
python3 -m venv venv

# Kích hoạt môi trường ảo (Linux / macOS / WSL)
source venv/bin/activate

# Hoặc kích hoạt trên Windows PowerShell:
# .\venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install streamlit torch torchvision timm scikit-learn scikit-image lightgbm opencv-python pillow pandas plotly
```

### Bước 3: Khởi chạy ứng dụng Streamlit
```bash
python3 -m streamlit run demo/app.py
```

### Bước 4: Mở giao diện trên trình duyệt
Mở trình duyệt web và truy cập địa chỉ:
```text
http://localhost:8501
```