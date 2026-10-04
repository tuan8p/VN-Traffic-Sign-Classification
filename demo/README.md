# Module Demo: Nhận Diện & Phân Loại Biển Báo Giao Thông Việt Nam

Ứng dụng web trực quan phục vụ kiểm thử, đánh giá hiệu năng và so sánh thực nghiệm giữa 3 pipeline mô hình (Deep Learning, SVM, Gradient Boosting) trên bộ dữ liệu [Vietnamese Traffic Signs](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).

---

## 1. Danh sách Chức năng của UI

### 1.1. Thu thập & Nạp dữ liệu (Data Acquisition)
* **File Upload:** Hỗ trợ tải lên trực tiếp các tệp ảnh định dạng phổ biến (`JPG`, `JPEG`, `PNG`).
* **Preset Gallery (Bộ ảnh mẫu có sẵn):** Cung cấp sẵn các mẫu biển báo phổ biến (Biển cấm, Biển nguy hiểm, Biển hiệu lệnh) phục vụ việc demo nhanh khi báo cáo mà không cần chuẩn bị tệp cục bộ.
* **Live Capture (Webcam Feed):** Chụp trực tiếp hình ảnh từ thiết bị ghi hình (webcam) để kiểm thử phản xạ mô hình trong điều kiện thực tế.

### 1.2. Quản trị & Điều hướng Mô hình (Model Orchestration)
* **Single Pipeline Inference:** Lựa chọn kiểm thử chuyên sâu từng kiến trúc:
  * **Deep Learning:** EfficientNet / ResNet / Vision Transformer.
  * **Machine Learning 1:** Support Vector Machine (SVM) kết hợp trích xuất đặc trưng (HOG / PCA).
  * **Machine Learning 2:** Gradient Boosting (XGBoost / LightGBM / CatBoost).
* **Comparative Benchmark (So sánh 3 Pipeline song song):**
  * Kích hoạt chế độ kiểm thử đối đầu trên cùng một ảnh đầu vào.
  * Xuất bảng chỉ số so sánh tức thời: Dự đoán Top-1, độ tin cậy và độ trễ xử lý (Latency) giữa 3 phương pháp.
* **Dynamic Checkpoint Path:** Cho phép nạp linh hoạt các file trọng số xuất xưởng (`.pth`, `.onnx`, `.pkl`, `.joblib`) từ các pipeline huấn luyện khác nhau.
* **Top-K Selector:** Thanh trượt điều chỉnh số lượng ứng viên nhận diện hiển thị (từ 1 đến 8).

### 1.3. Phân tích & Trực quan hóa Kết quả (Inference Analytics)
* **Metric Cards:** Hiển thị thẻ chỉ số trực quan gồm: Nhãn biển báo xác suất cao nhất (Top-1), Độ tin cậy (Confidence Score %) và Thời gian suy luận (Latency tính theo milli-giây).
* **Phân phối Xác suất (Probability Distribution Chart):** Biểu đồ thanh ngang tương tác (Plotly Bar Chart) thể hiện trực quan mức độ tin cậy và sự phân vân của mô hình trên các lớp Top-K.

---

## 2. Hướng dẫn Thiết lập & Khởi chạy Demo

### Yêu cầu Tiên quyết
* Python 3.9+
* Môi trường ảo (khuyến khích dùng `venv` hoặc `conda`)

### Bước 1: Khởi tạo & Kích hoạt Môi trường ảo
```bash
# Di chuyển về thư mục gốc của project
cd VN-Traffic-Sign-Classification

# Tạo môi trường ảo
python3 -m venv venv

# Kích hoạt trên Linux/macOS/WSL
source venv/bin/activate

# Kích hoạt trên Windows PowerShell
# .\venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt Thư viện Phụ thuộc
```bash
pip install -r requirements.txt
# Hoặc cài đặt các gói cần thiết cho demo:
pip install streamlit pillow pandas plotly
```

### Bước 3: Khởi chạy Ứng dụng Streamlit
```bash
streamlit run demo/app.py
```
### Bước 4: Truy cập Giao diện
Sau khi chạy thành công, mở trình duyệt và truy cập:

Local URL: http://localhost:8501