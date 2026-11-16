# VN Traffic Sign Classification

Hệ thống phân loại **52 lớp biển báo giao thông Việt Nam** trên bộ dữ liệu [VNTS](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs), cho phép so sánh công bằng ba hướng tiếp cận trên cùng quy trình tiền xử lý, chia tập chống rò rỉ và bộ chỉ số đánh giá thống nhất:

1. **LightGBM** — Gradient Boosting trên vector đặc trưng thủ công 2020 chiều (HOG + LBP + histogram màu HSV).
2. **SVM (RBF)** — Support Vector Classifier trên cùng vector đặc trưng 2020 chiều.
3. **EfficientNetV2-B0** — Mạng nơ-ron tích chập tinh chỉnh từ trọng số tiền huấn luyện trên ảnh RGB `224×224`.

---

## Cấu trúc thư mục

```text
VN-Traffic-Sign-Classification/
├── analysis/             # Kịch bản phân tích dữ liệu (EDA) và biểu đồ thống kê
├── configs/
│   ├── shared.yaml       # Cấu hình dùng chung (seed, tỉ lệ chia tập, đặc trưng, augment, metrics)
│   ├── classes.csv       # Danh mục 52 lớp biển báo, cặp đối xứng gương và nhóm dễ nhầm
│   ├── pipelines/        # Cấu hình riêng cho boosting.yaml, svm.yaml, dl.yaml
│   └── runtime/          # Cấu hình phần cứng (local.yaml, kaggle_2xt4.yaml)
├── data/                 # Thư mục chứa dữ liệu gốc (raw/) và dữ liệu đã xử lý (processed/)
├── demo/                 # Ứng dụng web Streamlit và động cơ suy luận hợp nhất
├── manual_test_images/   # Ảnh thử nghiệm bổ sung ngoài tập dữ liệu chuẩn
├── outputs/              # Nơi lưu kết quả các lần chạy (metrics, checkpoint, hình ảnh)
├── saved_models/         # Trọng số và metadata phục vụ chạy demo trực tiếp
├── tools/                # Giao diện dòng lệnh (preprocess, train, eval, pack, plot)
├── vn_tsc/               # Thư viện lõi (data, features, pipelines, eval, runtime, utils)
├── .env.example          # Mẫu biến môi trường (DATA_ROOT, WANDB_API_KEY)
├── pyproject.toml        # Khai báo gói cài đặt và entry points
└── requirements.txt      # Danh sách thư viện phụ thuộc
```

---

## Yêu cầu hệ thống & Cài đặt

- **Python ≥ 3.10**
- **Phần cứng:** CPU đủ cho tiền xử lý, EDA, SVM, LightGBM và chạy demo; khuyến nghị có GPU khi huấn luyện EfficientNetV2-B0 (hỗ trợ đa GPU qua DDP).

Khởi tạo môi trường ảo và cài đặt thư viện từ thư mục gốc repository:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Nếu sử dụng Weights & Biases hoặc đặt đường dẫn dữ liệu mặc định, tạo tệp `.env` từ mẫu:

```bash
cp .env.example .env
# Điền WANDB_API_KEY=... và DATA_ROOT=... (thư mục chứa dataset VNTS định dạng YOLO)
```

---

## Hướng dẫn chạy

### 1. Tiền xử lý offline (dùng chung cho cả 3 pipeline)

Chuyển đổi các bounding box YOLO của VNTS thành ảnh crop `224×224` RGB (nới lề 15%, loại box có cạnh ngắn dưới 12 px, letterbox với giá trị đệm 114), gom cụm chống rò rỉ bằng mã dHash 256-bit, chia tập theo cụm `70% / 15% / 15%`, tăng cường dữ liệu offline trên tập `train` và trích xuất ma trận đặc trưng `64×64` (2020 chiều):

```bash
python -m tools.run_preprocess --data-root <duong-dan-toi-VNTS> --out data/processed
```

Đầu ra mặc định tại `data/processed/`:

- `images/X_{train,val,test,train_aug}.npy`, `images/y_*.npy`
- `features/F_{train,val,test,train_aug}.npy`, `features/scaler.joblib`, `features/feature_report.json`
- `metadata.csv`, `class_table.csv`, `preprocess_report.json`

Tùy chọn hữu ích:

- `--skip-crops`: giữ nguyên kho ảnh crop, chỉ trích xuất lại đặc trưng cổ điển.
- `--no-augment`: bỏ qua bước sinh dữ liệu tăng cường offline.

### 2. Phân tích dữ liệu (EDA, không cần W&B)

```bash
python -m analysis.eda --processed data/processed --out analysis/figures
```

Chi tiết các biểu đồ và bảng tóm tắt xem tại [analysis/README.md](analysis/README.md).

### 3. Huấn luyện mô hình

Mặc định các lệnh huấn luyện ghi log lên Weights & Biases. Nếu chạy cục bộ không dùng W&B, thêm cờ `--no-wandb` (hoặc đặt `WANDB_MODE=offline`).

```bash
# LightGBM
python -m tools.run_train --pipeline boosting --data-root data/processed

# SVM (RBF)
python -m tools.run_train --pipeline svm --data-root data/processed

# EfficientNetV2-B0 (1 GPU cục bộ)
python -m tools.run_train --pipeline dl --runtime configs/runtime/local.yaml --data-root data/processed

# EfficientNetV2-B0 (2 GPU qua torchrun)
torchrun --nproc_per_node=2 -m tools.run_train --pipeline dl \
  --config configs/pipelines/dl.yaml \
  --runtime configs/runtime/kaggle_2xt4.yaml \
  --data-root data/processed
```

Mỗi lần chạy sẽ lưu đầy đủ `resolved_config.yaml`, `metrics.json`, `history.json`, `figures/` và `checkpoints/` vào `outputs/<pipeline>/<run_id>/`.

### 4. Đánh giá lại, vẽ đồ thị & đóng gói

```bash
# Đánh giá một checkpoint đã lưu trên tập val hoặc test
python -m tools.run_eval --pipeline dl --run-dir outputs/dl/<run_id> --split test

# Đóng gói kết quả một lần chạy thành file zip
python -m tools.pack_outputs --run-dir outputs/boosting/<run_id>

# Vẽ lại đường cong huấn luyện từ history.json
python -m tools.plot_from_json --history outputs/dl/<run_id>/history.json --out-dir outputs/dl/<run_id>/figures
```

### 5. Chạy ứng dụng Demo (Streamlit)

Đảm bảo thư mục `saved_models/` chứa đầy đủ các tệp trọng số (`checkpoint_best.pt`, `svm.joblib`, `boosting.joblib`, `scaler.joblib`, `class_table.csv` — xem hướng dẫn chi tiết trong [demo/README.md](demo/README.md)), sau đó khởi chạy từ thư mục gốc repository:

```bash
streamlit run demo/app.py
```

---

## Lưu ý quan trọng

1. **Cố định tập chia và không dùng phép lật ngang (`hflip: false`):** Bộ VNTS chứa 5 cặp lớp đối xứng gương (ví dụ cấm rẽ trái/cấm rẽ phải, ngoặt trái/ngoặt phải) và nhiều khung hình video liên tiếp. Luôn dùng chung đầu ra của `tools.run_preprocess` để bảo toàn tính công bằng và không rò rỉ dữ liệu giữa `train`, `val`, `test`.
2. **Tiêu chí chọn mô hình:** Dữ liệu mất cân bằng mạnh (tỉ lệ 357:1, lớp 44 chỉ có 3 mẫu và không có trong tập `test`). Mọi cấu hình được chọn dựa trên **macro-F1 trên tập `val`**; tập `test` chỉ dùng để đánh giá cuối cùng.
3. **Đầu vào của Demo / Inference:** Ứng dụng demo không tích hợp bộ phát hiện đối tượng (YOLO); ảnh tải lên phải là **ảnh crop chứa một biển báo**.
4. **Độ tương thích thư viện:** Tệp `scaler.joblib` và `svm.joblib` được lưu với `scikit-learn==1.6.1`. Nên cài đúng phiên bản trong `requirements.txt` để tránh cảnh báo không tương thích khi nạp lại mô hình.
5. **Điểm tin cậy của SVM:** Mô hình SVM cấu hình mặc định tắt `probability` nên điểm hiển thị trong demo là decision margin đã chuẩn hóa theo softmax để xếp hạng, không phải xác suất hiệu chỉnh.

---

## Kiểm tra định dạng mã nguồn

```bash
ruff check vn_tsc tools demo analysis
```

## Giấy phép

MIT — xem [LICENSE](LICENSE).
