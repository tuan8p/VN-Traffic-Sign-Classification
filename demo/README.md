# Demo Streamlit — So sánh 3 pipeline

Ứng dụng web (**Streamlit**) để **suy luận và so sánh** trên cùng một ảnh biển báo:

1. **Deep Learning** — EfficientNetV2-B0 (`timm`), ảnh RGB 224×224, ImageNet normalize  
2. **SVM** — vector 2020 chiều (HOG + LBP + HSV histogram), hỗ trợ **OvO → xác suất qua voting**  
3. **Gradient Boosting** — LightGBM trên cùng vector + `StandardScaler`

Nhãn hiển thị: **mã biển + tên tiếng Việt** (52 lớp), đọc từ `class_table.csv`.

## Cấu trúc mã

| File | Vai trò |
| ------ | --------- |
| `app.py` | Giao diện: upload / gallery, Top-K, chạy 3 model, biểu đồ Plotly, bảng chi tiết |
| `inference.py` | `TrafficSignPredictor`: nạp weight, preprocess, predict, đo latency |
| `sample_images/` | Ảnh mẫu (quét tự động trong UI) |

Chạy app từ **root repo** (để `saved_models/` resolve đúng):

```bash
streamlit run demo/app.py
```

URL: [http://localhost:8501](http://localhost:8501)

## Cài đặt

```bash
cd VN-Traffic-Sign-Classification
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# hoặc tối thiểu demo + DL + ML:
# pip install -e ".[demo,dl,boosting]"
```

Phụ thuộc chính: `streamlit`, `plotly`, `torch`, `torchvision`, `timm`, `opencv-python-headless`, `scikit-image`, `scikit-learn`, `lightgbm`, `joblib`, `pandas`, `Pillow`.

## Thư mục `saved_models/` (artifact suy luận)

Mặc định engine tìm **`saved_models/`** ở root repo. Các file được **tự quét** theo tên:

```text
saved_models/
├── class_table.csv          # Khuyến nghị: 52 dòng (sign_code, name_vi, …)
├── scaler.joblib            # StandardScaler fit trên train (ML)
├── checkpoint_best.pt       # DL — ưu tiên (state_dict PyTorch)
│   # hoặc dl_model.pt
├── svm.joblib               # SVM (cũng thử: svm.pkl, svm_pipeline.joblib, …)
└── boosting.joblib          # LightGBM (cũng thử: model.joblib, boosting_pipeline.joblib, …)
```

### Copy từ kết quả train

| Pipeline | Nguồn gợi ý |
| ---------- | ------------- |
| DL | `outputs/dl/<run_id>/checkpoints/checkpoint_best.pt` → `saved_models/checkpoint_best.pt` |
| SVM | `outputs/svm/<run_id>/checkpoints/` (model + scaler) hoặc artifact team export |
| Boosting | `outputs/boosting/<run_id>/checkpoints/model.joblib` + `data/processed/features/scaler.joblib` → `scaler.joblib` |
| Nhãn | `data/processed/class_table.csv` → `saved_models/class_table.csv` |

Repo có thể kèm sẵn metadata JSON trong `saved_models/` (báo cáo test) — **không** thay thế file model.

## Hành vi UI

- **Hai cột:** nhập ảnh (trái) / kết quả & biểu đồ (phải).
- **Nguồn ảnh:** upload (PNG, JPG, WEBP, …) hoặc **Thư viện ảnh mẫu** (`demo/sample_images/`).
- **Top-K:** 1–5 lớp có xác suất cao nhất **mỗi** model.
- Nút **「Chạy nhận diện và so sánh mô hình」** chạy lần lượt cả 3 pipeline trên cùng ảnh.
- **Trạng thái:**
  - *Mô hình thực thi* — load weight thành công (`is_mock=False`).
  - *Mô phỏng dự phòng* — thiếu file hoặc lỗi inference → kết quả ngẫu nhiên (chỉ để xem UI).

## Lưu ý kỹ thuật (fair compare)

- Demo **crop/resize ảnh full-frame** do người dùng upload; **không** chạy YOLO detect như pipeline preprocess.
- DL dùng pipeline ảnh giống train (224, normalize). ML **trích xuất lại** HOG/LBP/color tại **64×64** (cùng tinh thần `vn_tsc/features/classical.py`); cần **`scaler.joblib` đúng** với lúc train thì SVM/Boosting mới khớp metric offline.
- SVM không có `predict_proba`: app dùng `decision_function` + **OvO voting** (1326 cặp → 52 lớp).
- GPU: DL tự dùng CUDA nếu có; ML chủ yếu CPU.

## Khắc phục sự cố

| Triệu chứng | Gợi ý |
| ------------- | -------- |
| Cả 3 model đều “Mô phỏng” | Kiểm tra `saved_models/` và tên file; xem log terminal khi Streamlit khởi động |
| DL load lỗi | Checkpoint phải khớp `tf_efficientnetv2_b0`, 52 classes; thử `strict=True` state_dict |
| ML sai lớc / xác suất lạ | Copy lại `scaler.joblib` từ preprocess; đảm bảo `class_table.csv` đủ 52 lớp |
| Không có ảnh mẫu | Thêm file vào `demo/sample_images/` |
