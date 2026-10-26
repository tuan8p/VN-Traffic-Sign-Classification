# Phân tích dữ liệu (EDA)

Exploratory Data Analysis trên **kho đã preprocess** (`data/processed/`), phục vụ báo cáo và thiết kế thí nghiệm. **Không cần Weights & Biases.**

## Mục tiêu

- Mô tả phân bố 52 lớp, mất cân bằng và lớp hiếm (&lt; 30 mẫu).
- Đo tác dụng **offline augmentation** (`train` vs `train_aug`).
- Phân tích độ khó thực tế: kích thước bbox, vị trí biển trong khung, độ sáng / độ nét.
- Trực quan hóa **mirror-pair** (lý do cấm `hflip`) và nhóm **confusable**.
- Tóm tắt **leakage** gần trùng lặp giữa split (từ `preprocess_report.json`).

## Cấu trúc mã

| File | Vai trò |
|------|---------|
| `eda.py` | Logic EDA: vẽ figure, ghi JSON/CSV tóm tắt |
| `eda_stub.py` | Entry point tương thích scaffold cũ → gọi `eda.main()` |

**Input bắt buộc** (sau `python -m tools.run_preprocess`):

- `data/processed/metadata.csv`
- `data/processed/class_table.csv`
- `data/processed/images/X_train.npy` (memory-map cho lưới ví dụ)
- `data/processed/preprocess_report.json` (tùy chọn; thiếu thì bỏ qua figure leakage)

**Output mặc định:** `analysis/figures/`

## Chạy

Từ thư mục gốc repository (đã `pip install -e .` hoặc `pip install -r requirements.txt`):

```bash
python -m analysis.eda
```

Tùy chọn:

```bash
python -m analysis.eda --processed data/processed --out analysis/figures
```

## Artefact sinh ra

### Hình (`analysis/figures/`)

| File | Nội dung |
| ------ | ---------- |
| `01_class_distribution.png` | Số crop thật theo lớp, stack train/val/test (log scale) |
| `02_augmentation_effect.png` | Train trước/sau offline aug |
| `03_box_sizes.png` | Phân bố cạnh ngắn bbox & diện tích (so với ngưỡng 32×32 px) |
| `04_position_heatmap.png` | Vị trí tâm biển trong ảnh gốc |
| `05_leakage.png` | Cụm near-duplicate & ảnh “straddle” hai split |
| `06_class_examples.png` | Một ví dụ/lớp (crop lớn nhất trên train) |
| `07_mirror_pairs.png` | Cặp mirror — vì sao không flip ngang |
| `08_confusable_groups.png` | Nhóm dễ nhầm ở kích thước median |
| `09_brightness_sharpness.png` | Độ sáng & độ nét (Laplacian) trên mẫu train |

### Bảng số

- `eda_summary.json` — tổng quan: số crop, imbalance, danh sách lớp hiếm, mirror pairs, leakage, …
- `per_class_counts.csv` — đếm theo lớp cho từng split (`train`, `val`, `test`, `train_aug`)

## Phụ thuộc

Dùng stack core của project: `matplotlib`, `pandas`, `numpy`, `opencv-python-headless` (figure 09), module `vn_tsc`.

## Ghi chú

- Nhãn trục figure dùng **mã biển** (`P.102`, …) thay vì tên tiếng Việt dài (tránh lỗi console Windows).
- EDA **chỉ đọc** processed store; không thay đổi split hay nhãn.
