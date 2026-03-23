# Capstone Project - Phân loại ảnh ung thư da

Dự án này bám theo đề bài **Phân loại ảnh ung thư da (Skin Cancer Detection)**:
- Dùng dữ liệu `HAM10000_metadata.csv`
- Tiền xử lý missing value
- Chuẩn hóa dữ liệu
- Huấn luyện **SVM**
- Thực hiện **2 bài toán**:
  1. **Binary classification**: nhóm nguy cơ cao vs nhóm lành tính
  2. **Multi-class classification**: 7 loại tổn thương da
- Đánh giá bằng **Accuracy**, **Confusion Matrix**, **Classification Report**
- Có thêm **ứng dụng Streamlit** để demo dự đoán

> Lưu ý quan trọng: đề bài nói về phân loại ảnh ung thư da, nhưng tài liệu đi kèm yêu cầu dùng file `HAM10000_metadata.csv`. Vì vậy project này triển khai theo hướng **Machine Learning trên metadata** (age, sex, localization, dx), không huấn luyện trực tiếp trên ảnh thô.

## 1. Cấu trúc thư mục

```bash
skin_cancer_capstone/
│
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── skin_cancer_capstone.ipynb
├── data/
│   └── HAM10000_metadata.csv   # bạn đặt file vào đây
├── artifacts/
│   ├── binary_svm.joblib
│   └── multiclass_svm.joblib
├── outputs/
│   ├── binary_classification_report.txt
│   ├── multiclass_classification_report.txt
│   ├── binary_confusion_matrix.png
│   ├── multiclass_confusion_matrix.png
│   └── metrics_summary.json
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    ├── train.py
    └── predict.py
```

## 2. Cài đặt

### Tạo môi trường ảo

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Cài thư viện

```bash
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu

Đặt file `HAM10000_metadata.csv` vào thư mục:

```bash
data/HAM10000_metadata.csv
```

## 4. Huấn luyện mô hình

```bash
python -m src.train --csv data/HAM10000_metadata.csv
```

Sau khi chạy xong:
- mô hình sẽ được lưu trong `artifacts/`
- báo cáo và hình confusion matrix sẽ được lưu trong `outputs/`

## 5. Chạy giao diện demo Streamlit

```bash
streamlit run app.py
```

## 6. Ý nghĩa hai bài toán

### Binary classification
Theo đề bài:
- **Nhóm nguy cơ cao (1)**: `mel`, `bcc`, `akiec`, `vasc`
- **Nhóm lành tính (0)**: `nv`, `df`, `bkl`

### Multi-class classification
Phân loại 7 lớp:
- `nv`
- `mel`
- `bkl`
- `bcc`
- `akiec`
- `vasc`
- `df`

## 7. Ghi chú kỹ thuật

- Cột dùng cho mô hình: `age`, `sex`, `localization`
- Cột `image_id`, `lesion_id` chỉ là định danh, **không nên dùng để train**
- Mô hình dùng `SVC` với `class_weight="balanced"` để giảm ảnh hưởng mất cân bằng dữ liệu
- Tiền xử lý:
  - `SimpleImputer` cho missing value
  - `OneHotEncoder` cho biến phân loại
  - `StandardScaler` cho biến số

> Trong PDF có nhắc LabelEncoder cho biến phân loại. Trong thực hành ML chuẩn, `OneHotEncoder` phù hợp hơn cho feature đầu vào, còn `LabelEncoder` được dùng cho nhãn đầu ra của bài toán multi-class.

## 8. Notebook

Bạn có thể mở file:

```bash
notebooks/skin_cancer_capstone.ipynb
```

để trình bày bài tập theo kiểu Jupyter Notebook.

## 9. Mở rộng

Các hướng mở rộng đúng theo tinh thần đề bài:
- dùng ảnh thô và CNN
- triển khai FastAPI/Flask
- thêm ROC-AUC, cross-validation
- tối ưu hyperparameter bằng GridSearchCV

