# Machine Learning — Bài thực hành

Kho lưu trữ các bài thực hành môn **Học máy ứng dụng**. Mỗi bài nằm trong một thư mục riêng, gồm notebook Jupyter và các hình ảnh kết quả.

## Cấu trúc thư mục

```
.
├── 01. Linear Regression/
│   ├── Lab01_Hoi_quy_tuyen_tinh.ipynb   # Notebook bài 1
│   └── images/                          # Hình ảnh kết quả sinh ra từ notebook
├── 02. Logistic Regression/
│   ├── Lab02_Hoi_quy_logistic.ipynb     # Notebook bài 2
│   └── images/                          # Hình ảnh kết quả sinh ra từ notebook
└── README.md
```

> Các file `.csv` và `.pdf` được liệt kê trong `.gitignore` nên không được đẩy lên GitHub. Dữ liệu (`gia_nha.csv`, `sinh_vien.csv`) cần đặt cạnh notebook tương ứng khi chạy lại.

## Danh sách bài

### Bài 1 — Hồi quy tuyến tính

Dự đoán giá nhà từ bộ dữ liệu `gia_nha.csv` (các cột `dien_tich`, `so_phong`, `tuoi_nha`, `gia`).

**Phần A — Thực hành**

1. Đọc và quan sát dữ liệu bằng pandas
2. Đo sai số MSE của một đường thẳng
3. Tìm `w`, `b` bằng công thức bình phương tối thiểu
4. Huấn luyện với `sklearn.linear_model.LinearRegression`
5. Chia tập train/test và đánh giá (MAE, MSE, R²)
6. Tối ưu bằng gradient descent, khảo sát tốc độ học
7. Hồi quy nhiều biến và giới hạn ngoại suy

**Phần B — Bài tập**

1. Lọc nhóm căn hộ lớn
2. Biểu đồ phân tán theo số phòng
3. Đổi biến đầu vào sang tuổi nhà
4. Thêm số phòng vào mô hình
5. Thử hai tốc độ học khác
6. Viết hàm dự đoán có cảnh báo ngoài vùng dữ liệu

### Bài 2 — Hồi quy logistic

Dự đoán sinh viên qua hay rớt môn từ bộ dữ liệu `sinh_vien.csv` (các cột `gio_on`, `diem_giua_ky`, `qua_mon`).

**Phần A — Thực hành**

1. Đọc dữ liệu, so trung bình đặc trưng giữa hai lớp, vì sao đường thẳng của bài 1 không dùng được
2. Tự cài đặt và kiểm chứng hàm sigmoid
3. Khớp mô hình bằng `sklearn.linear_model.LogisticRegression`, đọc `predict_proba`
4. Ma trận nhầm lẫn và bốn thước đo (accuracy, precision, recall, F1)
5. Đổi ngưỡng quyết định, đánh đổi giữa precision và recall
6. Thêm biến thứ hai và vẽ biên quyết định

**Phần B — Bài tập**

1. Thống kê tỷ lệ qua môn theo nhóm điểm giữa kỳ
2. Vẽ hàm sigmoid
3. Hàm dự đoán cho một bạn cụ thể
4. Tự tính bốn thước đo từ TP, TN, FP, FN
5. Dò ngưỡng tốt nhất theo F1
6. Đổi lớp dương rồi chấm lại

## Môi trường

- Python 3.10+
- Thư viện: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter`

Cài đặt:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Chạy notebook:

```bash
jupyter notebook
```

## Tác giả

MinhNhat-2504
