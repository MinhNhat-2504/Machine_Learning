# Machine Learning — Bài thực hành

Kho lưu trữ các bài thực hành môn **Học máy ứng dụng**. Mỗi bài nằm trong một thư mục riêng, gồm notebook Jupyter và các hình ảnh kết quả.

## Cấu trúc thư mục

```
.
├── 01. Linear Regression/
│   ├── Lab01_Hoi_quy_tuyen_tinh.ipynb   # Notebook bài 1
│   └── images/                          # Hình ảnh kết quả sinh ra từ notebook
└── README.md
```

> Các file `.csv` và `.pdf` được liệt kê trong `.gitignore` nên không được đẩy lên GitHub. Dữ liệu `gia_nha.csv` cần đặt cạnh notebook khi chạy lại.

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
