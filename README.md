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
├── 03. Naive Bayes/
│   ├── Lab03_Naive_Bayes.ipynb          # Notebook bài 3
│   └── images/                          # Hình ảnh kết quả sinh ra từ notebook
├── 04. KNN/
│   ├── Lab04_KNN.ipynb                  # Notebook bài 4
│   └── images/                          # Hình ảnh kết quả sinh ra từ notebook
├── 05.DecisionTree_RandomForest/
│   ├── Lab05_CayQuyetDinh_RungNgauNhien.ipynb  # Notebook bài 5
│   └── images/                          # 13 hình kết quả, gồm cây năm lá
├── PROJECT_CONTEXT.md                   # Bối cảnh và tiến độ dự án
└── README.md
```

> Các file `.csv`, `.pdf` và gói dữ liệu `Lab05_data.zip` được liệt kê trong `.gitignore` nên không được đẩy lên GitHub. Dữ liệu (`gia_nha.csv`, `sinh_vien.csv`, `tin_nhan.csv`, `sinh_vien_mo_rong.csv`, `hai_vang_trang.csv`, `khach_vay.csv`) cần đặt cạnh notebook tương ứng hoặc trong thư mục `data/` của lab khi chạy lại. Với Lab5, giải nén `Lab05_data.zip` trong `05.DecisionTree_RandomForest/` để có `data/khach_vay.csv` và `data/gia_nha.csv`.

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

### Bài 3 — Naive Bayes

Lọc tin nhắn rác từ bộ dữ liệu `tin_nhan.csv` (các cột `noi_dung`, `la_rac`), và dùng lại `sinh_vien.csv` của bài 2 để so với hồi quy logistic.

**Phần A — Thực hành**

1. Tính định lý Bayes bằng tay (ví dụ xét nghiệm bệnh, hai dây chuyền sản xuất)
2. Biến văn bản thành bảng số bằng túi từ và `CountVectorizer`
3. Huấn luyện và chấm điểm bộ lọc tin rác bằng `MultinomialNB`
4. Tự cài Naive Bayes để thấy vấn đề số không, làm trơn Laplace, tràn dưới
5. `GaussianNB` cho số liệu liên tục, so với hồi quy logistic và khảo sát theo cỡ dữ liệu

**Phần B — Bài tập**

1. Bayes với ba nguyên nhân
2. Từ nào tố cáo tin rác mạnh nhất
3. Đổi tham số làm trơn alpha
4. Tự viết bộ lọc không dùng thư viện (cộng loga)
5. Chọn sai biến thể Naive Bayes
6. Lớp hiếm và xác suất tiên nghiệm

### Bài 4 — K láng giềng gần nhất

Dùng `sinh_vien_mo_rong.csv` (thêm cột `chi_tieu`), `hai_vang_trang.csv` (300 điểm có ranh giới cong) và `gia_nha.csv` của bài 1.

**Phần A — Thực hành**

1. Tự tính khoảng cách Euclid và tìm láng giềng bằng numpy
2. Hậu quả của việc quên chuẩn hoá, `StandardScaler` và `make_pipeline`
3. Chọn số láng giềng k bằng kiểm định chéo
4. So KNN với hồi quy logistic và Naive Bayes, lời nguyền số chiều
5. KNN cho bài toán hồi quy và giới hạn ngoại suy

**Phần B — Bài tập**

1. Tự viết KNN không dùng thư viện
2. Khoảng cách Manhattan
3. Đổi đơn vị đo
4. Láng giềng có trọng số
5. Chọn k cho bài toán hồi quy
6. Đo thời gian dự đoán (học lười)

### Bài 5 — Cây quyết định và Rừng ngẫu nhiên

[Mở notebook Lab5](05.DecisionTree_RandomForest/Lab05_CayQuyetDinh_RungNgauNhien.ipynb).

Dùng `khach_vay.csv` (600 hồ sơ, bốn đặc trưng: `thu_nhap`, `ty_le_no`, `co_viec_on_dinh`, `so_nguoi_phu_thuoc`; nhãn `tra_dung_han`) và `gia_nha.csv` (60 căn hộ).

**Phần A — Thực hành**

1. Tự tính Gini, thử mọi ngưỡng và đối chiếu cây một tầng
2. Học cây năm lá, đọc luật, đường đi và xác suất của hai khách mới
3. Nhận diện quá khớp, chọn độ sâu bằng kiểm định chéo 5 phần
4. Kiểm chứng cây không cần chuẩn hoá hoặc đổi đơn vị đo
5. Bootstrap, rừng 200 cây và so sánh với cây đơn
6. Độ quan trọng MDI và giới hạn khi thêm cột ngẫu nhiên
7. Cây hồi quy, so MAE với đường thẳng và giới hạn ngoại suy

**Phần B — Bài tập**

1. Vẽ cây năm lá và cây sâu 2; giải thích hai lá cùng nhãn vẫn được tách
2. Tự viết `tim_cau_hoi(X, y)` trên 450 khách; kiểm chứng ngưỡng thu nhập 15.05
3. So Gini với entropy, chọn độ sâu riêng bằng kiểm định chéo
4. Chọn `min_samples_leaf`, so với giới hạn độ sâu 5
5. Permutation importance với 20 lần xáo trộn, so thứ hạng cột ngẫu nhiên với MDI
6. Rừng hồi quy 200 cây, so MAE và giải thích dự đoán căn 250 m²

Notebook đã chạy toàn bộ bằng Python 3.11.9, NumPy 1.26.4, pandas 2.3.3, matplotlib 3.10.8 và scikit-learn 1.8.0; giữ sẵn kết quả và 13 hình trong `images/`.

| Thí nghiệm | Kết quả trên tập kiểm tra |
|---|---|
| Cây mọc tự do | Accuracy 0.7667 |
| Cây sâu 5, chọn bằng CV | Accuracy 0.8533 |
| Rừng phân loại 200 cây | Accuracy 0.9000 |
| Cây hồi quy sâu 3 | MAE 0.5039 tỷ đồng |
| Rừng hồi quy 200 cây | MAE 0.4559 tỷ đồng |
| Hồi quy tuyến tính | MAE 0.2578 tỷ đồng |

Chọn tham số chỉ bằng tập học; các thí nghiệm dùng chung cách chia theo đề. Để chạy lại, mở notebook và chọn **Restart Kernel and Run All Cells** sau khi đặt đủ dữ liệu.

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
