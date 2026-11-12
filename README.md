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
├── 06. SVM/
│   ├── Lab06_SVM.ipynb                   # Notebook bài 6
│   └── images/                          # 11 hình, gồm từng ảnh chữ số đoán sai
├── 07. Accuracy_Metrics/
│   ├── Lab07_DanhGiaMoHinh.ipynb          # Notebook bài 7
│   └── images/                          # 5 hình, gồm roc.png
├── 08. K-Means/
│   ├── Lab08_KMeans.ipynb                # Notebook bài 8
│   └── images/                          # 13 ảnh, gồm ảnh gốc và ảnh nén
├── 09. MLP/
│   ├── Lab09_MLP.ipynb                   # Notebook bài 9
│   └── images/                          # 7 hình, gồm toc_do_hoc.png
└── README.md
```

> Các file `.csv`, `.pdf` và gói dữ liệu `Lab*_data.zip` được liệt kê trong `.gitignore` nên không được đẩy lên GitHub. Để chạy lại, giải nén gói dữ liệu tương ứng vào thư mục lab để có `data/*.csv`, hoặc đặt CSV trực tiếp cạnh notebook. Các bộ `load_digits` và ảnh `china.jpg` có sẵn trong scikit-learn, không tải mạng. Notebook có sẵn output để xem kết quả khi chưa có CSV.

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

### Bài 6 — Máy vectơ hỗ trợ (SVM)

[Mở notebook Lab6](06.%20SVM/Lab06_SVM.ipynb).

Dữ liệu: `sinh_vien.csv`, `sinh_vien_mo_rong.csv`, `hai_vang_trang.csv`, 24 điểm `make_blobs` và ảnh chữ số `load_digits`.

**Phần A — Thực hành**

1. Học biên tuyến tính, tính độ rộng lề, xem vectơ hỗ trợ và học lại chỉ với các điểm hỗ trợ
2. Khảo sát C và hậu quả quên chuẩn hoá
3. Nâng chiều bằng tay, so kernel tuyến tính/đa thức/RBF với logistic và KNN
4. Khảo sát gamma, chọn C/gamma bằng kiểm định chéo 5 phần
5. Nhận dạng chữ số và so các mô hình trên cùng cách chia

**Phần B — Bài tập**

1. Tính tay f(x) và khoảng cách tới biên, kiểm chứng nửa độ rộng lề
2. Thêm điểm xa/gần biên, giải thích bằng vectơ hỗ trợ
3. Dò C cho kernel đa thức bậc 2, 3, 4
4. Dò lưới mịn và so mức cải thiện CV
5. Ma trận nhầm lẫn, lưu riêng từng ảnh chữ số bị đoán sai
6. Đo thời gian SVC/LinearSVC với 2.000–20.000 mẫu

Kết quả: lề **2.2126**, 3 vectơ hỗ trợ; RBF trên vầng trăng chọn **C=10, gamma=1**, accuracy **0.9200**; chữ số **0.9911**, sai 4/450 ảnh. Có **11 hình** trong `images/`.

### Bài 7 — Đánh giá mô hình

[Mở notebook Lab7](07.%20Accuracy_Metrics/Lab07_DanhGiaMoHinh.ipynb).

Dữ liệu: `sinh_vien_bo_hoc.csv` (1.200 sinh viên, 119 bỏ học), `gia_nha.csv` và `load_digits`.

**Phần A — Thực hành**

1. So accuracy của mô hình lười và logistic trên dữ liệu lệch lớp
2. Tự đếm TP/FP/FN/TN, đối chiếu ma trận nhầm lẫn
3. Precision, recall, F1 và classification report
4. Khảo sát ngưỡng; chọn ngưỡng từ xác suất out-of-fold trên tập học
5. ROC/AUC và so sáu mô hình
6. Phân tầng, CV nhiều thước đo và cân bằng lớp

**Phần B — Bài tập**

1. Hàm đánh giá chỉ dùng NumPy, xử lý rõ chia cho 0
2. Precision/recall từng chữ số, so macro/weighted
3. Chọn ngưỡng lớn nhất đạt recall OOF ≥0.8, rồi chấm kiểm tra
4. Tự vẽ ROC logistic/rừng, lưu `roc.png`, phân tích chỗ đường cắt nhau
5. So cân bằng lớp cho RandomForest và SVC
6. Tác động của một dự đoán giá nhà sai thêm 5 tỷ lên MAE/RMSE/R²

Kết quả: logistic accuracy **0.9467**, precision **0.8500**, recall **0.5667**, AUC **0.9449**. Ngưỡng recall yêu cầu chọn từ OOF là **0.1717736449**; kiểm tra đạt **0.8000**, precision **0.5106**, báo **47/300 bạn**. Có **5 hình**.

### Bài 8 — Phân cụm K-Means

[Mở notebook Lab8](08.%20K-Means/Lab08_KMeans.ipynb).

Dữ liệu: `khach_hang.csv` (210 khách), `hai_vang_trang.csv`, ảnh `china.jpg`, `load_digits`. Ảnh bổ sung dùng biểu đồ Lab5 trong repo theo lựa chọn của người dùng; không phải ảnh cá nhân.

**Phần A — Thực hành**

1. Tự viết K-Means, lưu từng vòng, kiểm tra cụm rỗng và inertia với sklearn
2. Phân khúc khách hàng, đọc tâm cụm ở đơn vị gốc và dự đoán khách mới
3. Chọn K bằng khuỷu tay/silhouette
4. Ba bẫy: thang đo, khởi tạo và cụm không tròn
5. Nén ảnh 4/16/64 màu, phân biệt số bit lý tưởng với dung lượng tệp thực

**Phần B — Bài tập**

1. Tự tính silhouette từng điểm bằng NumPy, đánh dấu khách có điểm nhỏ nhất
2. Thêm tuổi, chọn lại K và so sự thay đổi nhóm
3. So 50 lần khởi tạo random với 50 lần K-Means++
4. Lưu ảnh gốc/16 màu, so byte thực, nén thêm ảnh đọc từ repo và bỏ alpha
5. DBSCAN với ba eps, đếm cụm/nhiễu và đánh giá hậu nghiệm
6. Phân cụm chữ số không dùng nhãn khi fit, rồi phân tích bảng chéo

Kết quả: thuật toán tự viết có inertia **8.5250**; khách hàng chọn **K=5**, silhouette **0.7227**. K-Means++ đạt nghiệm inertia 23.0 sau làm tròn ở **50/50** seed, random **30/50**. PNG mẫu 16 màu nhỏ hơn khoảng **4.09 lần**. Có **13 ảnh**. Với dữ liệu vầng trăng hiện có, cả ba eps của đề đều tạo một cụm, không tách tốt hơn K-Means; notebook giữ đúng kết quả này.

### Bài 9 — Mạng nơ-ron nhiều lớp (MLP)

[Mở notebook Lab9](09.%20MLP/Lab09_MLP.ipynb).

Dữ liệu: `xor.csv`, `hai_vang_trang.csv`, `sinh_vien_mo_rong.csv`, `gia_nha.csv` và `load_digits`.

**Phần A — Thực hành**

1. Tự tính một nơ-ron và mạng ReLU giải bảng XOR bit
2. So logistic với mạng 1/2/4/8 nơ-ron ẩn trên CSV góc có nhiễu
3. Khảo sát kích thước mạng và chọn alpha bằng CV
4. Kiểm chứng hậu quả quên chuẩn hoá
5. Chữ số: đếm tham số, xem loss, dừng sớm và so các mô hình

**Phần B — Bài tập**

1. Tự lan truyền tiến từ coefs_/intercepts_, so xác suất với sklearn
2. Kiểm chứng mạng XOR thứ hai và giải thích bằng trị tuyệt đối
3. So logistic/tanh/ReLU trên cùng mạng hai lớp ẩn
4. So bốn tốc độ học, số vòng/loss/accuracy, lưu `toc_do_hoc.png`
5. GridSearchCV 12 cặp × 5 phần trên tập học chữ số
6. MLP hồi quy so với đường thẳng, phân tích dự đoán ngoài miền tại 250 m²

Kết quả: MLP chữ số cơ sở **7.510 tham số**, **144 vòng**, accuracy **0.9844**. Lưới chọn **(50,), alpha=1**, CV **0.9785**, kiểm tra **0.9822**, chưa vượt SVM **0.9911**. Lan truyền tiến NumPy khớp xác suất sklearn tới sai số khoảng **3.47×10⁻¹⁸**. Có **7 hình**, không ghi nhận cảnh báo hội tụ trong lần chạy kiểm tra.

### Kiểm tra Lab6–9

Cả **50 ô mã** của bốn notebook đã chạy từ kernel mới, không lỗi; có **36 tệp PNG** và kết quả nhúng trong notebook. Kiểm chứng gồm công thức lề SVM, TP/FP/FN/TN, chỉ số NumPy, ngưỡng OOF, K-Means/inertia/silhouette và lan truyền tiến MLP. Lab8 dùng nhãn thật chỉ sau phân cụm. Các giá trị thiếu có chủ đích trong bảng K=1 hoặc tỷ lệ thời gian dòng đầu được giữ là NaN vì chỉ số không xác định tại đó.

Môi trường kiểm tra: Python **3.11.9**, NumPy **1.26.4**, pandas **2.3.3**, matplotlib **3.10.8**, scikit-learn **1.8.0**; BLAS/OpenMP giới hạn 1 luồng. Thời gian chạy và nghiệm MLP có thể khác giữa các máy/phiên bản; các nhận xét ghi theo kết quả thực chạy thay vì ép số trùng PDF.

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
