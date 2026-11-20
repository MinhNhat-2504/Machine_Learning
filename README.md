# Học máy ứng dụng — Bài thực hành

Kho lưu trữ các bài thực hành môn **Học máy ứng dụng**, Trường Đại học Văn Lang. Nội dung gồm 9 lab, từ hồi quy và phân loại đến phân cụm và mạng nơ-ron.

Mỗi lab có một notebook Jupyter trình bày bằng tiếng Việt, gồm **phần A — Thực hành** và **phần B — Bài tập**. Mã nguồn, kết quả, hình minh họa và nhận xét được đặt cùng nhau để tiện theo dõi từ lý thuyết đến thực nghiệm.

## Danh sách notebook

| Lab | Chủ đề | Notebook |
|---|---|---|
| 1 | Hồi quy tuyến tính | [Lab01_Hoi_quy_tuyen_tinh.ipynb](01.%20Linear%20Regression/Lab01_Hoi_quy_tuyen_tinh.ipynb) |
| 2 | Hồi quy logistic | [Lab02_Hoi_quy_logistic.ipynb](02.%20Logistic%20Regression/Lab02_Hoi_quy_logistic.ipynb) |
| 3 | Naive Bayes | [Lab03_Naive_Bayes.ipynb](03.%20Naive%20Bayes/Lab03_Naive_Bayes.ipynb) |
| 4 | K láng giềng gần nhất (KNN) | [Lab04_KNN.ipynb](04.%20KNN/Lab04_KNN.ipynb) |
| 5 | Cây quyết định và rừng ngẫu nhiên | [Lab05_CayQuyetDinh_RungNgauNhien.ipynb](05.DecisionTree_RandomForest/Lab05_CayQuyetDinh_RungNgauNhien.ipynb) |
| 6 | Máy vectơ hỗ trợ (SVM) | [Lab06_SVM.ipynb](06.%20SVM/Lab06_SVM.ipynb) |
| 7 | Đánh giá mô hình | [Lab07_DanhGiaMoHinh.ipynb](07.%20Accuracy_Metrics/Lab07_DanhGiaMoHinh.ipynb) |
| 8 | Phân cụm K-Means | [Lab08_KMeans.ipynb](08.%20K-Means/Lab08_KMeans.ipynb) |
| 9 | Mạng nơ-ron nhiều lớp (MLP) | [Lab09_MLP.ipynb](09.%20MLP/Lab09_MLP.ipynb) |

## Nội dung từng lab

### Lab 1 — Hồi quy tuyến tính

**Dữ liệu:** `gia_nha.csv` — diện tích, số phòng, tuổi nhà và giá bán.

**Thực hành:** đọc và trực quan hóa dữ liệu; đo sai số MSE; tìm hệ số bằng công thức bình phương tối thiểu; huấn luyện `LinearRegression`; chia tập học/kiểm tra; đánh giá MAE, MSE, R²; tối ưu bằng gradient descent; hồi quy nhiều biến và khảo sát ngoại suy.

**Bài tập:** lọc căn hộ lớn; vẽ phân tán theo số phòng; dùng tuổi nhà làm đầu vào; thêm số phòng vào mô hình; thử tốc độ học; viết hàm dự đoán có cảnh báo ngoài vùng dữ liệu.

### Lab 2 — Hồi quy logistic

**Dữ liệu:** `sinh_vien.csv` — giờ ôn tập, điểm giữa kỳ và nhãn qua môn.

**Thực hành:** khám phá dữ liệu hai lớp; tự tính sigmoid; huấn luyện `LogisticRegression`; đọc xác suất dự đoán; ma trận nhầm lẫn; accuracy, precision, recall, F1; thay đổi ngưỡng quyết định; thêm đặc trưng và vẽ biên phân loại.

**Bài tập:** thống kê theo nhóm điểm; vẽ sigmoid; dự đoán cho một sinh viên; tự tính bốn thước đo; tìm ngưỡng theo F1; đổi lớp dương và giải thích kết quả.

### Lab 3 — Naive Bayes

**Dữ liệu:** `tin_nhan.csv` và `sinh_vien.csv`.

**Thực hành:** tính định lý Bayes bằng tay; biểu diễn văn bản bằng túi từ và `CountVectorizer`; lọc tin nhắn rác bằng `MultinomialNB`; tìm hiểu xác suất bằng không, làm trơn Laplace và phép cộng log; dùng `GaussianNB` cho dữ liệu liên tục và so với logistic.

**Bài tập:** Bayes với ba nguyên nhân; tìm từ đặc trưng của tin rác; thay đổi alpha; tự viết bộ lọc; thử sai biến thể Naive Bayes; khảo sát lớp hiếm và xác suất tiên nghiệm.

### Lab 4 — K láng giềng gần nhất (KNN)

**Dữ liệu:** `sinh_vien_mo_rong.csv`, `hai_vang_trang.csv` và `gia_nha.csv`.

**Thực hành:** tự tính khoảng cách Euclid và tìm láng giềng; chuẩn hóa với `StandardScaler` trong pipeline; chọn k bằng kiểm định chéo; so KNN với logistic và Naive Bayes; khảo sát số chiều; KNN hồi quy và giới hạn ngoại suy.

**Bài tập:** tự viết KNN; dùng khoảng cách Manhattan; đổi đơn vị đo; bỏ phiếu có trọng số; chọn k cho hồi quy; đo thời gian dự đoán.

### Lab 5 — Cây quyết định và rừng ngẫu nhiên

**Dữ liệu:** `khach_vay.csv` và `gia_nha.csv`.

**Thực hành:** tự tính Gini và chọn câu hỏi chia nút; học, đọc cây năm lá và dự đoán khách mới; nhận diện quá khớp; chọn độ sâu bằng kiểm định chéo; khảo sát thang đo; bootstrap và rừng ngẫu nhiên; độ quan trọng đặc trưng; cây hồi quy và ngoại suy.

**Bài tập:** vẽ cây; tự tìm câu hỏi gốc trên dữ liệu thật; so Gini với entropy; chọn `min_samples_leaf`; so MDI với permutation importance; dùng rừng ngẫu nhiên cho hồi quy.

### Lab 6 — Máy vectơ hỗ trợ (SVM)

**Dữ liệu:** `sinh_vien.csv`, `sinh_vien_mo_rong.csv`, `hai_vang_trang.csv`, dữ liệu sinh bằng `make_blobs` và bộ chữ số `load_digits`.

**Thực hành:** học biên tuyến tính và tính độ rộng lề; tìm vectơ hỗ trợ; khảo sát C và chuẩn hóa; nâng chiều bằng tay; so kernel tuyến tính, đa thức, RBF; chọn C/gamma bằng dò lưới; nhận dạng chữ số viết tay.

**Bài tập:** tính khoảng cách tới biên; thêm điểm xa/gần biên; dò kernel đa thức; dò lưới mịn; phân tích và lưu ảnh chữ số bị nhầm; so thời gian học SVC với LinearSVC khi tăng số mẫu.

### Lab 7 — Đánh giá mô hình

**Dữ liệu:** `sinh_vien_bo_hoc.csv`, `gia_nha.csv` và bộ chữ số `load_digits`.

**Thực hành:** so với mô hình luôn đoán lớp đông nhất; tự đếm TP, FP, FN, TN; tính precision, recall, F1; chọn ngưỡng từ dự đoán ngoài mẫu trên tập học; ROC/AUC; so các mô hình; phân tầng, kiểm định chéo nhiều thước đo và cân bằng lớp.

**Bài tập:** tự viết hàm đánh giá bằng NumPy; đọc báo cáo từng chữ số; chọn ngưỡng đạt recall yêu cầu; vẽ ROC logistic/rừng; cân bằng lớp cho SVC và RandomForest; khảo sát tác động của một dự đoán sai lớn lên MAE, RMSE và R².

### Lab 8 — Phân cụm K-Means

**Dữ liệu:** `khach_hang.csv`, `hai_vang_trang.csv`, ảnh mẫu `china.jpg` và bộ chữ số `load_digits`.

**Thực hành:** tự viết K-Means và theo dõi từng vòng; phân khúc khách hàng; đọc tâm cụm theo đơn vị gốc; chọn K bằng khuỷu tay và silhouette; khảo sát thang đo, khởi tạo và hình dạng cụm; nén màu ảnh.

**Bài tập:** tự tính silhouette từng điểm; thêm tuổi vào đặc trưng; so khởi tạo ngẫu nhiên với K-Means++; lưu và so dung lượng ảnh nén; thử DBSCAN; phân cụm chữ số rồi phân tích bảng chéo với nhãn thật.

### Lab 9 — Mạng nơ-ron nhiều lớp (MLP)

**Dữ liệu:** `xor.csv`, `hai_vang_trang.csv`, `sinh_vien_mo_rong.csv`, `gia_nha.csv` và bộ chữ số `load_digits`.

**Thực hành:** tự tính lan truyền tiến; xây mạng giải XOR; khảo sát số nơ-ron, kích thước mạng và điều chuẩn alpha; chuẩn hóa đầu vào; nhận dạng chữ số; đếm tham số, đọc đường mất mát và thử dừng sớm.

**Bài tập:** dự đoán bằng trọng số đã học; kiểm chứng một mạng XOR khác; so hàm kích hoạt; khảo sát tốc độ học; dò lưới kiến trúc/alpha; dùng MLP cho hồi quy và phân tích dự đoán ngoài miền học.

## Cấu trúc thư mục

```text
.
├── 01. Linear Regression/
├── 02. Logistic Regression/
├── 03. Naive Bayes/
├── 04. KNN/
├── 05.DecisionTree_RandomForest/
├── 06. SVM/
├── 07. Accuracy_Metrics/
├── 08. K-Means/
├── 09. MLP/
└── README.md
```

Mỗi thư mục lab chứa notebook `.ipynb` và thư mục `images/` lưu hình kết quả. Notebook có sẵn output để xem trực tiếp trên GitHub; các phép tính và nhận xét chi tiết nằm trong từng bài.

## Môi trường và cách chạy

Sử dụng **Python 3.10+** cùng các thư viện NumPy, pandas, Matplotlib, scikit-learn và Jupyter.

1. Cài thư viện:

   ```bash
   python -m pip install numpy pandas matplotlib scikit-learn jupyter
   ```

2. Chuẩn bị CSV theo danh sách dữ liệu của lab. Đặt chúng trong thư mục `data/` của lab hoặc trực tiếp cạnh notebook. Nếu có gói `LabXX_data.zip` đi kèm đề, giải nén trong thư mục lab tương ứng.

3. Mở terminal tại thư mục lab muốn chạy, sau đó khởi động Jupyter:

   ```bash
   jupyter notebook
   ```

4. Mở notebook và chọn **Restart Kernel and Run All Cells** để chạy tuần tự từ đầu. Hình kết quả được lưu vào `images/` của lab.

Các tệp `.csv`, `.pdf` và `Lab*_data.zip` được bỏ qua bởi `.gitignore`, vì vậy cần chuẩn bị lại dữ liệu khi tải repo về máy khác. Bộ chữ số `load_digits` và ảnh mẫu `china.jpg` có sẵn trong scikit-learn. Phần ảnh bổ sung của Lab8 dùng một biểu đồ trong thư mục ảnh Lab5; có thể thay đường dẫn bằng ảnh khác trong notebook.

Kết quả và thời gian chạy có thể khác theo phiên bản thư viện, phần cứng và cách chia dữ liệu. Khi đối chiếu bài, giữ các tham số và `random_state` được ghi trong notebook.

## Tác giả

[MinhNhat-2504](https://github.com/MinhNhat-2504)
