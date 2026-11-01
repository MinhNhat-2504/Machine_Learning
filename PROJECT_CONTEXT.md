# Bối cảnh dự án

- Mục tiêu: lưu bài thực hành Học máy ứng dụng (Văn Lang), mỗi lab có một notebook tiếng Việt và hình kết quả trong `images/`.
- Công nghệ: Python, Jupyter, NumPy, pandas, matplotlib, scikit-learn. Python 3.11 tại máy đã có đủ thư viện và nbclient/nbformat; Python 3.12 mặc định chưa có.
- Cấu trúc đã xác minh: Lab01–Lab04 nằm trong các thư mục đánh số; notebook gồm phần A thực hành, phần B bài tập, mã chạy từ trên xuống, giải thích và hình minh họa.
- Git: nhánh `main`, origin `https://github.com/MinhNhat-2504/Machine_Learning.git`. Người dùng xác nhận push vào origin hiện tại; link `AnhNguyenVLU/Machine_Learning_VLU` là nguồn tham khảo.
- Quy ước: `.gitignore` loại CSV/PDF, giữ notebook và ảnh; dữ liệu phải đặt cạnh notebook hoặc trong `data/`. Không đưa bí mật vào git.
- Trạng thái đầu phiên 2026-11-02: Lab01–Lab04 đã có trong git. `05.DecisionTree_RandomForest/` chưa được theo dõi, chứa đề PDF và `Lab05_data.zip` (hai tệp `data/khach_vay.csv`, `data/gia_nha.csv`).
- Lab5 theo đề địa phương: 7 phần thực hành và 6 bài tập; phân loại chia 75/25, `random_state=17`, `stratify=y`; mô hình `random_state=0`; hồi quy chia 80/20, `random_state=42`. Chọn tham số bằng CV trên tập học.
- Hoàn thành 2026-11-02: `Lab05_CayQuyetDinh_RungNgauNhien.ipynb` có 47 ô (18 ô mã), đủ 7 bước và 6 bài tập; đã chạy từ kernel mới bằng Python 3.11.9, sklearn 1.8.0, NumPy 1.26.4, pandas 2.3.3, matplotlib 3.10.8, không lỗi. Có 13 ảnh trong `images/`, output nhúng trong notebook. README đã thêm Lab5 và cách chuẩn bị dữ liệu; gói ZIP dữ liệu được bỏ qua bởi git.
- Kết quả xác minh: ngưỡng gốc tự tìm 15.05, CV chọn độ sâu 5; accuracy cây tự do/cây sâu 5/rừng 200 cây: 0.7667/0.8533/0.9000. Entropy sâu 5: 0.8467; min_samples_leaf=10: 0.8467 (CV hòa với 20, lấy giá trị xuất hiện trước). Cột nhiễu hạng MDI 3, permutation 5. MAE cây/rừng/đường thẳng: 0.5039/0.4559/0.2578 tỷ.
- Lưu ý: dữ liệu mô phỏng cho học tập; không chọn tham số từ điểm kiểm tra. Biên 2D học riêng trên hai cột; không đại diện trực tiếp cho mô hình bốn cột. Cây và rừng hồi quy không ngoại suy xu hướng. CSV/PDF/ZIP chỉ ở máy; cần cung cấp lại khi tải repo.
- Bàn giao Git: đích được người dùng xác nhận là `origin/main`; trạng thái đồng bộ cần kiểm tra bằng git, không suy ra từ file bối cảnh.
- Việc tiếp theo: khi có đề lab tiếp theo, tiếp tục cùng cấu trúc và kiểm tra từ kernel mới trước khi nộp.
