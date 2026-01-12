# 🏠 ĐỒ ÁN: DỰ ĐOÁN GIÁ PHÒNG TRỌ CHO SINH VIÊN

Dự án ứng dụng Khoa học dữ liệu (Data Science) để thu thập, làm sạch, phân tích và xây dựng mô hình học máy (Machine Learning) nhằm dự đoán giá phòng trọ, hỗ trợ sinh viên tìm kiếm nơi ở phù hợp.

## 📂 Cấu trúc Thư mục Dự án

```text
├── data/                       # Dữ liệu của dự án
│   ├── raw/                    # Dữ liệu thô sau khi cào (scraping)
│   ├── clear/                  # Dữ liệu sau khi làm sạch sơ bộ
│   └── processed/              # Dữ liệu sạch hoàn chỉnh (đã xử lý EDA)
├── GUI/                        # Giao diện người dùng
│   ├── categorical_option.py   # File xử lý tùy chọn danh mục
│   └── gui.py                  # File chạy giao diện chính
├── notebooks/                  # Các Jupyter Notebooks theo từng giai đoạn
│   ├── scraping/               # Code cào dữ liệu (Chotot, Homedy...)
│   ├── cleaning/               # Code làm sạch dữ liệu
│   ├── eda/                    # Khám phá dữ liệu (EDA) và trả lời câu hỏi
│   ├── feature_engineering/    # Trích xuất đặc trưng
│   ├── merge/                  # Gộp dữ liệu
│   └── modeling/               # Huấn luyện và đánh giá mô hình
├── reports/                    # Báo cáo, hình ảnh và slides
├── requirements.txt            # Danh sách thư viện cần cài đặt
└── README.md                   # Hướng dẫn sử dụng
```
## ⚙️ Cài đặt môi trường
Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt Python. Cài đặt các thư viện cần thiết bằng lệnh sau:
```bash
pip install -r requirements.txt
```
## 🚀 Hướng dẫn chạy Đồ án
Dưới đây là quy trình chi tiết để chạy dự án từ bước thu thập dữ liệu đến khi sử dụng mô hình.
### Bước 1: Tạo dữ liệu (Data Pipeline)
> **Lưu ý:** Dữ liệu mẫu đã được tạo sẵn trong thư mục `data`. Bạn có thể bỏ qua bước này nếu không muốn thu thập lại dữ liệu từ đầu.

1. **Thu thập dữ liệu (Scraping):**
   Chạy 4 notebook trong thư mục `notebooks/scraping/` để tải dữ liệu về `data/raw`:
   * `chotot_scraper.ipynb`
   * `homedy_scraper.ipynb`
   * `phongtro123_scraper.ipynb`
   * `tromoi_scraper.ipynb`

2. **Làm sạch dữ liệu (Cleaning):**
   Chạy 4 notebook trong thư mục `notebooks/cleaning/` để làm sạch và lưu vào `data/clear`.

3. **Xử lý và Khám phá dữ liệu (EDA Preprocessing):**
   Chạy file `notebooks/eda/eda_preprocessing.ipynb`. Quá trình này sẽ xử lý dữ liệu và lưu file kết quả `data_clean.csv` vào `data/processed`.

### Bước 2: Phân tích câu hỏi (Business Questions)
Để xem các phân tích và trả lời cho các câu hỏi nghiệp vụ, hãy chạy lần lượt 5 file sau trong thư mục `notebooks/eda/`:
* `question01.ipynb`
* `question02.ipynb`
* `question03.ipynb`
* `question04.ipynb`
* `question05.ipynb`

### Bước 3: Huấn luyện Mô hình (Training)
Để huấn luyện mô hình dự đoán:
1. Mở thư mục `notebooks/modeling/`.
2. Chạy file `modeling.ipynb`.
3. **Kết quả:** File `best_model.joblib` và `best_model_meta.json` sẽ được tạo ra để phục vụ cho GUI.

### Bước 4: Chạy Ứng dụng Dự đoán (GUI)
Để mở giao diện phần mềm, bạn cần chạy 2 file theo đúng thứ tự sau:

1. **Chạy cấu hình:**
   ```bash
   python GUI/categorical_option.py
   ```
2. **Mở giao diện:**
   ```bash
   python GUI/gui.py
   ```