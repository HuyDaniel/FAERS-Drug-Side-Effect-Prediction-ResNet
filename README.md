💊 FAERS Drug Side Effect Prediction using DrugResNet & SMOTENC

Dự án nghiên cứu ứng dụng Học sâu (Deep Learning) để dự báo mức độ nghiêm trọng của phản ứng phụ từ thuốc dựa trên bộ dữ liệu FDA FAERS 2025Q4.

🚀 Key Features

Architecture: Kiến trúc mạng thặng dư (DrugResNet) với cơ chế Skip Connections giúp huấn luyện sâu và bảo toàn thông tin.

Data Balancing: Ứng dụng thuật toán SMOTENC để xử lý dữ liệu bảng hỗn hợp bị mất cân bằng lớp nghiêm trọng.

Optimization: Tinh chỉnh mô hình với Cosine Annealing Learning Rate và Weight Decay để đạt độ chính xác tối ưu sau 300 Epochs.

📁 Project Structure

├── data/           # Bộ dữ liệu cleaned_data.csv (FDA 2025Q4)
├── docs/           # Báo cáo đồ án (.docx) và Slide thuyết trình
├── results/        # Biểu đồ Accuracy/Loss và kết quả huấn luyện
├── src/            # Mã nguồn tiền xử lý và huấn luyện (PyTorch)
└── venv/           # Môi trường ảo (Virtual Environment)

📊 Results

Mô hình đạt hiệu suất ấn tượng sau 300 epochs huấn luyện:

Train Accuracy: ~99%

Validation Accuracy: ~98%

Test Accuracy (Real-world): ~67.65%

C:\Users\ngoqu\Desktop\DRUGF\DrugAddiction_DeepLearning\results\smote_300_epochs_charts_fixed (fitlab version).png

🛠️ Installation
Bash
# Clone dự án
git clone https://github.com/your_username/repo_name.git

# Cài đặt thư viện
pip install -r requirements.txt
