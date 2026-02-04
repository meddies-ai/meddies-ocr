# Hướng Dẫn Setup Project

## Phân Tách Code và Data

### 🔧 Code → GitHub
- Tất cả source code, config, scripts
- Requirements, notebooks
- Documentation

### 💾 Data → Google Drive
- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Annotations: `data/annotations/`
- Log files: `*.log`

Link Google Drive: https://drive.google.com/drive/u/2/folders/15mf7pVt9sHxxeqFvtYHKLpesujW4vySy

## Bước 1: Setup GitHub Repository

```bash
# Khởi tạo git (nếu chưa có)
cd d:\Project\MeddiesOCR
git init

# Add remote repository
git remote add origin git@github.com:meddies-ai/meddies-ocr.git

# Add và commit code
git add .
git commit -m "MP-1: Initial project setup"

# Push lên GitHub
git push -u origin main
```

## Bước 2: Upload Data lên Google Drive

### Cách 1: Upload thủ công qua Web
1. Mở Google Drive: https://drive.google.com/drive/u/2/folders/15mf7pVt9sHxxeqFvtYHKLpesujW4vySy
2. Tạo cấu trúc thư mục:
   ```
   MeddiesOCR_Data/
   ├── raw/
   ├── processed/
   └── annotations/
   ```
3. Upload từng folder từ `d:\Project\MeddiesOCR\data\`

### Cách 2: Sử dụng Google Drive Desktop
1. Cài đặt [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Đồng bộ folder `data/` với Google Drive
3. Share link với team members

### Cách 3: Sử dụng rclone (Advanced)
```bash
# Cài đặt rclone
# Windows: download từ https://rclone.org/downloads/

# Configure Google Drive
rclone config

# Sync data folder
rclone sync d:\Project\MeddiesOCR\data\ gdrive:MeddiesOCR_Data/
```

## Bước 3: Team Members Clone và Setup

Khi team member khác muốn làm việc:

```bash
# Clone repository từ GitHub
git clone git@github.com:meddies-ai/meddies-ocr.git
cd meddies-ocr

# Setup conda environment
conda create -n data python=3.10
conda activate data
pip install -r requirements.txt

# Download data từ Google Drive
# Sử dụng một trong các cách ở trên để download data
# và đặt vào thư mục data/
```

## Tips

### Gitignore đã được cấu hình
File `.gitignore` đã được setup để:
- ✅ Commit: code, configs, notebooks (không có output)
- ❌ Không commit: data files, logs, cache, model checkpoints

### Code Style
```bash
# Format trước khi commit
black .
flake8 .

# Commit format
git commit -m "MP-X: Description"
```

### Data Updates
Nếu data thay đổi:
1. Upload phiên bản mới lên Google Drive
2. Thông báo cho team qua chat/issue
3. Team members re-download data mới
