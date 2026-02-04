# MeddiesOCR

OCR project for medical documents using table detection and recognition.

## Setup

### 1. Clone Repository
```bash
git clone git@github.com:meddies-ai/meddies-ocr.git
cd meddies-ocr
```

### 2. Create Conda Environment
```bash
conda create -n data python=3.10
conda activate data
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Copy [.env.example](.env.example) to `.env` and fill in your tokens:
```bash
cp .env.example .env
# Edit .env file with your tokens
```

Required tokens:
- `HF_TOKEN`: Hugging Face API token from https://huggingface.co/settings/tokens
- `NOTION_API_KEY`: Notion integration token (if using Notion)

### 4. Download Data from Google Drive

Data is stored separately on Google Drive: [MeddiesOCR Data](https://drive.google.com/drive/u/2/folders/15mf7pVt9sHxxeqFvtYHKLpesujW4vySy)

**Option 1: Manual Download**
1. Download the `data` folder from Google Drive
2. Place it in the project root: `MeddiesOCR/data/`

**Option 2: Using gdown**
```bash
pip install gdown

# Download entire folder
gdown --folder https://drive.google.com/drive/folders/15mf7pVt9sHxxeqFvtYHKLpesujW4vySy -O data/
```

**Expected data structure:**
```
data/
├── raw/
│   ├── finepdfs_full/
│   ├── finepdfs_images/
│   └── finepdfs_vie/
├── processed/
└── annotations/
```

## Project Structure

```
MeddiesOCR/
├── configs/          # Configuration files
├── data/            # Data (stored on Google Drive)
├── notebooks/       # Jupyter notebooks
├── scripts/         # Utility scripts
├── src/            # Source code
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── training/
└── tests/          # Unit tests
```

## Development Workflow

### Code Style
Before committing:
```bash
# Format code
black .

# Check style
flake8 .
```

### Commit Convention
Format: `MP-X: Short description`
- X is the issue/task number
- Write in English
- Keep it concise

Example:
```bash
git commit -m "MP-366: Add table detection preprocessing"
```

## Data Management

### Uploading New Data to Google Drive
1. Organize data in the `data/` folder locally
2. Upload to the [Google Drive folder](https://drive.google.com/drive/u/2/folders/15mf7pVt9sHxxeqFvtYHKLpesujW4vySy)
3. Update this README if structure changes

### Important Notes
- Never commit large data files to GitHub
- Data files are ignored by `.gitignore`
- Keep data folder structure consistent across team members
