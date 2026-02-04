#!/bin/bash

# Download full finepdfs Vietnamese dataset
# Run in background with nohup to prevent interruption

# Set your Hugging Face token
# export HF_TOKEN=your_token_here

# Full dataset - no max_samples limit
# Increase max_pages to 20 for more data
# Higher DPI for better quality
python scripts/download_pdfs_to_images.py \
    --output-dir data/raw/finepdfs_full \
    --max-pages 20 \
    --dpi 200 \
    2>&1 | tee data/raw/finepdfs_full_download.log

echo "Download complete! Check data/raw/finepdfs_full_download.log for details"
