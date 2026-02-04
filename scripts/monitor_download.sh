#!/bin/bash

# Monitor download progress

echo "=== Download Progress Monitor ==="
echo ""

# Check if process is running
if pgrep -f "download_pdfs_to_images.py" > /dev/null; then
    echo "✅ Download process is RUNNING"
else
    echo "❌ Download process is NOT running"
fi

echo ""
echo "--- Latest Progress ---"
tail -20 data/raw/finepdfs_full_progress.log | grep -E "(INFO|%|complete|Total pages)"

echo ""
echo "--- Statistics So Far ---"
if [ -f data/raw/finepdfs_full/statistics.json ]; then
    cat data/raw/finepdfs_full/statistics.json
else
    echo "Statistics file not yet created"
fi

echo ""
echo "--- Images Downloaded ---"
if [ -d data/raw/finepdfs_full/images ]; then
    IMAGE_COUNT=$(ls data/raw/finepdfs_full/images/ 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh data/raw/finepdfs_full/images/ 2>/dev/null | cut -f1)
    echo "Images: $IMAGE_COUNT files"
    echo "Size: $TOTAL_SIZE"
else
    echo "No images directory yet"
fi

echo ""
echo "--- Full Log ---"
echo "To view full log: tail -f data/raw/finepdfs_full_progress.log"
echo "To stop download: pkill -f download_pdfs_to_images.py"
