"""
Download PDFs from finepdfs dataset URLs and convert to images.

Strategy:
1. Download PDFs from URLs
2. Convert each page to image using pdf2image
3. Split text using page_ends to get per-page ground truth
4. Save images + annotations
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import json
import requests
from io import BytesIO
import time

from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
from PIL import Image

# PDF processing
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Install with: pip install pdf2image")

# Alternative: PyMuPDF (faster)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Install with: pip install pymupdf")


def download_pdf(url: str, timeout: int = 30) -> Optional[bytes]:
    """
    Download PDF from URL.

    Args:
        url: PDF URL
        timeout: Request timeout in seconds

    Returns:
        PDF bytes or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        # Check if actually PDF
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower():
            logger.warning(f"URL does not return PDF: {content_type}")
            return None

        return response.content

    except requests.RequestException as e:
        logger.debug(f"Failed to download PDF: {e}")
        return None


def pdf_to_images_pymupdf(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    """
    Convert PDF to images using PyMuPDF (faster).

    Args:
        pdf_bytes: PDF file bytes
        dpi: Resolution for rendering

    Returns:
        List of PIL Images
    """
    images = []

    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Convert DPI to zoom factor (72 DPI is base)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append(img)

        pdf_document.close()

    except Exception as e:
        logger.error(f"PyMuPDF conversion failed: {e}")

    return images


def pdf_to_images_pdf2image(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    """
    Convert PDF to images using pdf2image (requires poppler).

    Args:
        pdf_bytes: PDF file bytes
        dpi: Resolution for rendering

    Returns:
        List of PIL Images
    """
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        return images
    except Exception as e:
        logger.error(f"pdf2image conversion failed: {e}")
        return []


def split_text_by_pages(text: str, page_ends: List[int]) -> List[str]:
    """
    Split full document text into per-page texts using page_ends.

    Args:
        text: Full document text
        page_ends: List of character positions marking page boundaries

    Returns:
        List of per-page texts
    """
    if not page_ends:
        return [text]

    page_texts = []
    start = 0

    for end in page_ends:
        page_text = text[start:end].strip()
        page_texts.append(page_text)
        start = end

    return page_texts


def process_dataset(
    output_dir: str = "data/raw/finepdfs_images",
    max_samples: int = None,
    max_pages_per_pdf: int = 10,
    dpi: int = 150,
    retry_delay: int = 2
) -> Dict:
    """
    Process finepdfs dataset: download PDFs and convert to images.

    Args:
        output_dir: Output directory
        max_samples: Maximum PDFs to process
        max_pages_per_pdf: Skip PDFs with more pages
        dpi: Image resolution
        retry_delay: Delay between requests (seconds)

    Returns:
        Processing statistics
    """
    # Check dependencies
    if not (PYMUPDF_AVAILABLE or PDF2IMAGE_AVAILABLE):
        raise RuntimeError(
            "No PDF converter available. Install one of:\n"
            "  pip install pymupdf  (recommended, faster)\n"
            "  pip install pdf2image  (requires poppler)"
        )

    use_pymupdf = PYMUPDF_AVAILABLE
    logger.info(f"Using PDF converter: {'PyMuPDF' if use_pymupdf else 'pdf2image'}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    # Load dataset
    logger.info("Loading finepdfs Vietnamese dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/finepdfs",
        "vie_Latn",
        split="train",
        streaming=True
    )

    # Take only needed samples if max_samples specified
    if max_samples is not None:
        dataset = dataset.take(max_samples)
        logger.info(f"Limited to {max_samples} samples")
    else:
        logger.info("Processing ALL samples (no limit)")

    # Statistics
    stats = {
        "total_attempted": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "total_pages": 0,
        "skipped_too_many_pages": 0,
        "conversion_errors": 0
    }

    annotations = []
    image_counter = 0

    if max_samples:
        logger.info(f"Processing up to {max_samples} PDFs...")
    else:
        logger.info("Processing all available PDFs...")

    # Use tqdm with total only if max_samples specified
    iterator = tqdm(dataset, total=max_samples) if max_samples else tqdm(dataset)

    for sample_idx, sample in enumerate(iterator):
        stats["total_attempted"] += 1

        url = sample.get('url')
        text = sample.get('text', '')
        page_ends = sample.get('page_ends', [])
        doc_id = sample.get('id', f'doc_{sample_idx}')

        if not url:
            logger.debug(f"Sample {sample_idx}: No URL")
            continue

        # Check page count
        num_pages = len(page_ends)
        if num_pages > max_pages_per_pdf:
            logger.debug(f"Skipping {url}: {num_pages} pages (max: {max_pages_per_pdf})")
            stats["skipped_too_many_pages"] += 1
            continue

        # Download PDF
        logger.debug(f"Downloading: {url}")
        pdf_bytes = download_pdf(url)

        if not pdf_bytes:
            stats["failed_downloads"] += 1
            time.sleep(retry_delay)  # Rate limiting
            continue

        stats["successful_downloads"] += 1

        # Convert to images
        try:
            if use_pymupdf:
                images = pdf_to_images_pymupdf(pdf_bytes, dpi=dpi)
            else:
                images = pdf_to_images_pdf2image(pdf_bytes, dpi=dpi)

            if not images:
                stats["conversion_errors"] += 1
                continue

        except Exception as e:
            logger.error(f"Conversion error for {url}: {e}")
            stats["conversion_errors"] += 1
            continue

        # Split text by pages
        page_texts = split_text_by_pages(text, page_ends)

        # Ensure we have text for each page (pad if needed)
        while len(page_texts) < len(images):
            page_texts.append("")

        # Save images and create annotations
        for page_num, (img, page_text) in enumerate(zip(images, page_texts)):
            image_filename = f"image_{image_counter:06d}.jpg"
            image_path = images_dir / image_filename

            # Save image
            img.save(image_path, 'JPEG', quality=95)

            # Create annotation
            annotation = {
                "image_id": image_counter,
                "image_path": str(image_path.relative_to(output_path)),
                "text": page_text,
                "is_blank": len(page_text.strip()) == 0,
                "page_number": page_num,
                "total_pages": len(images),
                "source_url": url,
                "source_doc_id": doc_id,
                "image_size": img.size,  # (width, height)
            }
            annotations.append(annotation)

            image_counter += 1
            stats["total_pages"] += 1

        # Rate limiting
        time.sleep(retry_delay)

    # Save annotations
    annotations_file = output_path / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("Processing complete!")
    logger.info(f"{'='*60}")
    logger.info(f"PDFs attempted: {stats['total_attempted']}")
    logger.info(f"Successful downloads: {stats['successful_downloads']}")
    logger.info(f"Failed downloads: {stats['failed_downloads']}")
    logger.info(f"Skipped (too many pages): {stats['skipped_too_many_pages']}")
    logger.info(f"Conversion errors: {stats['conversion_errors']}")
    logger.info(f"Total pages extracted: {stats['total_pages']}")
    logger.info(f"Saved to: {output_path}")

    return stats


def preview_dataset(annotations_file: str, num_samples: int = 5):
    """Preview samples from processed dataset."""
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Preview - First {num_samples} samples")
    logger.info(f"{'='*60}\n")

    for i, ann in enumerate(annotations[:num_samples]):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Image: {ann['image_path']}")
        logger.info(f"  Size: {ann['image_size']}")
        logger.info(f"  Page: {ann['page_number']+1}/{ann['total_pages']}")
        logger.info(f"  Text length: {len(ann['text'])} chars")
        logger.info(f"  Is blank: {ann['is_blank']}")
        if not ann['is_blank'] and len(ann['text']) < 200:
            logger.info(f"  Text preview: {ann['text'][:200]}...")
        logger.info(f"  Source: {ann['source_url'][:80]}...")
        logger.info("")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDFs from finepdfs and convert to images"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/finepdfs_images",
        help="Output directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of PDFs to process (None = all)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Skip PDFs with more than this many pages"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image DPI (higher = better quality but larger files)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview samples after processing"
    )

    args = parser.parse_args()

    # Check HF token from environment
    if 'HF_TOKEN' not in os.environ:
        raise ValueError('HF_TOKEN environment variable not set')

    # Process dataset
    stats = process_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_pages_per_pdf=args.max_pages,
        dpi=args.dpi
    )

    # Preview if requested
    if args.preview:
        annotations_file = Path(args.output_dir) / "annotations.json"
        if annotations_file.exists():
            preview_dataset(str(annotations_file))
