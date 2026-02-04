"""
Download and prepare finepdfs Vietnamese dataset from HuggingFace.

This script:
1. Downloads the HuggingFaceFW/finepdfs dataset (vie_Latn subset)
2. Saves images and annotations to data/raw/
3. Creates initial data statistics
"""

import os
from pathlib import Path
from typing import Dict, List
import json

from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from loguru import logger


def download_finepdfs_vietnamese(
    output_dir: str = "data/raw/finepdfs_vie",
    max_samples: int = None,
    split: str = "train"
) -> Dict:
    """
    Download finepdfs Vietnamese dataset.

    Args:
        output_dir: Directory để lưu dataset
        max_samples: Giới hạn số samples (None = tải hết)
        split: Dataset split (train/test/validation)

    Returns:
        Statistics dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading finepdfs Vietnamese dataset from HuggingFace...")

    # Load dataset - use streaming if max_samples specified for faster download
    use_streaming = max_samples is not None and max_samples < 10000

    dataset = load_dataset(
        "HuggingFaceFW/finepdfs",
        "vie_Latn",  # Vietnamese Latin script
        split=split,
        streaming=use_streaming
    )

    if use_streaming:
        logger.info(f"Using streaming mode, will process {max_samples} samples")
        # Take only needed samples from stream
        dataset = dataset.take(max_samples)
    else:
        logger.info(f"Dataset loaded. Total samples: {len(dataset)}")
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited to {len(dataset)} samples")

    # Process and save samples
    annotations = []
    stats = {
        "total_samples": 0,
        "blank_images": 0,
        "text_samples": 0,
        "avg_text_length": 0,
        "total_text_chars": 0
    }

    logger.info("Processing samples...")
    for idx, sample in enumerate(tqdm(dataset)):
        try:
            # finepdfs structure: {'image': PIL.Image, 'text': str, 'metadata': dict}
            image = sample.get('image')
            text = sample.get('text', '').strip()
            metadata = sample.get('metadata', {})

            # Save image
            image_filename = f"sample_{idx:06d}.jpg"
            image_path = images_dir / image_filename

            if image is not None:
                if isinstance(image, Image.Image):
                    image.save(image_path, 'JPEG', quality=95)
                else:
                    logger.warning(f"Sample {idx}: Image not PIL.Image type")
                    continue
            else:
                logger.warning(f"Sample {idx}: No image found")
                continue

            # Create annotation entry
            annotation = {
                "image_id": idx,
                "image_path": str(image_path.relative_to(output_path)),
                "text": text,
                "is_blank": len(text) == 0,
                "metadata": metadata
            }
            annotations.append(annotation)

            # Update statistics
            stats["total_samples"] += 1
            if len(text) == 0:
                stats["blank_images"] += 1
            else:
                stats["text_samples"] += 1
                stats["total_text_chars"] += len(text)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue

    # Calculate final statistics
    if stats["text_samples"] > 0:
        stats["avg_text_length"] = stats["total_text_chars"] / stats["text_samples"]

    stats["blank_ratio"] = stats["blank_images"] / stats["total_samples"] if stats["total_samples"] > 0 else 0

    # Save annotations
    annotations_file = output_path / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset downloaded successfully!")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Text samples: {stats['text_samples']}")
    logger.info(f"Blank images: {stats['blank_images']} ({stats['blank_ratio']:.2%})")
    logger.info(f"Average text length: {stats['avg_text_length']:.1f} characters")
    logger.info(f"Saved to: {output_path}")

    return stats


def preview_samples(annotations_file: str, num_samples: int = 5):
    """
    Preview một số samples từ dataset.
    """
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Preview - First {num_samples} samples")
    logger.info(f"{'='*60}\n")

    for i, ann in enumerate(annotations[:num_samples]):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Image: {ann['image_path']}")
        logger.info(f"  Text length: {len(ann['text'])} chars")
        logger.info(f"  Is blank: {ann['is_blank']}")
        if not ann['is_blank'] and len(ann['text']) < 200:
            logger.info(f"  Text: {ann['text'][:200]}...")
        logger.info("")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download finepdfs Vietnamese dataset")
    parser.add_argument(
        "--output-dir",
        default="data/raw/finepdfs_vie",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download (None = all)"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview samples after downloading"
    )

    args = parser.parse_args()

    # Download dataset
    stats = download_finepdfs_vietnamese(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split=args.split
    )

    # Preview if requested
    if args.preview:
        annotations_file = Path(args.output_dir) / "annotations.json"
        preview_samples(str(annotations_file))
