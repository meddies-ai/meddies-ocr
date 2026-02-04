"""
Explore the structure of finepdfs dataset to understand its schema.
"""

from datasets import load_dataset
from loguru import logger
import os

# Ensure HF_TOKEN is set in your environment
if 'HF_TOKEN' not in os.environ:
    raise ValueError('HF_TOKEN environment variable not set. Please set it before running.')

logger.info("Loading finepdfs Vietnamese dataset...")

# Load with streaming to check structure quickly
dataset = load_dataset(
    "HuggingFaceFW/finepdfs",
    "vie_Latn",
    split="train",
    streaming=True
)

logger.info("Dataset loaded. Examining first sample...")

# Get first sample
sample = next(iter(dataset))

logger.info(f"Sample keys: {sample.keys()}")
logger.info(f"\nSample structure:")
for key, value in sample.items():
    logger.info(f"  {key}: {type(value)} - {str(value)[:200] if value else 'None'}")

# Try to understand the data
logger.info(f"\n{'='*60}")
logger.info("Detailed examination:")
logger.info(f"{'='*60}")

for key in sample.keys():
    value = sample[key]
    logger.info(f"\nKey: {key}")
    logger.info(f"Type: {type(value)}")

    if isinstance(value, dict):
        logger.info(f"Dict keys: {value.keys()}")
        for k, v in value.items():
            logger.info(f"  {k}: {type(v)} = {str(v)[:100]}")
    elif isinstance(value, str):
        logger.info(f"Length: {len(value)} chars")
        logger.info(f"Preview: {value[:200]}")
    else:
        logger.info(f"Value: {value}")
