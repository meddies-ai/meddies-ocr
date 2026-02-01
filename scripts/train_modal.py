import os
import sys
import modal

# Define Modal Image with source code copied in
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        # LFM2.5-VL requires a specific transformers commit (not the PyPI version)
        "git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe",
        "accelerate",
        "datasets",
        "wandb",
        "pydantic",
        "pyyaml",
        "trl",  # For SFTTrainer and GOLDTrainer
        "pillow",  # For image processing
        "torchvision",  # Required for LFM2.5-VL processor
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("meddies-ocr-two-stage-training", image=image)

# Create a volume for persisting checkpoints
volume = modal.Volume.from_name("meddies-checkpoints", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("meddies-ocr")],
    volumes={"/checkpoints": volume},
)
def stage1_sft(dataset_name: str = None, max_steps: int = 100):
    """
    Stage 1: Supervised Fine-Tuning (SFT) for cold start
    Fine-tune LFM2.5-VL student model on task-specific data
    """
    # Set up Python Path
    sys.path.insert(0, "/root")

    # Import after sys.path is configured
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoConfig,
    )
    import traceback
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer, SFTConfig
    from src.common.logging_utils import get_logger
    from PIL import Image

    logger = get_logger("sft_stage")
    logger.info("=" * 50)
    logger.info("STAGE 1: SFT Cold Start")
    logger.info("=" * 50)

    # Load student model
    student_model_name = "LiquidAI/LFM2.5-VL-1.6B"
    logger.info(f"Loading student model: {student_model_name}")

    try:
        # Load config first to inspect if needed, though mostly for debugging here
        config = AutoConfig.from_pretrained(student_model_name, trust_remote_code=True)
        logger.info(f"Loaded config: {type(config)}")
    except Exception as e:
        logger.warning(f"Could not load config explicitly: {e}")

    try:
        # Try AutoModelForImageTextToText first (recommended for LFM2.5-VL)
        logger.info("Attempting with AutoModelForImageTextToText...")
        student = AutoModelForImageTextToText.from_pretrained(
            student_model_name, trust_remote_code=True, torch_dtype="auto"
        )
        processor = AutoProcessor.from_pretrained(
            student_model_name, trust_remote_code=True
        )
    except Exception as e1:
        logger.warning(f"AutoModelForImageTextToText failed: {e1}")
        logger.warning(traceback.format_exc())
        
        try:
            # Try AutoModelForCausalLM (common for newer VLMs)
            logger.info("Attempting with AutoModelForCausalLM...")
            student = AutoModelForCausalLM.from_pretrained(
                student_model_name, trust_remote_code=True, torch_dtype="auto"
            )
            processor = AutoProcessor.from_pretrained(
                student_model_name, trust_remote_code=True
            )
        except Exception as e2:
            logger.warning(f"AutoModelForCausalLM failed: {e2}")
            
            try:
                # Fallback to Vision2Seq
                logger.info("Attempting with AutoModelForVision2Seq...")
                student = AutoModelForVision2Seq.from_pretrained(
                    student_model_name, trust_remote_code=True, torch_dtype="auto"
                )
                processor = AutoProcessor.from_pretrained(
                    student_model_name, trust_remote_code=True
                )
            except Exception as e3:
                logger.error(f"AutoModelForVision2Seq failed: {e3}")
                logger.info("Falling back to generic AutoModel...")
                # Last resort fallback, though likely won't work for training if head is missing
                from transformers import AutoModel, AutoTokenizer
    
                student = AutoModel.from_pretrained(
                    student_model_name, trust_remote_code=True, torch_dtype="auto"
                )
                try:
                    processor = AutoProcessor.from_pretrained(
                        student_model_name, trust_remote_code=True
                    )
                except:
                    processor = AutoTokenizer.from_pretrained(
                        student_model_name, trust_remote_code=True
                    )

    # Ensure processor has pad token
    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # PATCH: property injection for SFTTrainer
        if not hasattr(processor, "pad_token_id"):
            processor.pad_token_id = processor.tokenizer.pad_token_id
            
    elif hasattr(processor, "pad_token"):
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        if not hasattr(processor, "pad_token_id") and hasattr(processor, "token_id"): # generic fallback
            processor.pad_token_id = processor.eos_token_id

    logger.info(f"Student model loaded: {type(student)}")
    logger.info(
        "NOTE: If you see warnings about 'some weights not being initialized', this is EXPECTED when fine-tuning a base model on a new task."
    )

    # Prepare dataset
    logger.info("Preparing training dataset...")
    if not dataset_name:
        dataset_name = "mychen76/wildreceipts_ocr_train"
        logger.info(f"Using default dataset: {dataset_name}")

    train_dataset = load_dataset(dataset_name, split="train[:100]")
    
    def transform_dataset(example):
        # Image handling: SFTTrainer vision expects 'images' column as a list
        if "image" in example:
            example["images"] = [example["image"]]
        elif "images" in example:
            if not isinstance(example["images"], list):
                example["images"] = [example["images"]]
        else:
            # Create a dummy image if none exists (safety)
            example["images"] = [Image.new("RGB", (224, 224), color="white")]
        
        # Text handling
        json_str = ""
        if "ocr_json" in example:
            json_str = str(example["ocr_json"])
        elif "text" in example:
            json_str = example["text"]
        
        # Format as 'messages' for SFTTrainer to automaticallly handle VLM tokenization
        example["messages"] = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract the key information from this receipt in JSON format."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json_str}]
            }
        ]
        
        return example

    logger.info("Transforming dataset...")
    train_dataset = train_dataset.map(transform_dataset, remove_columns=["text", "ocr_json"] if "ocr_json" in train_dataset.column_names else ["text"])
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Dataset columns: {train_dataset.column_names}")

    # Configure SFT
    sft_config = SFTConfig(
        output_dir="/checkpoints/sft",
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # Reduced to 1 to save memory
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        max_steps=max_steps,
        max_length=4096,  # 4096 with batch_size=1 + checkpointing to fit high-res images
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        report_to="none",  # Disable wandb for now
        remove_unused_columns=False,  # Keep image column
        # dataset_text_field="text",  # Removed to let SFTTrainer use 'messages'
    )

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=student,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info("Saving SFT checkpoint...")
    trainer.save_model("/checkpoints/sft/final")
    processor.save_pretrained("/checkpoints/sft/final")

    # Commit volume changes
    volume.commit()

    logger.info("Stage 1 (SFT) completed successfully!")
    return "/checkpoints/sft/final"


@app.function(
    gpu="A10G",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("meddies-ocr")],
    volumes={"/checkpoints": volume},
)
def stage2_gold(sft_checkpoint_path: str, max_steps: int = 500):
    """
    Stage 2: GOLD Distillation
    Distill knowledge from Chandra teacher to SFT'd student using Universal Logit Distillation
    """
    # Set up Python Path
    sys.path.insert(0, "/root")

    # Import after sys.path is configured
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoConfig,
    )
    import traceback
    from datasets import load_dataset, Dataset
    from trl.experimental.gold import GOLDTrainer, GOLDConfig
    from src.common.logging_utils import get_logger

    logger = get_logger("gold_stage")
    logger.info("=" * 50)
    logger.info("STAGE 2: GOLD Distillation")
    logger.info("=" * 50)

    # Load SFT checkpoint as student
    logger.info(f"Loading SFT checkpoint from: {sft_checkpoint_path}")
    try:
        # Try AutoModelForImageTextToText first
        logger.info("Attempting with AutoModelForImageTextToText...")
        student = AutoModelForImageTextToText.from_pretrained(
            sft_checkpoint_path, trust_remote_code=True, torch_dtype="auto"
        )
        processor = AutoProcessor.from_pretrained(
            sft_checkpoint_path, trust_remote_code=True
        )
    except Exception as e1:
        logger.warning(f"AutoModelForImageTextToText failed: {e1}")
        logger.warning(traceback.format_exc())
        
        try:
            # Try AutoModelForCausalLM
            logger.info("Attempting with AutoModelForCausalLM...")
            student = AutoModelForCausalLM.from_pretrained(
                sft_checkpoint_path, trust_remote_code=True, torch_dtype="auto"
            )
            processor = AutoProcessor.from_pretrained(
                sft_checkpoint_path, trust_remote_code=True
            )
        except Exception as e2:
            logger.warning(f"AutoModelForCausalLM failed: {e2}")
            
            try:
                logger.info("Attempting with AutoModelForVision2Seq...")
                student = AutoModelForVision2Seq.from_pretrained(
                    sft_checkpoint_path, trust_remote_code=True, torch_dtype="auto"
                )
                processor = AutoProcessor.from_pretrained(
                    sft_checkpoint_path, trust_remote_code=True
                )
            except Exception as e3:
                logger.error(f"AutoModelForVision2Seq failed: {e3}")
                logger.info("Falling back to generic AutoModel...")
                student = AutoModel.from_pretrained(
                    sft_checkpoint_path, trust_remote_code=True, torch_dtype="auto"
                )
                try:
                    processor = AutoProcessor.from_pretrained(
                        sft_checkpoint_path, trust_remote_code=True
                    )
                except:
                    processor = AutoTokenizer.from_pretrained(
                        sft_checkpoint_path, trust_remote_code=True
                    )

    # Ensure processor has pad token
    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # PATCH: property injection for SFTTrainer
        if not hasattr(processor, "pad_token_id"):
            processor.pad_token_id = processor.tokenizer.pad_token_id
            
    elif hasattr(processor, "pad_token"):
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        if not hasattr(processor, "pad_token_id"):
            processor.pad_token_id = processor.eos_token_id

    logger.info(f"Student loaded: {type(student)}")

    # Teacher model
    teacher_model_name = "datalab-to/chandra"
    logger.info(f"Teacher model: {teacher_model_name}")

    # Prepare dataset
    # Prepare dataset
    logger.info("Preparing distillation dataset...")
    
    # Use the same dataset for distillation to ensure domain consistency
    dataset_name = "mychen76/wildreceipts_ocr_train"
    gold_dataset = load_dataset(dataset_name, split="train[:100]") # Use subset for demo, or full for real
    
    def transform_gold_dataset(example):
        # Image handling
        if "image" in example:
            example["images"] = [example["image"]]
        elif "images" in example:
            if not isinstance(example["images"], list):
                example["images"] = [example["images"]]
        else:
            example["images"] = [Image.new("RGB", (224, 224), color="white")]
            
        # Text handling - For GOLD, strictly we might just need the input prompt and let the model generate?
        # But GOLDTrainer usually takes the same dataset format as SFTTrainer (text field with completion) within TRL?
        # Actually GOLD calculates NLL on the target. So we provide the full text (Prompt + Completion).
        
        json_str = ""
        if "ocr_json" in example:
            json_str = str(example["ocr_json"])
        elif "text" in example:
            json_str = example["text"]
            
        example["messages"] = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract the key information from this receipt in JSON format."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json_str}]
            }
        ]
        return example

    logger.info("Transforming GOLD dataset...")
    train_dataset = gold_dataset.map(transform_gold_dataset, remove_columns=["text", "ocr_json"] if "ocr_json" in gold_dataset.column_names else ["text"])
    
    logger.info(f"Dataset size: {len(train_dataset)}")

    # Configure GOLD
    gold_config = GOLDConfig(
        output_dir="/checkpoints/gold",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        use_uld_loss=True,  # Enable Universal Logit Distillation
        teacher_model_name_or_path=teacher_model_name,
        max_length=4096, # Increased to match SFT
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True, # Enable gradient checkpointing
        report_to="none",
    )

    logger.info("Initializing GOLDTrainer...")
    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher_model_name,
        args=gold_config,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    logger.info("Starting GOLD distillation training...")
    trainer.train()

    logger.info("Saving final distilled model...")
    trainer.save_model("/checkpoints/gold/final")
    processor.save_pretrained("/checkpoints/gold/final")

    # Commit volume changes
    volume.commit()

    logger.info("Stage 2 (GOLD) completed successfully!")
    return "/checkpoints/gold/final"


@app.function(
    gpu="A10G",
    timeout=14400,  # 4 hours for both stages
    secrets=[modal.Secret.from_name("meddies-ocr")],
    volumes={"/checkpoints": volume},
)
def train_two_stage(
    dataset_name: str = None, sft_steps: int = 100, gold_steps: int = 500
):
    """
    Run complete two-stage training pipeline:
    1. SFT cold start
    2. GOLD distillation
    """
    import sys

    sys.path.insert(0, "/root")
    from src.common.logging_utils import get_logger

    logger = get_logger("two_stage_training")
    logger.info("Starting two-stage training pipeline...")

    # Stage 1: SFT
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTING STAGE 1: SFT Cold Start")
    logger.info("=" * 60)
    sft_checkpoint = stage1_sft.local(dataset_name, sft_steps)

    # Stage 2: GOLD
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTING STAGE 2: GOLD Distillation")
    logger.info("=" * 60)
    final_model = stage2_gold.local(sft_checkpoint, gold_steps)

    logger.info("\n" + "=" * 60)
    logger.info("TWO-STAGE TRAINING COMPLETED!")
    logger.info(f"Final model saved at: {final_model}")
    logger.info("=" * 60)

    return final_model


@app.local_entrypoint()
def main(
    stage: str = "all", dataset: str = None, sft_steps: int = 100, gold_steps: int = 500
):
    """
    Main entrypoint for training

    Args:
        stage: Which stage to run - 'sft', 'gold', or 'all'
        dataset: Optional dataset name from HuggingFace
        sft_steps: Number of steps for SFT stage
        gold_steps: Number of steps for GOLD stage
    """
    print(f"Running training stage: {stage}")

    if stage == "all":
        print("Executing full two-stage training pipeline...")
        result = train_two_stage.remote(dataset, sft_steps, gold_steps)
        print(f"Training complete! Final model: {result}")
    elif stage == "sft":
        print("Running Stage 1: SFT only...")
        result = stage1_sft.remote(dataset, sft_steps)
        print(f"SFT complete! Checkpoint: {result}")
    elif stage == "gold":
        print("Running Stage 2: GOLD only...")
        sft_path = "/checkpoints/sft/final"
        result = stage2_gold.remote(sft_path, gold_steps)
        print(f"GOLD complete! Model: {result}")
    else:
        print(f"Error: Unknown stage '{stage}'. Use 'sft', 'gold', or 'all'")
