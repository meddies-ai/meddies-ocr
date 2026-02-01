import modal
import sys

# Reuse the image definition from train_modal.py
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe",
        "accelerate",
        "datasets",
        "wandb",
        "pydantic",
        "pyyaml",
        "trl",
        "pillow",
        "torchvision",
    )
    # .add_local_dir("src", remote_path="/root/src") # Not strictly needed for this debug
)

app = modal.App("debug-lfm-processor", image=image)

@app.function(gpu="A10G", timeout=600)
def debug_processor():
    from transformers import AutoProcessor, AutoConfig
    from PIL import Image
    import torch
    
    model_name = "LiquidAI/LFM2.5-VL-1.6B"
    print(f"Loading processor for {model_name}...")
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    print("\nProcessor Config:")
    print(processor)
    
    print("\nImage Processor Config:")
    if hasattr(processor, "image_processor"):
        print(processor.image_processor)
        
    print(f"\nModel Config Image Token ID: {getattr(config, 'image_token_id', 'Not Found')}")
    
    # Create dummy image
    dummy_image = Image.new("RGB", (224, 224), color="white")
    
    # Test 1: Chat Template
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is a white image."}]
        }
    ]
    
    print("\n--- Test 1: processing with apply_chat_template ---")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    print(f"Prompt output type: {type(prompt)}")
    print(f"Prompt content: {prompt}")
    
    # Test 2: Processor Call
    print("\n--- Test 2: processor(text=prompt, images=image) ---")
    inputs = processor(text=prompt, images=[dummy_image], return_tensors="pt")
    
    print(f"Input keys: {inputs.keys()}")
    if "input_ids" in inputs:
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        # Count image tokens
        # Assuming <image> token is... let's find it. 
        # Usually it's in the config or tokenizer.
        # But we saw the error "tokens: 626".
        print(f"Total tokens: {inputs['input_ids'].shape[1]}")
    
    if "pixel_values" in inputs:
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")

    # Inspect the error logic from LFM2 source if possible (simulated)
    # The error was "features 1891". 
    # If the model produces 1891 features, we need 1891 image tokens.
    
    return "Debug Complete"

@app.local_entrypoint()
def main():
    debug_processor.remote()
