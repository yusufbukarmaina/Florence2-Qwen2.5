"""
Test script to verify setup and dataset before training
Run this before starting actual training to catch issues early
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_environment():
    """Check Python environment and packages"""
    print_section("Environment Check")
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"✓ Python version: {python_version}")
    
    # PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check other packages
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    try:
        import peft
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError:
        print("✗ PEFT not installed")
        return False
    
    try:
        import gradio
        print(f"✓ Gradio version: {gradio.__version__}")
    except ImportError:
        print("✗ Gradio not installed")
        return False
    
    try:
        from datasets import load_dataset
        print(f"✓ Datasets library available")
    except ImportError:
        print("✗ Datasets library not installed")
        return False
    
    return True

def check_dataset(dataset_name):
    """Check if dataset can be loaded and has correct structure"""
    print_section("Dataset Check")
    
    try:
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name)
        print(f"✓ Dataset loaded successfully")
        
        # Check splits
        print(f"✓ Available splits: {list(dataset.keys())}")
        
        # Check first split
        first_split = list(dataset.keys())[0]
        print(f"\nExamining '{first_split}' split:")
        print(f"  - Number of samples: {len(dataset[first_split])}")
        
        # Check sample structure
        sample = dataset[first_split][0]
        print(f"  - Sample keys: {list(sample.keys())}")
        
        # Verify required fields
        required_fields = ['image', 'volume']
        missing_fields = [field for field in required_fields if field not in sample.keys()]
        
        if missing_fields:
            print(f"✗ Missing required fields: {missing_fields}")
            return False
        
        print(f"✓ All required fields present: {required_fields}")
        
        # Check image
        image = sample['image']
        if isinstance(image, Image.Image):
            print(f"✓ Image format: PIL.Image ({image.size})")
        else:
            print(f"! Image format: {type(image)} (should be PIL.Image)")
        
        # Check volume
        volume = sample['volume']
        print(f"✓ Volume: {volume} (type: {type(volume).__name__})")
        
        # Check volume range
        all_volumes = [float(s['volume']) for s in dataset[first_split]]
        print(f"  - Volume range: {min(all_volumes):.1f} - {max(all_volumes):.1f} mL")
        print(f"  - Mean volume: {np.mean(all_volumes):.1f} mL")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

def test_model_loading(model_type):
    """Test if model can be loaded"""
    print_section(f"Model Loading Test: {model_type}")
    
    try:
        if model_type == "qwen2vl":
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
        elif model_type == "florence2":
            model_name = "microsoft/Florence-2-base"
        else:
            print(f"✗ Unknown model type: {model_type}")
            return False
        
        print(f"Loading processor from {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Processor loaded successfully")
        
        print(f"Loading model from {model_name}...")
        print("  (This may take a few minutes on first run...)")
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        print(f"✓ Model loaded successfully")
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {param_count / 1e6:.1f}M")
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Trainable parameters: {trainable_params / 1e6:.1f}M")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_inference(model_type, dataset_name):
    """Test a quick inference on a sample"""
    print_section(f"Inference Test: {model_type}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        first_split = list(dataset.keys())[0]
        sample = dataset[first_split][0]
        
        # Load model
        if model_type == "qwen2vl":
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
        else:
            model_name = "microsoft/Florence-2-base"
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Prepare input
        image = sample['image']
        
        if model_type == "qwen2vl":
            prompt = "What is the liquid volume in this beaker in mL?"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        else:
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING> What is the liquid volume in mL?"
            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        
        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        print("Generating prediction...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        # Decode
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"✓ Inference completed successfully")
        print(f"  - Ground truth: {sample['volume']} mL")
        print(f"  - Model output: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_disk_space():
    """Check available disk space"""
    print_section("Disk Space Check")
    
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    
    print(f"Total: {total / (2**30):.1f} GB")
    print(f"Used: {used / (2**30):.1f} GB")
    print(f"Free: {free / (2**30):.1f} GB")
    
    if free / (2**30) < 50:
        print("⚠ Warning: Less than 50GB free space available")
        print("  Recommended: At least 50GB for model training")
        return False
    else:
        print("✓ Sufficient disk space available")
        return True

def check_huggingface_auth():
    """Check HuggingFace authentication"""
    print_section("HuggingFace Authentication")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        print(f"  - Type: {user_info.get('type', 'user')}")
        return True
    except Exception as e:
        print(f"✗ Not logged in to HuggingFace")
        print(f"  Run: huggingface-cli login")
        return False

def main():
    """Run all checks"""
    print("\n" + "🔍 " + "="*58)
    print("  Pre-Training Setup Verification")
    print("="*60 + "\n")
    
    results = {}
    
    # Environment check
    results['environment'] = check_environment()
    
    # HuggingFace auth check
    results['hf_auth'] = check_huggingface_auth()
    
    # Disk space check
    results['disk_space'] = check_disk_space()
    
    # Get dataset name from user
    print_section("Dataset Configuration")
    dataset_name = input("Enter your HuggingFace dataset name (e.g., username/beaker-dataset): ").strip()
    
    if dataset_name:
        results['dataset'] = check_dataset(dataset_name)
        
        # Model loading tests
        print("\nWhich model would you like to test?")
        print("1) Qwen2-VL")
        print("2) Florence2")
        print("3) Both")
        print("4) Skip model tests")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            results['qwen_model'] = test_model_loading("qwen2vl")
            results['qwen_inference'] = test_inference("qwen2vl", dataset_name)
        elif choice == "2":
            results['florence_model'] = test_model_loading("florence2")
            results['florence_inference'] = test_inference("florence2", dataset_name)
        elif choice == "3":
            results['qwen_model'] = test_model_loading("qwen2vl")
            results['florence_model'] = test_model_loading("florence2")
            results['qwen_inference'] = test_inference("qwen2vl", dataset_name)
            results['florence_inference'] = test_inference("florence2", dataset_name)
    
    # Summary
    print_section("Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("1. Edit config.ini with your settings")
        print("2. Run: ./train.sh or python train.py [args]")
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Login to HuggingFace: huggingface-cli login")
        print("- Verify dataset structure and permissions")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
