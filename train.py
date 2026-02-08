import os
import json
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from huggingface_hub import HfApi
import wandb

class BeakerVolumeDataset(torch.utils.data.Dataset):
    """Custom dataset for beaker volume prediction"""
    
    def __init__(self, hf_dataset, processor, model_type="qwen2vl"):
        self.dataset = hf_dataset
        self.processor = processor
        self.model_type = model_type
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        volume = float(item['volume'])  # Ground truth volume in mL
        
        if self.model_type == "qwen2vl":
            # Qwen2-VL specific prompt
            prompt = "What is the liquid volume in this beaker in mL? Provide only the numerical value."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # Add labels (volume as text for training)
            volume_text = f"{volume}"
            labels = self.processor.tokenizer(
                volume_text,
                return_tensors="pt",
                padding=True
            )["input_ids"]
            
            inputs["labels"] = labels
            
        elif self.model_type == "florence2":
            # Florence2 specific prompt
            prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            task_prompt = f"{prompt} What is the liquid volume in mL?"
            
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Add labels for Florence2
            volume_text = f"The liquid volume is {volume} mL"
            labels = self.processor.tokenizer(
                volume_text,
                return_tensors="pt",
                padding=True
            )["input_ids"]
            
            inputs["labels"] = labels
        
        # Squeeze batch dimension and add ground truth
        inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
        inputs["ground_truth_volume"] = volume
        
        return inputs


def collate_fn(batch):
    """Custom collate function to handle batching"""
    # Extract ground truth volumes
    ground_truths = [item.pop("ground_truth_volume") for item in batch]
    
    # Get all keys from first item
    keys = batch[0].keys()
    
    # Pad and stack tensors
    padded_batch = {}
    for key in keys:
        if key == "labels":
            # Pad labels with -100 (ignore index)
            max_len = max(item[key].shape[-1] for item in batch)
            padded = []
            for item in batch:
                pad_len = max_len - item[key].shape[-1]
                padded_item = torch.nn.functional.pad(
                    item[key], 
                    (0, pad_len), 
                    value=-100
                )
                padded.append(padded_item)
            padded_batch[key] = torch.stack(padded)
        else:
            # Try to stack, if shapes match
            try:
                padded_batch[key] = torch.stack([item[key] for item in batch])
            except:
                # If shapes don't match, pad to max length
                if isinstance(batch[0][key], torch.Tensor) and batch[0][key].dim() > 0:
                    max_len = max(item[key].shape[-1] for item in batch)
                    padded = []
                    for item in batch:
                        if item[key].dim() == 1:
                            pad_len = max_len - item[key].shape[0]
                            padded_item = torch.nn.functional.pad(item[key], (0, pad_len), value=0)
                        else:
                            pad_len = max_len - item[key].shape[-1]
                            padded_item = torch.nn.functional.pad(item[key], (0, pad_len), value=0)
                        padded.append(padded_item)
                    padded_batch[key] = torch.stack(padded)
                else:
                    padded_batch[key] = torch.stack([item[key] for item in batch])
    
    padded_batch["ground_truth_volumes"] = torch.tensor(ground_truths, dtype=torch.float32)
    
    return padded_batch


def extract_volume_from_text(text):
    """Extract numerical volume from generated text"""
    import re
    
    # Try to find a number (integer or float)
    matches = re.findall(r'\d+\.?\d*', text)
    
    if matches:
        try:
            return float(matches[0])
        except:
            return 0.0
    return 0.0


def compute_metrics(eval_preds, processor, model):
    """Compute MAE, RMSE, and R-squared metrics"""
    predictions, label_ids = eval_preds
    
    # Decode predictions
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    
    # Extract volumes from text
    predicted_volumes = [extract_volume_from_text(pred) for pred in decoded_preds]
    
    # Get ground truth volumes (stored during evaluation)
    # This will be passed separately
    true_volumes = eval_preds.ground_truth_volumes if hasattr(eval_preds, 'ground_truth_volumes') else []
    
    if len(true_volumes) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    
    # Calculate metrics
    mae = mean_absolute_error(true_volumes, predicted_volumes)
    rmse = np.sqrt(mean_squared_error(true_volumes, predicted_volumes))
    r2 = r2_score(true_volumes, predicted_volumes)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


class VolumeTrainer(Trainer):
    """Custom trainer to handle ground truth volumes during evaluation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth_volumes = []
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Store ground truth volumes
        if "ground_truth_volumes" in inputs:
            self.ground_truth_volumes.extend(inputs["ground_truth_volumes"].cpu().numpy())
            inputs = {k: v for k, v in inputs.items() if k != "ground_truth_volumes"}
        
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def evaluate(self, *args, **kwargs):
        # Reset ground truth volumes before evaluation
        self.ground_truth_volumes = []
        result = super().evaluate(*args, **kwargs)
        
        # Add ground truth to predictions for metric computation
        if hasattr(self, 'ground_truth_volumes') and len(self.ground_truth_volumes) > 0:
            result['ground_truth_volumes'] = self.ground_truth_volumes
        
        return result


def setup_lora_config(model_type):
    """Setup LoRA configuration for the model"""
    if model_type == "qwen2vl":
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    elif model_type == "florence2":
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
    
    return lora_config


def train_model(args):
    """Main training function"""
    
    # Initialize wandb if token provided
    if args.wandb_token:
        wandb.login(key=args.wandb_token)
        wandb.init(project=f"beaker-volume-{args.model_type}", name=args.run_name)
    
    # Load dataset from HuggingFace
    print(f"Loading dataset from {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name)
    
    # Split dataset: 70% train, 15% validation, 15% test
    if 'train' not in dataset:
        # If no split, create splits
        train_test = dataset['train'].train_test_split(test_size=0.3, seed=42)
        test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
        
        dataset = DatasetDict({
            'train': train_test['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })
    
    print(f"Dataset splits - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    
    # Load model and processor
    if args.model_type == "qwen2vl":
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"Loading {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    elif args.model_type == "florence2":
        model_name = "microsoft/Florence-2-base"
        print(f"Loading {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config(args.model_type)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    train_dataset = BeakerVolumeDataset(dataset['train'], processor, args.model_type)
    val_dataset = BeakerVolumeDataset(dataset['validation'], processor, args.model_type)
    test_dataset = BeakerVolumeDataset(dataset['test'], processor, args.model_type)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="wandb" if args.wandb_token else "none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    
    # Initialize trainer
    trainer = VolumeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    # Save final model
    print(f"Saving model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")
    processor.save_pretrained(f"{args.output_dir}/final_model")
    
    # Save test results
    with open(f"{args.output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Upload to HuggingFace if requested
    if args.upload_to_hf and args.hf_token:
        print(f"Uploading model to HuggingFace as {args.hf_repo_name}...")
        model.push_to_hub(
            args.hf_repo_name,
            token=args.hf_token,
            commit_message=f"Fine-tuned {args.model_type} for beaker volume prediction"
        )
        processor.push_to_hub(
            args.hf_repo_name,
            token=args.hf_token
        )
        print(f"Model uploaded successfully to https://huggingface.co/{args.hf_repo_name}")
    
    if args.wandb_token:
        wandb.finish()
    
    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train beaker volume prediction model")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="HuggingFace dataset name (e.g., username/beaker-dataset)")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, required=True, choices=["qwen2vl", "florence2"],
                       help="Model type to train")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    
    # HuggingFace upload arguments
    parser.add_argument("--upload_to_hf", action="store_true",
                       help="Upload trained model to HuggingFace")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token for uploading")
    parser.add_argument("--hf_repo_name", type=str, default=None,
                       help="HuggingFace repository name for upload")
    
    # Logging arguments
    parser.add_argument("--wandb_token", type=str, default=None,
                       help="Weights & Biases token for logging")
    parser.add_argument("--run_name", type=str, default="beaker-volume-training",
                       help="Run name for experiment tracking")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.upload_to_hf and (not args.hf_token or not args.hf_repo_name):
        raise ValueError("--hf_token and --hf_repo_name are required when --upload_to_hf is set")
    
    # Train model
    results = train_model(args)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Final test metrics:")
    print(f"  MAE: {results.get('mae', 'N/A')}")
    print(f"  RMSE: {results.get('rmse', 'N/A')}")
    print(f"  R²: {results.get('r2', 'N/A')}")
    print("="*50)
