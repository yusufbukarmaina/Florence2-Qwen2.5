import os
import json
import torch
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

class ModelEvaluator:
    """Evaluate trained models on test dataset"""
    
    def __init__(self, model_path, model_type="qwen2vl"):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_type} model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        if model_type == "qwen2vl":
            base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
        elif model_type == "florence2":
            base_model_name = "microsoft/Florence-2-base"
        
        # Load base model
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def extract_volume_from_text(self, text):
        """Extract numerical volume from generated text"""
        matches = re.findall(r'\d+\.?\d*', text)
        if matches:
            try:
                return float(matches[0])
            except:
                return 0.0
        return 0.0
    
    def predict_single(self, image):
        """Predict volume for a single image"""
        if self.model_type == "qwen2vl":
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
        
        elif self.model_type == "florence2":
            prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            task_prompt = f"{prompt} What is the liquid volume in mL?"
            
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                do_sample=False
            )
        
        # Decode and extract volume
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        volume = self.extract_volume_from_text(generated_text)
        
        return volume, generated_text
    
    def evaluate_dataset(self, dataset, save_dir="./evaluation"):
        """Evaluate model on entire dataset"""
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = []
        ground_truths = []
        raw_outputs = []
        
        print(f"Evaluating {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(dataset)}")
            
            image = sample['image']
            true_volume = float(sample['volume'])
            
            pred_volume, raw_output = self.predict_single(image)
            
            predictions.append(pred_volume)
            ground_truths.append(true_volume)
            raw_outputs.append(raw_output)
        
        # Calculate metrics
        mae = mean_absolute_error(ground_truths, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
        r2 = r2_score(ground_truths, predictions)
        
        # Calculate percentage errors
        percentage_errors = [
            abs(pred - true) / true * 100 if true > 0 else 0
            for pred, true in zip(predictions, ground_truths)
        ]
        mean_percentage_error = np.mean(percentage_errors)
        
        results = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mean_percentage_error": float(mean_percentage_error),
            "num_samples": len(dataset),
            "predictions": predictions,
            "ground_truths": ground_truths,
            "raw_outputs": raw_outputs[:10]  # Save first 10 for inspection
        }
        
        # Save results
        with open(f"{save_dir}/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self.create_visualizations(predictions, ground_truths, save_dir)
        
        return results
    
    def create_visualizations(self, predictions, ground_truths, save_dir):
        """Create evaluation plots"""
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Scatter plot: Predicted vs Actual
        plt.figure(figsize=(10, 8))
        plt.scatter(ground_truths, predictions, alpha=0.5, s=50)
        
        # Perfect prediction line
        min_val = min(min(ground_truths), min(predictions))
        max_val = max(max(ground_truths), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Volume (mL)', fontsize=12)
        plt.ylabel('Predicted Volume (mL)', fontsize=12)
        plt.title('Predicted vs Actual Volume', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error distribution
        errors = np.array(predictions) - np.array(ground_truths)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error (mL)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Error Distribution', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(errors, vert=True)
        plt.ylabel('Prediction Error (mL)', fontsize=12)
        plt.title('Error Box Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ground_truths, errors, alpha=0.5, s=50)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Actual Volume (mL)', fontsize=12)
        plt.ylabel('Residual (mL)', fontsize=12)
        plt.title('Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/residual_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate beaker volume prediction model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--model_type", type=str, required=True, choices=["qwen2vl", "florence2"],
                       help="Model type")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate (train/validation/test)")
    parser.add_argument("--save_dir", type=str, default="./evaluation",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name)
    
    if args.split not in dataset:
        print(f"Split '{args.split}' not found. Available splits: {list(dataset.keys())}")
        return
    
    eval_dataset = dataset[args.split]
    print(f"Loaded {len(eval_dataset)} samples from {args.split} split")
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.model_type)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(eval_dataset, args.save_dir)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset_name} ({args.split})")
    print(f"Number of samples: {results['num_samples']}")
    print("-"*60)
    print(f"MAE (Mean Absolute Error):     {results['mae']:.2f} mL")
    print(f"RMSE (Root Mean Squared Error): {results['rmse']:.2f} mL")
    print(f"R² (R-squared):                 {results['r2']:.4f}")
    print(f"Mean Percentage Error:          {results['mean_percentage_error']:.2f}%")
    print("="*60)
    print(f"\nDetailed results saved to: {args.save_dir}/evaluation_results.json")
    print(f"Visualizations saved to: {args.save_dir}/")
    print("\nGenerated plots:")
    print("  - scatter_plot.png: Predicted vs Actual volumes")
    print("  - error_distribution.png: Error histogram and box plot")
    print("  - residual_plot.png: Residuals vs Actual volumes")
    

if __name__ == "__main__":
    main()
