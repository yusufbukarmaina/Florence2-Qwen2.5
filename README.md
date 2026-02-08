# 🔬 Beaker Liquid Volume Prediction

A deep learning project for predicting liquid volume in beakers from images using vision-language models fine-tuned with LoRA.

## 📋 Overview

This project trains two state-of-the-art vision models to predict liquid volume in laboratory beakers from images:
- **Qwen2-VL-2B-Instruct**: Advanced vision-language model
- **Florence2-Base**: Microsoft's vision foundation model

Both models are fine-tuned using **LoRA (Low-Rank Adaptation)** for efficient training on a dataset of 2000 beaker images with varying backgrounds.

## 🎯 Key Features

- **Dual Model Support**: Train and compare Qwen2-VL and Florence2 models
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Comprehensive Evaluation**: MAE, RMSE, and R² metrics
- **Interactive Demo**: Gradio web interface for easy inference
- **HuggingFace Integration**: Direct upload to HuggingFace Hub
- **Background Robustness**: Trained on both normal and cluttered backgrounds
- **JarvisLab Ready**: Optimized for cloud GPU training

## 📊 Dataset

- **Total Images**: 2000 beaker images
- **Training Split**: 70% (1400 images)
- **Validation Split**: 15% (300 images)
- **Test Split**: 15% (300 images)
- **Backgrounds**: Normal (clean) and Cluttered (complex)
- **Format**: Images with volume labels in mL

### Dataset Structure

Your HuggingFace dataset should have the following structure:

```
{
    "image": PIL.Image,
    "volume": float,  # Volume in mL
    "background": str  # "normal" or "cluttered"
}
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/beaker-volume-prediction.git
cd beaker-volume-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Setup for JarvisLab

If using JarvisLab, the environment is pre-configured with CUDA support. Just install the requirements.

## 🏋️ Training

### Train Qwen2-VL Model

```bash
python train.py \
    --dataset_name YOUR_HF_USERNAME/beaker-dataset \
    --model_type qwen2vl \
    --output_dir ./output_qwen \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --upload_to_hf \
    --hf_token YOUR_HF_TOKEN \
    --hf_repo_name YOUR_HF_USERNAME/beaker-qwen2vl \
    --wandb_token YOUR_WANDB_TOKEN \
    --run_name qwen2vl-beaker-training
```

### Train Florence2 Model

```bash
python train.py \
    --dataset_name YOUR_HF_USERNAME/beaker-dataset \
    --model_type florence2 \
    --output_dir ./output_florence \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --upload_to_hf \
    --hf_token YOUR_HF_TOKEN \
    --hf_repo_name YOUR_HF_USERNAME/beaker-florence2 \
    --wandb_token YOUR_WANDB_TOKEN \
    --run_name florence2-beaker-training
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_name` | HuggingFace dataset name | Required |
| `--model_type` | Model type (qwen2vl or florence2) | Required |
| `--output_dir` | Output directory for checkpoints | ./output |
| `--epochs` | Number of training epochs | 10 |
| `--batch_size` | Batch size per GPU | 4 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 4 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--warmup_steps` | Warmup steps | 100 |
| `--upload_to_hf` | Upload to HuggingFace Hub | False |
| `--hf_token` | HuggingFace API token | None |
| `--hf_repo_name` | HuggingFace repo name | None |
| `--wandb_token` | Weights & Biases token | None |
| `--run_name` | Experiment run name | beaker-volume-training |

## 🎨 Gradio Demo

### Local Inference

```bash
python gradio_app.py
```

The app will be available at `http://localhost:7860`

### HuggingFace Spaces Deployment

1. Create a new Space on HuggingFace
2. Upload `app.py` and `requirements.txt`
3. Set environment variables:
   - `QWEN_MODEL_PATH`: Your Qwen2-VL model path
   - `FLORENCE_MODEL_PATH`: Your Florence2 model path
4. The Space will automatically build and deploy

## 📈 Evaluation Metrics

The models are evaluated using three key metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual volumes
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **R² (R-squared)**: Proportion of variance explained by the model

Results are saved in `{output_dir}/test_results.json` after training.

## 🏗️ Model Architecture

### LoRA Configuration

Both models use LoRA for efficient fine-tuning:

**Qwen2-VL:**
- Rank (r): 16
- Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Dropout: 0.05

**Florence2:**
- Rank (r): 16
- Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Dropout: 0.05

## 📁 Project Structure

```
beaker-volume-prediction/
├── train.py                 # Main training script
├── gradio_app.py           # Local Gradio demo
├── app.py                  # HuggingFace Spaces app
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── output_qwen/           # Qwen2-VL training output
│   ├── final_model/       # Final trained model
│   ├── logs/              # Training logs
│   └── test_results.json  # Test metrics
└── output_florence/       # Florence2 training output
    ├── final_model/       # Final trained model
    ├── logs/              # Training logs
    └── test_results.json  # Test metrics
```

## 💻 JarvisLab Training

### Setup

1. Create a JarvisLab instance with GPU (recommended: A100 or RTX 4090)
2. Clone this repository
3. Install dependencies
4. Run training script

### Example JarvisLab Command

```bash
# SSH into JarvisLab instance
ssh user@jarvislab-instance

# Navigate to project
cd /workspace/beaker-volume-prediction

# Install dependencies
pip install -r requirements.txt

# Start training
python train.py \
    --dataset_name YOUR_HF_USERNAME/beaker-dataset \
    --model_type qwen2vl \
    --output_dir /workspace/output_qwen \
    --epochs 15 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --upload_to_hf \
    --hf_token YOUR_HF_TOKEN \
    --hf_repo_name YOUR_HF_USERNAME/beaker-qwen2vl \
    --wandb_token YOUR_WANDB_TOKEN
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `--batch_size` (try 2 or 1)
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Use smaller models or reduce LoRA rank

### Slow Training

- Increase `--batch_size` if GPU memory allows
- Reduce `--gradient_accumulation_steps`
- Use mixed precision training (enabled by default with bf16)

### Model Not Loading

- Ensure you have enough disk space
- Check HuggingFace authentication with `huggingface-cli login`
- Verify model paths are correct

## 📊 Expected Performance

Based on the training setup:

| Model | MAE (mL) | RMSE (mL) | R² |
|-------|----------|-----------|-----|
| Qwen2-VL | ~5-10 | ~8-15 | >0.90 |
| Florence2 | ~8-12 | ~12-18 | >0.85 |

*Note: Actual performance depends on dataset quality and training duration*

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen2-VL model
- **Microsoft** for the Florence2 model
- **HuggingFace** for the Transformers library and model hosting
- **JarvisLab** for GPU infrastructure

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: your.email@example.com
- HuggingFace: [@YOUR_USERNAME](https://huggingface.co/YOUR_USERNAME)

## 🔗 Links

- [HuggingFace Dataset](https://huggingface.co/datasets/YOUR_USERNAME/beaker-dataset)
- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Florence2 Model](https://huggingface.co/microsoft/Florence-2-base)
- [Gradio Documentation](https://gradio.app/docs)
- [PEFT Library](https://github.com/huggingface/peft)

---

**Built with ❤️ using PyTorch, Transformers, and Gradio**
