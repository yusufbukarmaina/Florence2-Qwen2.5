# 🚀 Quick Start Guide for JarvisLab

This guide will help you get started with training beaker volume prediction models on JarvisLab.

## 📋 Prerequisites

- JarvisLab account with GPU instance
- HuggingFace account and token
- Dataset uploaded to HuggingFace Hub

## 🔧 Step-by-Step Setup

### 1. Launch JarvisLab Instance

1. Log in to [JarvisLab](https://jarvislabs.ai/)
2. Create a new instance with:
   - **GPU**: A100 (40GB) or RTX 4090 (recommended)
   - **Storage**: At least 50GB
   - **Framework**: PyTorch (latest)
   - **Duration**: Based on your training needs (4-8 hours recommended)

### 2. Connect to Instance

```bash
# SSH into your instance (use connection details from JarvisLab)
ssh user@your-instance.jarvislabs.ai
```

### 3. Clone Repository

```bash
# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/YOUR_USERNAME/beaker-volume-prediction.git
cd beaker-volume-prediction
```

### 4. Run Setup Script

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This will:
- ✓ Check Python version
- ✓ Verify CUDA installation
- ✓ Install all dependencies
- ✓ Create necessary directories
- ✓ Verify package installations

### 5. Configure Training

Edit `config.ini` with your settings:

```bash
nano config.ini
```

Update the following:
- `dataset_name`: Your HuggingFace dataset path
- `username`: Your HuggingFace username
- `token`: Your HuggingFace token
- `wandb.token`: (Optional) Your W&B token

### 6. Authenticate HuggingFace

```bash
huggingface-cli login
# Enter your token when prompted
```

### 7. (Optional) Setup Weights & Biases

```bash
wandb login
# Enter your token when prompted
```

## 🏋️ Training Models

### Option A: Interactive Training Script

```bash
# Make training script executable
chmod +x train.sh

# Run interactive training
./train.sh
```

Follow the prompts to select which model(s) to train.

### Option B: Direct Python Commands

**Train Qwen2-VL:**
```bash
python train.py \
    --dataset_name YOUR_USERNAME/beaker-dataset \
    --model_type qwen2vl \
    --output_dir /workspace/output_qwen \
    --epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --upload_to_hf \
    --hf_token YOUR_HF_TOKEN \
    --hf_repo_name YOUR_USERNAME/beaker-qwen2vl
```

**Train Florence2:**
```bash
python train.py \
    --dataset_name YOUR_USERNAME/beaker-dataset \
    --model_type florence2 \
    --output_dir /workspace/output_florence \
    --epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --upload_to_hf \
    --hf_token YOUR_HF_TOKEN \
    --hf_repo_name YOUR_USERNAME/beaker-florence2
```

## 📊 Monitoring Training

### View Training Progress

Training logs will show:
- Loss curves
- Evaluation metrics (MAE, RMSE, R²)
- Learning rate schedule
- GPU utilization

### Monitor with W&B (if enabled)

1. Visit [wandb.ai](https://wandb.ai)
2. Navigate to your project: `beaker-volume-prediction`
3. View real-time metrics, system stats, and more

### Using TensorBoard (alternative)

```bash
# In a new terminal
tensorboard --logdir=/workspace/output_qwen/logs
```

## 💾 Saving Your Work

### Download Trained Models

```bash
# Compress model directory
tar -czf qwen_model.tar.gz output_qwen/final_model
tar -czf florence_model.tar.gz output_florence/final_model

# Download using JarvisLab interface or scp
```

### Automatic HuggingFace Upload

If you used `--upload_to_hf` flag, models are automatically uploaded to HuggingFace Hub after training.

## 🧪 Testing Your Models

### Run Evaluation

```bash
# Evaluate Qwen2-VL
python evaluate.py \
    --model_path /workspace/output_qwen/final_model \
    --model_type qwen2vl \
    --dataset_name YOUR_USERNAME/beaker-dataset \
    --save_dir /workspace/evaluation_qwen

# Evaluate Florence2
python evaluate.py \
    --model_path /workspace/output_florence/final_model \
    --model_type florence2 \
    --dataset_name YOUR_USERNAME/beaker-dataset \
    --save_dir /workspace/evaluation_florence
```

### Launch Gradio Demo

```bash
# Start Gradio app
python gradio_app.py
```

Access the demo at: `http://your-instance.jarvislabs.ai:7860`

## 🔍 Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```bash
python train.py ... --batch_size 2 --gradient_accumulation_steps 8
```

**Solution 2: Use gradient checkpointing**
Add to training arguments in `train.py`:
```python
gradient_checkpointing=True
```

### Slow Training

**Check GPU utilization:**
```bash
nvidia-smi -l 1
```

**Expected utilization:** 80-95%

If lower:
- Increase batch size
- Reduce num_workers if CPU bottleneck
- Check data loading speed

### Connection Lost During Training

**Use tmux or screen:**
```bash
# Start tmux session
tmux new -s training

# Run training inside tmux
python train.py ...

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Model Not Uploading to HuggingFace

**Verify authentication:**
```bash
huggingface-cli whoami
```

**Check token permissions:**
- Token needs "write" access
- Verify repository exists or can be created

## 📈 Expected Training Times

On A100 (40GB):
- **Qwen2-VL**: ~2-3 hours for 10 epochs (2000 images)
- **Florence2**: ~1.5-2 hours for 10 epochs (2000 images)

On RTX 4090:
- **Qwen2-VL**: ~3-4 hours for 10 epochs
- **Florence2**: ~2-3 hours for 10 epochs

## 💰 Cost Optimization Tips

1. **Use early stopping**: Automatically stop when model stops improving
2. **Start with fewer epochs**: Try 5-7 epochs first, increase if needed
3. **Use mixed precision**: Already enabled by default (bf16)
4. **Monitor closely**: Stop training if metrics plateau
5. **Use spot instances**: Cheaper but can be interrupted

## 📝 Best Practices

1. **Always use version control**: Commit changes regularly
2. **Save checkpoints frequently**: Don't lose progress
3. **Monitor metrics closely**: Watch for overfitting
4. **Test on validation set**: Before committing to long training
5. **Document hyperparameters**: Keep track of what works

## 🔗 Useful Commands

```bash
# Check GPU memory usage
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check disk space
df -h

# View running processes
htop

# Kill a process
pkill -f train.py

# Check Python packages
pip list | grep torch

# View logs
tail -f output_qwen/logs/events.out.tfevents.*
```

## 📚 Additional Resources

- [JarvisLab Documentation](https://jarvislabs.ai/docs)
- [HuggingFace Hub](https://huggingface.co/docs/hub)
- [Weights & Biases](https://docs.wandb.ai)
- [PyTorch Documentation](https://pytorch.org/docs)

## 🆘 Getting Help

If you encounter issues:

1. Check the main README.md
2. Review error messages carefully
3. Search HuggingFace forums
4. Open an issue on GitHub
5. Contact JarvisLab support for infrastructure issues

## ✅ Checklist Before Training

- [ ] JarvisLab instance launched with GPU
- [ ] Repository cloned
- [ ] Dependencies installed (`./setup.sh`)
- [ ] config.ini updated with credentials
- [ ] HuggingFace authenticated
- [ ] Dataset accessible
- [ ] Sufficient disk space (50GB+)
- [ ] W&B setup (optional)
- [ ] Tmux/screen session started

Happy training! 🚀
