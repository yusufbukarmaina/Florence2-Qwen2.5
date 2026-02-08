#!/bin/bash

# Beaker Volume Prediction - Training Script for JarvisLab
# This script helps you easily train both models

echo "================================================"
echo "  Beaker Volume Prediction - Training Script"
echo "================================================"
echo ""

# Configuration - EDIT THESE VALUES
DATASET_NAME="YOUR_HF_USERNAME/beaker-dataset"
HF_TOKEN="YOUR_HF_TOKEN"
WANDB_TOKEN="YOUR_WANDB_TOKEN"  # Optional, leave empty if not using
HF_USERNAME="YOUR_HF_USERNAME"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=4
GRADIENT_ACCUM_STEPS=4
LEARNING_RATE=2e-5
WARMUP_STEPS=100

# Ask user which model to train
echo "Which model would you like to train?"
echo "1) Qwen2-VL"
echo "2) Florence2"
echo "3) Both models sequentially"
read -p "Enter choice (1-3): " MODEL_CHOICE

train_qwen2vl() {
    echo ""
    echo "Training Qwen2-VL model..."
    echo "========================================"
    
    python train.py \
        --dataset_name "$DATASET_NAME" \
        --model_type qwen2vl \
        --output_dir ./output_qwen \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE \
        --warmup_steps $WARMUP_STEPS \
        --upload_to_hf \
        --hf_token "$HF_TOKEN" \
        --hf_repo_name "$HF_USERNAME/beaker-qwen2vl" \
        ${WANDB_TOKEN:+--wandb_token "$WANDB_TOKEN"} \
        --run_name "qwen2vl-beaker-$(date +%Y%m%d-%H%M%S)"
    
    echo ""
    echo "Qwen2-VL training completed!"
    echo "Model saved to: ./output_qwen/final_model"
    echo "Results saved to: ./output_qwen/test_results.json"
}

train_florence2() {
    echo ""
    echo "Training Florence2 model..."
    echo "========================================"
    
    python train.py \
        --dataset_name "$DATASET_NAME" \
        --model_type florence2 \
        --output_dir ./output_florence \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE \
        --warmup_steps $WARMUP_STEPS \
        --upload_to_hf \
        --hf_token "$HF_TOKEN" \
        --hf_repo_name "$HF_USERNAME/beaker-florence2" \
        ${WANDB_TOKEN:+--wandb_token "$WANDB_TOKEN"} \
        --run_name "florence2-beaker-$(date +%Y%m%d-%H%M%S)"
    
    echo ""
    echo "Florence2 training completed!"
    echo "Model saved to: ./output_florence/final_model"
    echo "Results saved to: ./output_florence/test_results.json"
}

# Execute based on user choice
case $MODEL_CHOICE in
    1)
        train_qwen2vl
        ;;
    2)
        train_florence2
        ;;
    3)
        train_qwen2vl
        train_florence2
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "  All training completed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Check test_results.json for model metrics"
echo "2. Run 'python gradio_app.py' to test models"
echo "3. View models on HuggingFace Hub"
echo ""
