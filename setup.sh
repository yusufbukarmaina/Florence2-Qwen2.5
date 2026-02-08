#!/bin/bash

# Beaker Volume Prediction - Setup Script
# This script helps you set up the environment and verify everything is working

set -e  # Exit on error

echo "================================================"
echo "  Beaker Volume Prediction - Setup Script"
echo "================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python $PYTHON_VERSION is installed"
else
    print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Check if CUDA is available
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_success "CUDA $CUDA_VERSION is available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "CUDA not found. Training will use CPU (much slower)"
fi

# Check if running in virtual environment
echo ""
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_success "Running in virtual environment: $VIRTUAL_ENV"
else
    print_warning "Not running in a virtual environment"
    read -p "Would you like to create a virtual environment? (y/n): " CREATE_VENV
    
    if [[ "$CREATE_VENV" == "y" || "$CREATE_VENV" == "Y" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        print_success "Virtual environment created and activated"
    fi
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed successfully"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Verify key packages
echo ""
echo "Verifying key packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && print_success "PyTorch installed"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" && print_success "Transformers installed"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')" && print_success "PEFT installed"
python3 -c "import gradio; print(f'Gradio: {gradio.__version__}')" && print_success "Gradio installed"

# Check PyTorch CUDA availability
echo ""
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ PyTorch can access CUDA")
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("! PyTorch cannot access CUDA - will use CPU")
EOF

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p output_qwen
mkdir -p output_florence
mkdir -p logs
mkdir -p evaluation
print_success "Directories created"

# Check for HuggingFace token
echo ""
echo "Checking HuggingFace authentication..."
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    print_success "HuggingFace authentication verified"
else
    print_warning "Not logged in to HuggingFace"
    print_info "Run 'huggingface-cli login' to authenticate"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x train.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
print_success "Scripts made executable"

# Configuration file check
echo ""
if [ -f "config.ini" ]; then
    print_info "Configuration file found: config.ini"
    print_warning "Please edit config.ini with your settings before training"
else
    print_error "Configuration file not found!"
fi

# Final summary
echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit config.ini with your HuggingFace credentials and settings"
echo "2. Login to HuggingFace: huggingface-cli login"
echo "3. (Optional) Login to W&B: wandb login"
echo "4. Run training: ./train.sh or python train.py [args]"
echo "5. Evaluate model: python evaluate.py [args]"
echo "6. Launch demo: python gradio_app.py"
echo ""
echo "For help, see README.md"
echo ""

# Test imports
echo "Testing critical imports..."
python3 << 'EOF'
try:
    import torch
    import transformers
    import peft
    import gradio
    import datasets
    from PIL import Image
    import numpy as np
    print("✓ All critical imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF

print_success "Setup verification complete!"
