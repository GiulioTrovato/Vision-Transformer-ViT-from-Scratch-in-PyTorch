# Vision Transformer (ViT) from Scratch in PyTorch

A lightweight, fully custom implementation of the Vision Transformer (ViT) architecture from the ground up using PyTorch. 
This project focuses on educational clarity and architectural understanding, implementing the core components of the 2020 paper *"An Image is Worth 16x16 Words"* without relying on pre-built transformer libraries.

## 🧠 Architectural Highlights
- **Custom Patch Embedding:** Tensor manipulation using `einops` for dynamic image patching.
- **Native Multi-Head Self-Attention:** Pure linear algebra implementation of scaled dot-product attention.
- **Pre-Norm Architecture:** Utilizing Layer Normalization *before* the attention and MLP blocks, combined with Residual Connections, for superior gradient flow.
- **Learnable Positional & CLS Tokens:** Implemented as custom `nn.Parameter` tensors.

## 🚀 Performance
Trained on the **CIFAR-10** dataset (32x32 images) for just 20 epochs, the model successfully achieves **~75.7% accuracy** on the test set, demonstrating highly effective learning capabilities for a lightweight transformer trained from scratch.

## 📂 Project Structure
\`\`\`text
├── src/
│   ├── modules.py   # Core building blocks (Attention, MLP, Patching)
│   ├── vit.py       # The main Vision Transformer assembly
│   └── utils.py     # Data augmentation and CIFAR-10 dataloaders
└── train.py         # The training and validation loop
\`\`\`

## 🛠️ Quick Start

**1. Clone the repository and install dependencies:**
\`\`\`bash
git clone https://github.com/GiulioTrovato/vision-transformer-from-scratch.git
cd vision-transformer-from-scratch
pip install torch torchvision einops
\`\`\`

**2. Run the training loop:**
\`\`\`bash
python train.py
\`\`\`
*Note: The script will automatically download the CIFAR-10 dataset into a local `./data` folder and utilize CUDA if available.*

## ⚙️ Hyperparameters Used
- **Patch Size:** 4x4
- **Embedding Dimension:** 256
- **Transformer Blocks (Depth):** 6
- **Attention Heads:** 8
- **Optimizer:** AdamW (LR: 3e-4, Weight Decay: 1e-4)
- **Batch Size:** 128
