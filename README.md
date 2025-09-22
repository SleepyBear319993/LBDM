# LBDM
Lattice Boltzmann Diffusion Model

## Results

The following image demonstrates the LBDM diffusion and reversal process results:

![Diffusion and Reversal Process](assets/diffusion_reversal_1000_girl_0.01.png)

*Diffusion and Reversal Process - MSE: 0.00000001, PSNR: 79.82 dB*

## Setup

### 1. Install Anaconda (if not already installed)

If you don't have Anaconda installed, download and install it from the official website: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

Follow the installer instructions for your operating system.

This project can be run in your base Anaconda environment. Ensure your base environment has Python 3.9 or a compatible version. Core libraries like NumPy, Numba, and Matplotlib are typically included with Anaconda.

### 2. Install PyTorch and Torchvision

With your Anaconda environment activated (which is usually the base environment by default when you open Anaconda Prompt), install PyTorch and Torchvision. It's recommended to install them using the official PyTorch website's instructions to ensure compatibility with your CUDA version if you have an NVIDIA GPU.

Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your preferences (e.g., OS, Package, Compute Platform). Then, run the provided command in your Anaconda Prompt.

Alternatively, you can try installing via pip using the `requirements.txt` file (ensure pip is using the Anaconda environment's Python):

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install torch torchvision
```
