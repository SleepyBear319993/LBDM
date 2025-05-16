# LBDM
Lattice Boltzmann Diffusion Model

## Setup

### 1. Install Anaconda (if not already installed)

If you don't have Anaconda installed, download and install it from the official website: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

Follow the installer instructions for your operating system.

### 2. Prepare Your Anaconda Environment

This project can be run in your base Anaconda environment. Ensure your base environment has Python 3.9 or a compatible version. Core libraries like NumPy, Numba, and Matplotlib are typically included with Anaconda.

Open an Anaconda Prompt (or terminal on Linux/macOS). You will run the subsequent installation commands in this environment.

### 3. Install PyTorch and Torchvision

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

If you have a CUDA-enabled GPU and want to use it, ensure you install the correct PyTorch build. For example, for a specific CUDA version (e.g., CUDA 11.8), the command might look like this (always check the PyTorch website for the latest command):

```bash
# Example for CUDA 11.8 - verify on PyTorch website
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

If you are using a GPU based on the Blackwell architecture, please install PyTorch with CUDA 12.8 or above, the command might look like this:

```bash
# Example for CUDA 12.8
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128