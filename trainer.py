import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import glob

# Assuming UNetLBM is in unet_lbm.py
from unet_lbm import UNetLBM

# --- Configuration ---
DATA_DIR = "lbm_diffusion_data_cifar0_32" # <<< CHANGE HERE
IMG_SIZE = 32 # <<< CHANGE HERE
N_CHANNELS = 27 # 3 RGB * 9 LBM distributions
N_OUT_CHANNELS = 27
TOTAL_STEPS = 1000 # T used during data generation
BATCH_SIZE = 64 # Adjust based on GPU memory (might be able to increase for smaller images)
LEARNING_RATE = 1e-4
EPOCHS = 25 # Adjust as needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "unet_lbm_model_32.pth" # <<< CHANGE HERE
# --- ---

class LBMDiffusionDataset(Dataset):
    def __init__(self, data_dir, total_steps):
        self.data_dir = data_dir
        self.total_steps = total_steps
        self.file_paths = glob.glob(os.path.join(data_dir, "*.h5"))
        self.num_images = len(self.file_paths)
        self._check_data()

    def _check_data(self):
        if not self.file_paths:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")
        # Optional: Open one file to check dimensions
        with h5py.File(self.file_paths[0], 'r') as hf:
            assert hf['f_t'].shape[0] == self.total_steps, "Mismatch in total_steps"
            assert hf['f_t'].shape[1] * hf['f_t'].shape[2] == N_CHANNELS, "Mismatch in channels/distributions"
            # Add more checks if needed
        print(f"Found {self.num_images} HDF5 files.")

    def __len__(self):
        # Total number of samples is num_images * total_steps
        return self.num_images * self.total_steps

    def __getitem__(self, idx):
        # Calculate which image and which timestep this index corresponds to
        image_idx = idx // self.total_steps
        time_step_idx = idx % self.total_steps # Index within HDF5 file (0 to total_steps-1)

        file_path = self.file_paths[image_idx]
        with h5py.File(file_path, 'r') as hf:
            # f_t shape in HDF5: (total_steps, num_channels=3, num_distributions=9, ny, nx)
            f_t_data = hf['f_t'][time_step_idx] # Shape: (3, 9, ny, nx)
            delta_f_data = hf['delta_f'][time_step_idx] # Shape: (3, 9, ny, nx)

            # Combine channels and distributions: (3, 9, ny, nx) -> (27, ny, nx)
            f_t_combined = f_t_data.reshape(-1, f_t_data.shape[-2], f_t_data.shape[-1])
            delta_f_combined = delta_f_data.reshape(-1, delta_f_data.shape[-2], delta_f_data.shape[-1])

        # The timestep 't' for the model is 1-based (1 to total_steps)
        time_step_t = torch.tensor([time_step_idx + 1], dtype=torch.float32) # Model expects float

        return torch.from_numpy(f_t_combined).float(), time_step_t, torch.from_numpy(delta_f_combined).float()


def train():
    print(f"Using device: {DEVICE}")
    # Dataset and DataLoader
    dataset = LBMDiffusionDataset(DATA_DIR, TOTAL_STEPS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # Adjust num_workers

    # Model
    model = UNetLBM(n_channels=N_CHANNELS, n_out_channels=N_OUT_CHANNELS).to(DEVICE)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (f_t, t, delta_f_true) in enumerate(dataloader):
            f_t = f_t.to(DEVICE)
            t = t.to(DEVICE).squeeze(-1) # Ensure t is [B]
            delta_f_true = delta_f_true.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            delta_f_pred = model(f_t, t)

            # Calculate loss
            loss = criterion(delta_f_pred, delta_f_true)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0: # Print progress every 100 batches
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}')

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {epoch_loss:.6f}')

        # Optional: Add validation loop here

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()