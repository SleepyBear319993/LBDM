import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split # Added import
import logging # Added import
from datetime import datetime # Added import

# Assuming UNetLBM is in unet_lbm.py
from unet_lbm import UNetLBM

# --- Configuration ---
DATA_DIR = "lbm_reversal_data_cifar0_32" # <<< CHANGE HERE
IMG_SIZE = 32 # <<< CHANGE HERE
N_CHANNELS = 27 # 3 RGB * 9 LBM distributions
N_OUT_CHANNELS = 27
TOTAL_STEPS = 1000 # T used during data generation
BATCH_SIZE = 64 # Adjust based on GPU memory (might be able to increase for smaller images)
LEARNING_RATE = 1e-4
EPOCHS = 5 # Adjust as needed
VAL_SPLIT = 0.2 # Proportion of data to use for validation <<< ADDED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "unet_lbm_model_32_reversal_e{epoch}.pth" # <<< CHANGE HERE: Added {epoch} placeholder
# --- ---

class LBMDiffusionDataset(Dataset):
    def __init__(self, total_steps, file_paths_list=None, data_dir=None):
        self.total_steps = total_steps
        if file_paths_list is not None:
            self.file_paths = file_paths_list
        elif data_dir is not None:
            self.data_dir = data_dir
            self.file_paths = glob.glob(os.path.join(data_dir, "*.h5"))
        else:
            raise ValueError("Either file_paths_list or data_dir must be provided.")

        self.num_images = len(self.file_paths)
        self._check_data()

    def _check_data(self):
        if not self.file_paths:
            if hasattr(self, 'data_dir'):
                raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")
            else:
                raise FileNotFoundError(f"No HDF5 files provided in file_paths_list")
        # Optional: Open one file to check dimensions
        with h5py.File(self.file_paths[0], 'r') as hf:
            assert hf['f_t'].shape[0] == self.total_steps, "Mismatch in total_steps"
            assert hf['f_t'].shape[1] * hf['f_t'].shape[2] == N_CHANNELS, "Mismatch in channels/distributions"
            # Add more checks if needed
        logging.info(f"Dataset check: Found {self.num_images} HDF5 files for this dataset instance.") # Changed print to logging.info

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
    # --- Logging Setup ---
    LOG_DIR = "log"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE_NAME = f"training_{timestamp}.log"
    LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler() # To also print to console
        ]
    )
    # ---

    logging.info(f"Using device: {DEVICE}")

    # Get all file paths
    all_file_paths = glob.glob(os.path.join(DATA_DIR, "*.h5"))
    if not all_file_paths:
        # Use logging.error for errors before raising them
        logging.error(f"No HDF5 files found in {DATA_DIR}. Cannot split dataset.")
        raise FileNotFoundError(f"No HDF5 files found in {DATA_DIR}. Cannot split dataset.")

    # Split file paths into training and validation sets
    train_files, val_files = train_test_split(all_file_paths, test_size=VAL_SPLIT, random_state=42) # Added random_state for reproducibility

    logging.info(f"Found {len(all_file_paths)} total files. Using {len(train_files)} for training and {len(val_files)} for validation.")

    # Datasets and DataLoaders
    train_dataset = LBMDiffusionDataset(total_steps=TOTAL_STEPS, file_paths_list=train_files)
    # Accessing train_dataset.num_images for logging after it's initialized
    logging.info(f"Train dataset: {train_dataset.num_images} HDF5 files, {len(train_dataset)} total samples.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = LBMDiffusionDataset(total_steps=TOTAL_STEPS, file_paths_list=val_files)
    # Accessing val_dataset.num_images for logging after it's initialized
    logging.info(f"Validation dataset: {val_dataset.num_images} HDF5 files, {len(val_dataset)} total samples.")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    # Model
    model = UNetLBM(n_channels=N_CHANNELS, n_out_channels=N_OUT_CHANNELS).to(DEVICE)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logging.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (f_t, t, delta_f_true) in enumerate(train_dataloader): # Changed to train_dataloader
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
                logging.info(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Training Loss: {loss.item():.10f}')

        epoch_loss = running_loss / len(train_dataloader)
        logging.info(f'Epoch [{epoch+1}/{EPOCHS}] completed. Average Training Loss: {epoch_loss:.10f}')

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i_val, (f_t_val, t_val, delta_f_true_val) in enumerate(val_dataloader):
                f_t_val = f_t_val.to(DEVICE)
                t_val = t_val.to(DEVICE).squeeze(-1)
                delta_f_true_val = delta_f_true_val.to(DEVICE)

                delta_f_pred_val = model(f_t_val, t_val)
                val_loss_item = criterion(delta_f_pred_val, delta_f_true_val)
                running_val_loss += val_loss_item.item()
                if (i_val + 1) % 100 == 0: # Print progress every 100 batches
                    logging.info(f'Epoch [{epoch+1}/{EPOCHS}], Validation Step [{i_val+1}/{len(val_dataloader)}], Validation Loss: {val_loss_item.item():.10f}')
        
        epoch_val_loss = running_val_loss / len(val_dataloader)
        logging.info(f'Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {epoch_val_loss:.10f}')
        # Optional: Add logic for early stopping or saving best model based on validation loss here

    # Save the trained model
    # Format the MODEL_SAVE_PATH with the total number of epochs
    final_model_path = MODEL_SAVE_PATH.format(epoch=EPOCHS) 
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Training finished. Model saved to {final_model_path}")

if __name__ == "__main__":
    train()