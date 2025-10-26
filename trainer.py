import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import glob
import logging
from datetime import datetime

# Assuming UNetLBM is in unet_lbm.py
from unet_lbm import UNetLBM

# --- Configuration ---
TRAIN_DATA_DIR = "lbm_reversal_data_cifar0_32_train" # <<< ADDED: Path to training HDF5 files
TEST_DATA_DIR = "lbm_reversal_data_cifar0_32_test"   # <<< ADDED: Path to test/validation HDF5 files
IMG_SIZE = 32
N_CHANNELS = 27 # 3 RGB * 9 LBM distributions
N_OUT_CHANNELS = 27
TOTAL_STEPS = 1000 # T used during data generation
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "unet_lbm_model_32_reversal_epoch{epoch}.pth"
# --- ---

class LBMDiffusionDataset(Dataset):
    def __init__(self, total_steps, file_paths_list=None, data_dir=None):
        self.total_steps = total_steps
        if file_paths_list is not None:
            self.file_paths = file_paths_list
        elif data_dir is not None:
            self.data_dir = data_dir # Store data_dir for error messages
            self.file_paths = glob.glob(os.path.join(data_dir, "*.h5"))
        else:
            raise ValueError("Either file_paths_list or data_dir must be provided.")

        self.num_images = len(self.file_paths)
        self._check_data()

    def _check_data(self):
        if not self.file_paths:
            if hasattr(self, 'data_dir') and self.data_dir: # Check if data_dir was provided
                raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")
            else: # This case should ideally not be hit if constructor logic is correct
                raise FileNotFoundError(f"No HDF5 files provided or found.")
        # Optional: Open one file to check dimensions
        with h5py.File(self.file_paths[0], 'r') as hf:
            assert hf['f_t'].shape[0] == self.total_steps, f"Mismatch in total_steps in {self.file_paths[0]}"
            assert hf['f_t'].shape[1] * hf['f_t'].shape[2] == N_CHANNELS, f"Mismatch in channels/distributions in {self.file_paths[0]}"
            # Add more checks if needed
        logging.info(f"Dataset check: Found {self.num_images} HDF5 files for this dataset instance (from {getattr(self, 'data_dir', 'list')}).")


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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE_NAME = f"training_{timestamp}.log"
    LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )
    # ---

    logging.info(f"Using device: {DEVICE}")

    # Get training file paths
    train_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.h5"))
    if not train_files:
        logging.error(f"No HDF5 files found in training directory: {TRAIN_DATA_DIR}.")
        raise FileNotFoundError(f"No HDF5 files found in training directory: {TRAIN_DATA_DIR}.")
    logging.info(f"Found {len(train_files)} files in training directory: {TRAIN_DATA_DIR}")

    # Get test/validation file paths
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.h5"))
    if not test_files:
        logging.warning(f"No HDF5 files found in test directory: {TEST_DATA_DIR}. Proceeding without test/validation set.")
        # Proceeding without test files means test_dataloader will be None or empty
    else:
        logging.info(f"Found {len(test_files)} files in test directory: {TEST_DATA_DIR}")

    # Datasets and DataLoaders
    train_dataset = LBMDiffusionDataset(total_steps=TOTAL_STEPS, data_dir=TRAIN_DATA_DIR) # Use data_dir directly
    logging.info(f"Train dataset: {train_dataset.num_images} HDF5 files, {len(train_dataset)} total samples.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    test_dataloader = None
    if test_files: # Only create test dataloader if test files were found
        test_dataset = LBMDiffusionDataset(total_steps=TOTAL_STEPS, data_dir=TEST_DATA_DIR) # Use data_dir directly
        logging.info(f"Test dataset: {test_dataset.num_images} HDF5 files, {len(test_dataset)} total samples.")
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        logging.info("Skipping test dataset creation as no files were found.")


    # Model
    model = UNetLBM(n_channels=N_CHANNELS, n_out_channels=N_OUT_CHANNELS).to(DEVICE)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logging.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (f_t, t, delta_f_true) in enumerate(train_dataloader):
            f_t = f_t.to(DEVICE)
            t = t.to(DEVICE).squeeze(-1) 
            delta_f_true = delta_f_true.to(DEVICE)

            optimizer.zero_grad()
            delta_f_pred = model(f_t, t)
            loss = criterion(delta_f_pred, delta_f_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0: 
                logging.info(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Training Loss: {loss.item():.10f}')

        epoch_loss = running_loss / len(train_dataloader)
        logging.info(f'Epoch [{epoch+1}/{EPOCHS}] completed. Average Training Loss: {epoch_loss:.10f}')

        # Test/Validation phase
        if test_dataloader: # Only run validation if test_dataloader was created
            model.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for i_test, (f_t_test, t_test, delta_f_true_test) in enumerate(test_dataloader):
                    f_t_test = f_t_test.to(DEVICE)
                    t_test = t_test.to(DEVICE).squeeze(-1)
                    delta_f_true_test = delta_f_true_test.to(DEVICE)

                    delta_f_pred_test = model(f_t_test, t_test)
                    test_loss_item = criterion(delta_f_pred_test, delta_f_true_test)
                    running_test_loss += test_loss_item.item()
                    if (i_test + 1) % 100 == 0: 
                        logging.info(f'Epoch [{epoch+1}/{EPOCHS}], Test Step [{i_test+1}/{len(test_dataloader)}], Test Loss: {test_loss_item.item():.10f}')
            
            if len(test_dataloader) > 0:
                epoch_test_loss = running_test_loss / len(test_dataloader)
                logging.info(f'Epoch [{epoch+1}/{EPOCHS}] Test Loss: {epoch_test_loss:.10f}')
            else:
                 logging.info(f'Epoch [{epoch+1}/{EPOCHS}] Test Loss: N/A (empty test dataloader)')
        else:
            logging.info(f'Epoch [{epoch+1}/{EPOCHS}] Test/Validation skipped (no test data).')

    # Save the trained model
    final_model_path = MODEL_SAVE_PATH.format(epoch=EPOCHS) 
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Training finished. Model saved to {final_model_path}")

if __name__ == "__main__":
    train()