import math
import os
import warnings

import numpy as np
import torch
from PIL import Image
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from diffusion_reversal_sto import LBMDiffusionReversalSolverStochastic
from lattice_constants import DTYPE
from unet_lbm import UNetLBM

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


IMG_SIZE = 32
NUM_RGB_CHANNELS = 3
NUM_DIRECTIONS = 9
TOTAL_STEPS = 1000
OMEGA = 0.01
OMEGA_NOISE = 0.01
MODEL_PATH = "unet_lbm_sto_model_32_reversal_epoch25.pth"
INPUT_IMAGE_NAME = "img35"
INPUT_IMAGE_SUFFIX = "png"
OUTPUT_DIR = "bin3_sto"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lbm_state_to_tensor(f_state, nx, ny, device):
	data_channels = []
	for ch in range(NUM_RGB_CHANNELS):
		host_flat = f_state[ch].copy_to_host()
		data_channels.append(host_flat.reshape(NUM_DIRECTIONS, ny, nx))
	stacked = np.stack(data_channels, axis=0).reshape(NUM_RGB_CHANNELS * NUM_DIRECTIONS, ny, nx)
	tensor = torch.from_numpy(stacked.astype(np.float32, copy=False)).unsqueeze(0).to(device)
	return tensor


def tensor_noise_to_device_arrays(noise_tensor, nx, ny, step_index, solver):
	size_per_step = nx * ny * NUM_DIRECTIONS
	noise_np = noise_tensor.squeeze(0).detach().cpu().numpy()
	reshaped = noise_np.reshape(NUM_RGB_CHANNELS, NUM_DIRECTIONS, ny, nx)
	start = step_index * size_per_step
	end = start + size_per_step
	for ch in range(NUM_RGB_CHANNELS):
		noise_flat = reshaped[ch].reshape(-1).astype(DTYPE, copy=False)
		cuda.to_device(noise_flat, to=solver.noise_storage[ch][start:end])


def prepare_input_image(path, nx, ny):
	img = Image.open(path).convert("RGB")
	if img.size != (nx, ny):
		img = img.resize((nx, ny))
	arr = np.array(img, dtype=np.float32) / 255.0
	return np.ascontiguousarray(arr.astype(DTYPE, copy=False))


def main():
	nx = ny = IMG_SIZE
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	model = UNetLBM(n_channels=NUM_RGB_CHANNELS * NUM_DIRECTIONS,
					n_out_channels=NUM_RGB_CHANNELS * NUM_DIRECTIONS).to(DEVICE)
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()

	print(f"Using device: {DEVICE}")
	print(f"Loaded model from {MODEL_PATH}")

	image_path = f"assets/{INPUT_IMAGE_NAME}.{INPUT_IMAGE_SUFFIX}"
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Input image not found: {image_path}")
	rgb_values = prepare_input_image(image_path, nx, ny)
	print(f"Loaded input image {image_path}")

	solver = LBMDiffusionReversalSolverStochastic(nx, ny, OMEGA, OMEGA_NOISE)
	solver.initialize_from_image(rgb_values)
	solver.prepare_noise_storage(TOTAL_STEPS)

	print(f"Running forward stochastic diffusion for {TOTAL_STEPS} steps...")
	for step in range(TOTAL_STEPS):
		solver.forward_step()
		if (step + 1) % 100 == 0 or step == TOTAL_STEPS - 1:
			print(f"Forward step {step + 1}/{TOTAL_STEPS}")
	cuda.synchronize()

	current_t = TOTAL_STEPS
	for step in range(TOTAL_STEPS, 0, -1):
		state_tensor = lbm_state_to_tensor(solver.f, nx, ny, DEVICE)
		t_tensor = torch.tensor([current_t], dtype=torch.float32, device=DEVICE)
		with torch.no_grad():
			predicted_noise = model(state_tensor, t_tensor)
		if DEVICE.type == "cuda":
			torch.cuda.synchronize()
		tensor_noise_to_device_arrays(predicted_noise, nx, ny, current_t - 1, solver)
		solver.reverse_step()
		cuda.synchronize()
		if step % 100 == 0 or step == 1:
			print(f"Reversed to step {current_t - 1}")
		current_t -= 1

	generated = solver.get_result_image()

	output_filename = os.path.join(OUTPUT_DIR, f"generated_{INPUT_IMAGE_NAME}_T{TOTAL_STEPS}.png")
	Image.fromarray(np.clip(generated * 255.0, 0, 255).astype(np.uint8)).save(output_filename)

	print(f"Saved generated image to {output_filename}")

if __name__ == "__main__":
	main()
