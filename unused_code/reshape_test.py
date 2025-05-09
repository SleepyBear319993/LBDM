import numpy as np

# --- Parameters for testing ---
nx = 3
ny = 2
num_distributions = 9
# --- ---

def idx_host(i, j, k, nx, ny):
    """Linear memory indexing for CPU (matches the definition in diffusion.py)"""
    # flat_index = i + j * nx + k * nx * ny
    # Order: k (slowest), j (middle), i (fastest)
    return i + j * nx + k * nx * ny

# 1. Create a predictable flat array
flat_size = nx * ny * num_distributions
flat_array = np.zeros(flat_size, dtype=np.int32)

print(f"Creating flat array of size {flat_size} with nx={nx}, ny={ny}, num_distributions={num_distributions}")
print("Value stored at index for (i, j, k) will be: k * 10000 + j * 100 + i")

for k in range(num_distributions):
    for j in range(ny):
        for i in range(nx):
            flat_index = idx_host(i, j, k, nx, ny)
            # Store a unique value encoding (k, j, i)
            encoded_value = k * 10000 + j * 100 + i
            flat_array[flat_index] = encoded_value
            print(f"  idx({i},{j},{k}) -> flat_index={flat_index}, value={encoded_value}") # Uncomment for detailed view

# 2. Apply the reshape and transpose logic
# This is the line from data_generator.py
# reshaped_array = flat_array.reshape((ny, nx, num_distributions)).transpose(2, 0, 1)
reshaped_array = flat_array.reshape((num_distributions, ny, nx))

print(f"\nFlat array reshaped to shape: {reshaped_array.shape}")
expected_shape = (num_distributions, ny, nx)
if reshaped_array.shape != expected_shape:
    print(f"ERROR: Reshaped array has incorrect shape! Expected {expected_shape}, got {reshaped_array.shape}")
    exit()
else:
    print(f"Reshaped array has the expected shape: {expected_shape}")


# 3. Verify the values in the reshaped array
correct = True
print("\nVerifying values in the reshaped array (should match reshaped_array[k, j, i] == encoded_value(k,j,i))...")
for k_test in range(num_distributions):
    for j_test in range(ny):
        for i_test in range(nx):
            # Get the value from the reshaped array at index (k_test, j_test, i_test)
            value_in_reshaped = reshaped_array[k_test, j_test, i_test]

            # Decode the original (k, j, i) from the stored value
            original_k = value_in_reshaped // 10000
            original_j = (value_in_reshaped % 10000) // 100
            original_i = value_in_reshaped % 100

            # Check if the indices match
            if not (k_test == original_k and j_test == original_j and i_test == original_i):
                print(f"ERROR at index ({k_test}, {j_test}, {i_test}):")
                print(f"  Value found: {value_in_reshaped}")
                print(f"  Decoded original indices: (i={original_i}, j={original_j}, k={original_k})")
                print(f"  Expected indices based on position: (i={i_test}, j={j_test}, k={k_test})")
                correct = False
                # break # Optional: stop at first error
    # if not correct: break
# if not correct: break

if correct:
    print("\nVerification successful! The reshape operation correctly maps flat index -> (k, j, i)")
else:
    print("\nVerification failed! The reshape operation does not correctly map flat index -> (k, j, i)")

# Optional: Alternative reshape (more direct based on idx_host)
# print("\nTesting alternative reshape: flat_array.reshape((num_distributions, ny, nx))")
# try:
#     alt_reshaped_array = flat_array.reshape((num_distributions, ny, nx))
#     print(f"Alternative reshape successful, shape: {alt_reshaped_array.shape}")
#     # Verify this one too
#     alt_correct = True
#     for k_test in range(num_distributions):
#         for j_test in range(ny):
#             for i_test in range(nx):
#                 value_in_alt_reshaped = alt_reshaped_array[k_test, j_test, i_test]
#                 original_k = value_in_alt_reshaped // 10000
#                 original_j = (value_in_alt_reshaped % 10000) // 100
#                 original_i = value_in_alt_reshaped % 100
#                 if not (k_test == original_k and j_test == original_j and i_test == original_i):
#                     alt_correct = False
#                     break
#         if not alt_correct: break
#     if not alt_correct: break

#     if alt_correct:
#         print("Verification successful for alternative reshape!")
#         # Check if results are identical
#         if np.array_equal(reshaped_array, alt_reshaped_array):
#             print("Both reshape methods produce identical results.")
#         else:
#             print("WARNING: Alternative reshape produces a different result than the original method.")
#     else:
#         print("Verification failed for alternative reshape!")

# except ValueError as e:
#     print(f"Alternative reshape failed with error: {e}")
