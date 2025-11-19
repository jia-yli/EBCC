import os
import xarray as xr
import time
import numpy as np

from ebcc_wrapper import ErrorBoundedJP2KCodec

era5_path_np = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
year = '2024'
month = '12'
variable = '2m_temperature'

# File paths
reanalysis_file = os.path.join(era5_path_np, f'single_level/reanalysis/{year}/{month}/{variable}.npy')
interpolated_ensemble_spread_file = os.path.join(era5_path_np, f'single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.npy')

data = np.load(reanalysis_file)
error_bound = np.load(interpolated_ensemble_spread_file)

steps = 8
data = data[:steps]
error_bound = error_bound[:steps]
ratio = 1.0
# error_bound = error_bound*0 + 1
# print("[WARNING] Using constant error bound 1 for testing purposes.")

start_time = time.time()
arr = np.asarray(data, dtype=np.float32)

# record original stats
original_size = arr.nbytes

codec = ErrorBoundedJP2KCodec()
# blob = codec.compress(arr, error_bound*ratio, cratio=20, key_fail_u16='2')
blob = codec.compress(arr, error_bound*ratio, cratio=20)
arr_hat = codec.decompress(blob)

compressed_size = len(blob)

elapsed_time = time.time() - start_time
throughput = original_size / elapsed_time / (1024**2)  # MB/s
print(f"\nCompression completed in {elapsed_time:.2f}s, throughput: {throughput:.2f} MB/s")

# compute error metrics
diff = arr_hat - arr.astype(np.float32)
abs_err = np.abs(diff)
max_abs_err = float(np.max(abs_err))
mse = float(np.mean(diff.astype(np.float64) ** 2))

# check error bounds
error_check = np.all(abs_err <= error_bound * ratio)

compression_ratio = float(original_size) / float(compressed_size) if compressed_size > 0 else float('inf')

print("\nResults:")
print(f"Variable: {variable}")
print(f"Frames: {arr.shape[0]}, frame size: {arr.shape[1]}x{arr.shape[2]}")
print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {compression_ratio:.4f}x")
print(f"Max absolute error: {max_abs_err}")
print(f"MSE: {mse}")
print(f"Error bounds satisfied: {error_check}")
