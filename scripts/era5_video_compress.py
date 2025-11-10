import os
import sys
import time
import numpy as np
import xarray as xr

from ebcc_wrapper.video_wrapper import FFmpegVideoArrayCompressor
  
# ERA5 data configuration
era5_path = '/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
year = '2024'
month = '12'
variable = '10m_u_component_of_wind'

# File paths
reanalysis_file = os.path.join(era5_path, f'single_level/reanalysis/{year}/{month}/{variable}.nc')
interpolated_ensemble_spread_file = os.path.join(era5_path, f'single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc')

# Load ERA5 data and ensemble spread (error bounds)
data_ds = xr.open_dataset(reanalysis_file)
spread_ds = xr.open_dataset(interpolated_ensemble_spread_file)

# Extract data arrays (assuming single variable per file)
data_var_name = list(data_ds.data_vars.keys())[0]
spread_var_name = list(spread_ds.data_vars.keys())[0]

data = data_ds[data_var_name].values.astype(np.float32)
error_bound = spread_ds[spread_var_name].values.astype(np.float32)

steps = 8
data = data[:steps]
error_bound = error_bound[:steps]

start_time = time.time()
arr = np.asarray(data, dtype=np.float32)

# record original stats
original_size = arr.nbytes

# normalize to [0,1] using global min/max
arr_min = float(arr.min())
arr_max = float(arr.max())
if arr_max == arr_min:
  print("Warning: constant array detected; producing all-zero normalized frames")
  arr_norm = np.zeros_like(arr, dtype=np.float32)
else:
  arr_norm = (arr - arr_min) / (arr_max - arr_min)
  arr_norm = np.clip(arr_norm, 0.0, 1.0).astype(np.float32)

# compress using FFmpegVideoArrayCompressor
compressor = FFmpegVideoArrayCompressor()

bitstream = compressor.compress(arr_norm)
compressed_size = len(bitstream)

arr_dec_norm = compressor.decompress(bitstream)

# map back to original scale
if arr_max == arr_min:
  arr_hat = np.full_like(arr, fill_value=arr_min, dtype=np.float32)
else:
  arr_hat = arr_dec_norm.astype(np.float32) * (arr_max - arr_min) + arr_min

elapsed_time = time.time() - start_time
throughput = original_size / elapsed_time / (1024**2)  # MB/s
print(f"\nCompression completed in {elapsed_time:.2f}s, throughput: {throughput:.2f} MB/s")

# compute error metrics
diff = arr_hat - arr.astype(np.float32)
abs_err = np.abs(diff)
max_abs_err = float(np.max(abs_err))
mse = float(np.mean(diff.astype(np.float64) ** 2))

compression_ratio = float(original_size) / float(compressed_size) if compressed_size > 0 else float('inf')

print("\nResults:")
print(f"Variable: {variable}")
print(f"Frames: {arr.shape[0]}, frame size: {arr.shape[1]}x{arr.shape[2]}")
print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {compression_ratio:.4f}x")
print(f"Max absolute error: {max_abs_err}")
print(f"MSE: {mse}")