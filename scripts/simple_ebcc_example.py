#!/usr/bin/env python3
"""
Simple ERA5 compression using EBCC with ensemble spread as error bounds
"""

import numpy as np
import xarray as xr
import sys
import os

from ebcc_wrapper import EBCCDirectWrapper

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
ratio = 1
# Create compressor and compress
compressor = EBCCDirectWrapper()
compressed = compressor.compress(data, error_bound, ratio=ratio)

# Decompress
decompressed = compressor.decompress(compressed)

# Results
check_passed = ((decompressed-data) <= error_bound*ratio).all()
original_size = data.nbytes
compressed_size = len(compressed)
compression_ratio = original_size / compressed_size

print(f"ERA5 {variable} {year}/{month}")
print(f"Check passed: {check_passed}")
print(f"Data shape: {data.shape}")
print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {compression_ratio:.2f}x")