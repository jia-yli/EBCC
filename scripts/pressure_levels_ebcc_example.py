#!/usr/bin/env python3
"""
ERA5 pressure levels compression using EBCC with ensemble spread as error bounds
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
variable = 'temperature'

# Pressure levels to test
pressure_levels = [
    "1", # "2", "3", # rgb 74.8x 721x1440
    # "5", "7", "10",
    # "20", "30", "50",
    # "70", "100", "125", # 55x
    "150", # "175", "200",
    # "225", "250", "300",
    # "350", "400", "450",
    "500", # "550", "600",
    # "650", "700", "750",
    # "775", "800", "825",
    "850", #"875", "900", # 26.22x
    # "925", "950", "975",
    "1000"
]

# Initialize compression statistics
total_original_size = 0
total_compressed_size = 0
individual_results = []

print(f"ERA5 pressure levels compression test")
print(f"Variable: {variable}, Year: {year}, Month: {month}")
print(f"Testing pressure levels: {pressure_levels}")
print("-" * 60)

# Create compressor
compressor = EBCCDirectWrapper()
ratio = 1

# Process each pressure level
for pressure_level in pressure_levels:
    print(f"\nProcessing pressure level {pressure_level} hPa...")
    
    # File paths for reanalysis and ensemble spread
    reanalysis_file = os.path.join(era5_path, f'pressure_level/reanalysis/{year}/{month}/{pressure_level}/{variable}.nc')
    ensemble_spread_file = os.path.join(era5_path, f'pressure_level/interpolated_ensemble_spread/{year}/{month}/{pressure_level}/{variable}.nc')
    
    try:
        # Load ERA5 data and ensemble spread (error bounds)
        data_ds = xr.open_dataset(reanalysis_file)
        spread_ds = xr.open_dataset(ensemble_spread_file)
        
        # Extract data arrays (assuming single variable per file)
        data_var_name = list(data_ds.data_vars.keys())[0]
        spread_var_name = list(spread_ds.data_vars.keys())[0]
        
        data = data_ds[data_var_name].values.astype(np.float32)
        error_bound = spread_ds[spread_var_name].values.astype(np.float32)
        
        # Limit to specified number of time steps
        steps = 8
        data = data[:steps]
        error_bound = error_bound[:steps]
        
        # Compress
        compressed = compressor.compress(data, error_bound, ratio=ratio)
        
        # Decompress and verify
        decompressed = compressor.decompress(compressed)
        
        # Check compression quality
        check_passed = ((decompressed - data) <= error_bound * ratio).all()
        
        # Calculate compression statistics
        original_size = data.nbytes
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size
        
        # Update totals
        total_original_size += original_size
        total_compressed_size += compressed_size
        
        # Store individual results
        individual_results.append({
            'pressure_level': pressure_level,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'check_passed': check_passed,
            'data_shape': data.shape
        })
        
        print(f"  Data shape: {data.shape}")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Compressed size: {compressed_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Quality check: {'PASSED' if check_passed else 'FAILED'}")
        
        # Close datasets
        data_ds.close()
        spread_ds.close()
        
    except FileNotFoundError as e:
        print(f"  ERROR: File not found - {e}")
        continue
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Calculate overall compression ratio
overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0

print("\n" + "=" * 60)
print("COMPRESSION SUMMARY")
print("=" * 60)
print(f"Total original size: {total_original_size:,} bytes")
print(f"Total compressed size: {total_compressed_size:,} bytes")
print(f"Overall compression ratio: {overall_compression_ratio:.2f}x")
print(f"Successfully processed: {len(individual_results)} pressure levels")

print("\nIndividual Results:")
print("-" * 60)
for result in individual_results:
    print(f"{result['pressure_level']:>4} hPa: {result['compression_ratio']:6.2f}x "
          f"({result['original_size']:,} -> {result['compressed_size']:,} bytes) "
          f"{'✓' if result['check_passed'] else '✗'}")