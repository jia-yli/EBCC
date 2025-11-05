#!/usr/bin/env python3
"""
Comprehensive test script for delta compression scheme on ERA5 pressure levels data.
Tests compression ratios and validates error bounds are respected.
"""

import numpy as np
import sys
import os

from ebcc_wrapper.delta_compression import PressureLevelDeltaCompressor
from ebcc_wrapper import EBCCDirectWrapper

def main():
    # ERA5 data configuration
    era5_path = '/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
    year = '2024'
    month = '12'
    variable = 'temperature'
    
    # Pressure levels to test (ordered from high to low pressure)
    test_pressure_levels = ["1000", "950", "900", "850", "800", "750", "700", "650", "600", "550", "500"]
    
    print(f"ERA5 Delta Compression Test")
    print(f"Variable: {variable}, Year: {year}, Month: {month}")
    print(f"Testing pressure levels: {test_pressure_levels}")
    print("=" * 80)
    
    # Initialize compressors
    delta_compressor = PressureLevelDeltaCompressor()
    standard_compressor = EBCCDirectWrapper()
    
    # Load data for all pressure levels
    print("Loading pressure level data...")
    data_dict, error_bound_dict = delta_compressor.load_pressure_level_data(
        era5_path, year, month, variable, test_pressure_levels, steps=8, max_workers=64
    )
    
    if not data_dict:
        print("ERROR: No data could be loaded!")
        return False
    
    print(f"Successfully loaded {len(data_dict)} pressure levels")
    for level in test_pressure_levels:
        if level in data_dict:
            print(f"  {level} hPa: shape {data_dict[level].shape}")
    
    # Test parameters
    ratio = 1.0
    
    # Standard compression (baseline)
    print("\n" + "-" * 60)
    print("STANDARD COMPRESSION (Baseline)")
    print("-" * 60)
    
    standard_compressed = {}
    standard_total_original = 0
    standard_total_compressed = 0
    standard_error_check = True
    
    for pressure_level in test_pressure_levels:
        if pressure_level not in data_dict:
            continue
            
        data = data_dict[pressure_level]
        error_bound = error_bound_dict[pressure_level]
        
        # Compress and decompress
        compressed = standard_compressor.compress(data, error_bound, ratio=ratio)
        decompressed = standard_compressor.decompress(compressed)
        
        # Check error bounds
        errors = np.abs(decompressed - data)
        violations = np.sum(errors > error_bound * ratio)
        level_check = violations == 0
        standard_error_check = standard_error_check and level_check
        
        standard_compressed[pressure_level] = compressed
        
        original_size = data.nbytes
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size
        
        standard_total_original += original_size
        standard_total_compressed += compressed_size
        
        print(f"{pressure_level:>4} hPa: {compression_ratio:6.2f}x "
              f"({original_size:,} -> {compressed_size:,} bytes) "
              f"{'✓' if level_check else '✗'} ({violations} violations)")
    
    standard_overall_ratio = standard_total_original / standard_total_compressed
    
    # Delta compression
    print("\n" + "-" * 60)
    print("DELTA COMPRESSION")
    print("-" * 60)
    
    delta_compressed = delta_compressor.compress_delta(
        data_dict, error_bound_dict, test_pressure_levels, ratio=ratio
    )
    
    # Decompress and validate
    decompressed_dict = delta_compressor.decompress_delta(delta_compressed, test_pressure_levels)
    
    # Analyze delta compression results and error bounds
    delta_stats = delta_compressor.analyze_compression_efficiency(
        data_dict, delta_compressed, test_pressure_levels
    )
    
    delta_error_check = True
    for pressure_level in test_pressure_levels:
        if pressure_level in data_dict and pressure_level in decompressed_dict:
            original = data_dict[pressure_level]
            decompressed = decompressed_dict[pressure_level]
            error_bound = error_bound_dict[pressure_level]
            
            # Check error bounds
            errors = np.abs(decompressed - original)
            violations = np.sum(errors > error_bound * ratio)
            level_check = violations == 0
            delta_error_check = delta_error_check and level_check
            
            stats = delta_stats['level_stats'][pressure_level]
            print(f"{pressure_level:>4} hPa: {stats['compression_ratio']:6.2f}x "
                  f"({stats['original_size']:,} -> {stats['compressed_size']:,} bytes) "
                  f"{'✓' if level_check else '✗'} ({violations} violations)")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPRESSION COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"Standard compression:")
    print(f"  Total original size: {standard_total_original:,} bytes")
    print(f"  Total compressed size: {standard_total_compressed:,} bytes")
    print(f"  Overall compression ratio: {standard_overall_ratio:.2f}x")
    print(f"  Error bounds respected: {'YES' if standard_error_check else 'NO'}")
    
    print(f"\nDelta compression:")
    print(f"  Total original size: {delta_stats['total_original_size']:,} bytes")
    print(f"  Total compressed size: {delta_stats['total_compressed_size']:,} bytes")
    print(f"  Overall compression ratio: {delta_stats['overall_compression_ratio']:.2f}x")
    print(f"  Error bounds respected: {'YES' if delta_error_check else 'NO'}")
    
    # Calculate improvement
    improvement = delta_stats['overall_compression_ratio'] / standard_overall_ratio
    bytes_saved = standard_total_compressed - delta_stats['total_compressed_size']
    percent_saved = (bytes_saved / standard_total_compressed) * 100 if standard_total_compressed > 0 else 0
    
    print(f"\n" + "-" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 60)
    print(f"Delta compression improvement: {improvement:.2f}x")
    print(f"Bytes saved: {bytes_saved:,} ({percent_saved:.1f}%)")
    print(f"Pressure levels processed: {len(data_dict)}")
    
    # Final validation status
    overall_success = standard_error_check and delta_error_check
    print(f"\n" + "=" * 80)
    print("FINAL VALIDATION RESULTS")
    print("=" * 80)
    print(f"Standard compression error bounds: {'PASS' if standard_error_check else 'FAIL'}")
    print(f"Delta compression error bounds: {'PASS' if delta_error_check else 'FAIL'}")
    print(f"Overall test status: {'PASS' if overall_success else 'FAIL'}")
    
    if improvement > 1.0 and overall_success:
        print(f"SUCCESS: Delta compression achieved {improvement:.2f}x better ratio with error bounds respected!")
    elif overall_success:
        print(f"SUCCESS: Error bounds respected, but compression improvement is {improvement:.2f}x")
    else:
        print("FAILURE: Error bounds violated!")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)