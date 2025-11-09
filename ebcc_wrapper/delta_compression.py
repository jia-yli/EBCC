#!/usr/bin/env python3
"""
Delta compression scheme for ERA5 pressure levels data.
Exploits correlation between adjacent pressure levels to improve compression efficiency.
"""

import numpy as np
import xarray as xr
import zlib
import pickle
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from .direct_wrapper import EBCCDirectWrapper


def _load_single_pressure_level_worker(era5_path: str, year: str, month: str,
                                      variable: str, pressure_level: str, steps: int) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Standalone function to load data for a single pressure level.
    This needs to be a module-level function for ProcessPoolExecutor.
    
    Returns:
        Tuple of (pressure_level, data, error_bound) or (pressure_level, None, None) if failed
    """
    reanalysis_file = f'{era5_path}/pressure_level/reanalysis/{year}/{month}/{pressure_level}/{variable}.nc'
    ensemble_spread_file = f'{era5_path}/pressure_level/interpolated_ensemble_spread/{year}/{month}/{pressure_level}/{variable}.nc'
    
    try:
        # Load data
        data_ds = xr.open_dataset(reanalysis_file)
        spread_ds = xr.open_dataset(ensemble_spread_file)
        
        data_var_name = list(data_ds.data_vars.keys())[0]
        spread_var_name = list(spread_ds.data_vars.keys())[0]
        
        data = data_ds[data_var_name].values.astype(np.float32)[:steps]
        error_bound = spread_ds[spread_var_name].values.astype(np.float32)[:steps]
        
        # Remove pressure dimension if it exists and equals 1
        if data.ndim == 4 and data.shape[1] == 1:
            data = data.squeeze(axis=1)  # Remove pressure dim
            error_bound = error_bound.squeeze(axis=1)
        
        data_ds.close()
        spread_ds.close()
        
        return pressure_level, data, error_bound
        
    except Exception as e:
        print(f"Warning: Could not load pressure level {pressure_level}: {e}")
        return pressure_level, None, None


class PressureLevelDeltaCompressor:
    """
    Delta compression for pressure level data.
    
    The scheme works by:
    1. Compressing the first (reference) pressure level normally
    2. For subsequent levels, computing deltas from the previous level
    3. Compressing deltas using adjusted error bounds
    4. Optionally using predictive models for better delta estimation
    """
    
    def __init__(self):
        """
        Initialize delta compressor.
        """
        self.compressor = EBCCDirectWrapper()

    def load_pressure_level_data(self, era5_path: str, year: str, month: str, 
                                variable: str, pressure_levels: List[str],
                                steps: int = 8, max_workers: int = 4) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load data for multiple pressure levels in parallel using processes.
        
        Args:
            max_workers: Maximum number of parallel processes for loading (recommended: 2-6)
        
        Returns:
            Tuple of (data_dict, error_bound_dict) where keys are pressure levels
        """
        data_dict = {}
        error_bound_dict = {}
        
        # Limit max_workers to prevent resource exhaustion
        max_workers = min(max_workers, 64)  # Cap at 64 processes max
        max_workers = min(max_workers, len(pressure_levels))  # Don't exceed number of files
        
        print(f"Loading {len(pressure_levels)} pressure levels in parallel (max_workers={max_workers} processes)...")
        
        # Use ProcessPoolExecutor for parallel I/O operations
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_level = {
                    executor.submit(_load_single_pressure_level_worker, era5_path, year, month, variable, level, steps): level
                    for level in pressure_levels
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_level):
                    try:
                        pressure_level, data, error_bound = future.result()
                        completed_count += 1
                        
                        if data is not None and error_bound is not None:
                            data_dict[pressure_level] = data
                            error_bound_dict[pressure_level] = error_bound
                            print(f"  ✓ Loaded {pressure_level} hPa ({completed_count}/{len(pressure_levels)})")
                        else:
                            print(f"  ✗ Failed {pressure_level} hPa ({completed_count}/{len(pressure_levels)})")
                    except Exception as e:
                        completed_count += 1
                        print(f"  ✗ Error loading pressure level: {e} ({completed_count}/{len(pressure_levels)})")
                        
        except Exception as e:
            print(f"Error in parallel loading: {e}")
            print("Falling back to sequential loading...")
            # Fallback to sequential loading
            return self._load_sequential(era5_path, year, month, variable, pressure_levels, steps)
                
        return data_dict, error_bound_dict
    
    def _load_sequential(self, era5_path: str, year: str, month: str, 
                        variable: str, pressure_levels: List[str],
                        steps: int = 8) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Fallback sequential loading method.
        """
        data_dict = {}
        error_bound_dict = {}
        
        print("Loading sequentially...")
        for i, pressure_level in enumerate(pressure_levels):
            print(f"  Loading {pressure_level} hPa ({i+1}/{len(pressure_levels)})...")
            level, data, error_bound = _load_single_pressure_level_worker(
                era5_path, year, month, variable, pressure_level, steps
            )
            
            if data is not None and error_bound is not None:
                data_dict[pressure_level] = data
                error_bound_dict[pressure_level] = error_bound
                print(f"    ✓ Success")
            else:
                print(f"    ✗ Failed")
                
        return data_dict, error_bound_dict
    
    def compute_prediction(self, prev_data: np.ndarray, curr_pressure: float, 
                          prev_pressure: float) -> np.ndarray:
        """
        Compute prediction for current level based on previous level.
        
        For simplicity and to avoid circular dependencies in decompression,
        we use only the previous level for prediction.
        
        Args:
            prev_data: Data from previous pressure level
            curr_pressure: Current pressure level value
            prev_pressure: Previous pressure level value
            
        Returns:
            Predicted data for current level
        """
        # Simple strategy: use previous level as prediction
        # This avoids circular dependencies and is still effective
        # for correlated atmospheric data
        return prev_data.copy()
    
    def compress_delta(self, data_dict: Dict[str, np.ndarray], 
                      error_bound_dict: Dict[str, np.ndarray],
                      pressure_levels: List[str],
                      ratio: float = 1.0) -> Dict[str, bytes]:
        """
        Compress multiple pressure levels using delta compression.
        
        Args:
            data_dict: Dictionary of data arrays keyed by pressure level
            error_bound_dict: Dictionary of error bounds keyed by pressure level
            pressure_levels: Ordered list of pressure levels to compress
            ratio: Compression ratio parameter
            
        Returns:
            Dictionary of compressed data keyed by pressure level
        """
        compressed_dict = {}
        pressure_values = [float(p) for p in pressure_levels]
        
        for i, pressure_level in enumerate(pressure_levels):
            if pressure_level not in data_dict:
                continue
                
            data = data_dict[pressure_level]
            error_bound = error_bound_dict[pressure_level]
            
            if i == 0:
                # Compress first level normally (reference level)
                compressed = self.compressor.compress(data, error_bound, ratio=ratio)
                compressed_dict[pressure_level] = compressed
                cr = data.nbytes / len(compressed)
                print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(compressed):,} bytes) [DIRECT]")
            else:
                # Try both delta and direct compression, choose smaller
                prev_pressure_level = pressure_levels[i-1]
                
                prev_data = data_dict[prev_pressure_level]
                prediction = self.compute_prediction(prev_data, pressure_values[i], pressure_values[i-1])
                delta = data - prediction
                
                # Compress both delta and direct
                compressed_delta = self.compressor.compress(delta, error_bound, ratio=ratio)
                compressed_direct = self.compressor.compress(data, error_bound, ratio=ratio)
                
                # Choose smaller, store flag
                if len(compressed_delta) < len(compressed_direct):
                    compressed_dict[pressure_level] = b'DELTA' + compressed_delta
                    cr = data.nbytes / len(compressed_delta)
                    print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(compressed_delta):,} bytes) [DELTA]")
                else:
                    compressed_dict[pressure_level] = b'DIRECT' + compressed_direct
                    cr = data.nbytes / len(compressed_direct)
                    print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(compressed_direct):,} bytes) [DIRECT]")
                
        return compressed_dict
    
    def decompress_delta(self, compressed_dict: Dict[str, bytes],
                        pressure_levels: List[str]) -> Dict[str, np.ndarray]:
        """
        Decompress delta-compressed pressure level data.
        
        Args:
            compressed_dict: Dictionary of compressed data
            pressure_levels: Ordered list of pressure levels
            
        Returns:
            Dictionary of decompressed data
        """
        decompressed_dict = {}
        pressure_values = [float(p) for p in pressure_levels]
        
        for i, pressure_level in enumerate(pressure_levels):
            if pressure_level not in compressed_dict:
                continue
                
            compressed = compressed_dict[pressure_level]
            
            if i == 0 or compressed.startswith(b'DIRECT'):
                # Direct decompression
                bitstream = compressed[6:] if compressed.startswith(b'DIRECT') else compressed
                decompressed_dict[pressure_level] = self.compressor.decompress(bitstream)
            elif compressed.startswith(b'DELTA'):
                # Delta decompression
                prev_pressure_level = pressure_levels[i-1]
                if prev_pressure_level not in decompressed_dict:
                    decompressed_dict[pressure_level] = self.compressor.decompress(compressed[5:])
                    continue
                
                prev_data = decompressed_dict[prev_pressure_level]
                prediction = self.compute_prediction(prev_data, pressure_values[i], pressure_values[i-1])
                delta = self.compressor.decompress(compressed[5:])
                decompressed_dict[pressure_level] = prediction + delta
                
        return decompressed_dict
    
    def analyze_compression_efficiency(self, data_dict: Dict[str, np.ndarray],
                                     compressed_dict: Dict[str, bytes],
                                     pressure_levels: List[str]) -> Dict:
        """
        Analyze compression efficiency and provide statistics.
        """
        total_original = 0
        total_compressed = 0
        level_stats = {}
        
        for pressure_level in pressure_levels:
            if pressure_level in data_dict and pressure_level in compressed_dict:
                original_size = data_dict[pressure_level].nbytes
                compressed_size = len(compressed_dict[pressure_level])
                ratio = original_size / compressed_size
                
                level_stats[pressure_level] = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': ratio,
                    'data_shape': data_dict[pressure_level].shape
                }
                
                total_original += original_size
                total_compressed += compressed_size
        
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
        
        return {
            'level_stats': level_stats,
            'total_original_size': total_original,
            'total_compressed_size': total_compressed,
            'overall_compression_ratio': overall_ratio
        }