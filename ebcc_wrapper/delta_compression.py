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
                
                # Binary search for optimal constant error bound for delta
                # Start from RMSE of delta
                rmse = np.sqrt(np.mean(delta**2))
                const_error_bound = np.full_like(delta, rmse)
                
                # Try compressing delta with constant error bound
                compressed_delta_base = self.compressor.compress(delta, const_error_bound, ratio=ratio)
                decompressed_delta = self.compressor.decompress(compressed_delta_base)
                reconstructed = prediction + decompressed_delta
                
                # Find fail values where reconstructed exceeds original error bound
                fail_mask = np.abs(data - reconstructed) > error_bound * ratio
                fail_idx = np.flatnonzero(fail_mask).astype(np.int32)
                fail_val = data.flat[fail_idx].astype(np.float32) if fail_idx.size else np.array([], dtype=np.float32)
                
                # Encode fail values
                if fail_idx.size:
                    cmask = zlib.compress(np.packbits(fail_mask.ravel()).tobytes(), level=6)
                    cidx = zlib.compress(fail_idx.tobytes(), level=6)
                    cval = zlib.compress(fail_val.tobytes(), level=6)
                    if len(cmask) <= len(cidx):
                        fail_info = pickle.dumps({'mask': cmask, 'val': cval})
                    else:
                        fail_info = pickle.dumps({'idx': cidx, 'val': cval})
                    fail_bytes = b'DELTAFAIL' + fail_info
                else:
                    fail_bytes = b''
                
                current_length = len(compressed_delta_base) + len(fail_bytes)
                best_length = current_length
                best_error_bound = rmse
                best_compressed = compressed_delta_base
                best_fail_bytes = fail_bytes
                
                # Binary search: try halving error bound
                test_eb = rmse / 2
                while test_eb > 0:
                    const_eb = np.full_like(delta, test_eb)
                    comp_delta = self.compressor.compress(delta, const_eb, ratio=ratio)
                    dec_delta = self.compressor.decompress(comp_delta)
                    recon = prediction + dec_delta
                    
                    fmask = np.abs(data - recon) > error_bound * ratio
                    fidx = np.flatnonzero(fmask).astype(np.int32)
                    
                    if fidx.size:
                        fval = data.flat[fidx].astype(np.float32)
                        cmask = zlib.compress(np.packbits(fmask.ravel()).tobytes(), level=6)
                        cidx = zlib.compress(fidx.tobytes(), level=6)
                        cval = zlib.compress(fval.tobytes(), level=6)
                        if len(cmask) <= len(cidx):
                            finfo = pickle.dumps({'mask': cmask, 'val': cval})
                        else:
                            finfo = pickle.dumps({'idx': cidx, 'val': cval})
                        fbytes = b'DELTAFAIL' + finfo
                    else:
                        fbytes = b''
                    
                    test_length = len(comp_delta) + len(fbytes)
                    if test_length < best_length:
                        best_length = test_length
                        best_error_bound = test_eb
                        best_compressed = comp_delta
                        best_fail_bytes = fbytes
                        test_eb /= 2
                    else:
                        break
                
                # Binary search: try doubling error bound
                test_eb = rmse * 2
                for _ in range(10):  # Limit iterations
                    const_eb = np.full_like(delta, test_eb)
                    comp_delta = self.compressor.compress(delta, const_eb, ratio=ratio)
                    dec_delta = self.compressor.decompress(comp_delta)
                    recon = prediction + dec_delta
                    
                    fmask = np.abs(data - recon) > error_bound * ratio
                    fidx = np.flatnonzero(fmask).astype(np.int32)
                    
                    if fidx.size:
                        fval = data.flat[fidx].astype(np.float32)
                        cmask = zlib.compress(np.packbits(fmask.ravel()).tobytes(), level=6)
                        cidx = zlib.compress(fidx.tobytes(), level=6)
                        cval = zlib.compress(fval.tobytes(), level=6)
                        if len(cmask) <= len(cidx):
                            finfo = pickle.dumps({'mask': cmask, 'val': cval})
                        else:
                            finfo = pickle.dumps({'idx': cidx, 'val': cval})
                        fbytes = b'DELTAFAIL' + finfo
                    else:
                        fbytes = b''
                    
                    test_length = len(comp_delta) + len(fbytes)
                    if test_length < best_length:
                        best_length = test_length
                        best_error_bound = test_eb
                        best_compressed = comp_delta
                        best_fail_bytes = fbytes
                        test_eb *= 2
                    else:
                        break
                
                # Combine compressed delta and fail values
                compressed_delta = best_compressed + best_fail_bytes
                
                # Compare with direct compression
                compressed_direct = self.compressor.compress(data, error_bound, ratio=ratio)
                
                # Choose smaller, store flag
                if len(compressed_delta) < len(compressed_direct):
                    compressed_dict[pressure_level] = b'DELTA' + compressed_delta
                    cr = data.nbytes / len(compressed_delta)
                    print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(compressed_delta):,} bytes) [DELTA, eb={best_error_bound:.4f}]")
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
                
                # Extract delta bitstream and fail values
                bitstream = compressed[5:]
                if b'DELTAFAIL' in bitstream:
                    pos = bitstream.index(b'DELTAFAIL')
                    delta_bytes = bitstream[:pos]
                    fail_info = pickle.loads(bitstream[pos + 9:])
                    
                    # Decompress delta
                    delta = self.compressor.decompress(delta_bytes)
                    reconstructed = prediction + delta
                    
                    # Restore fail values
                    if 'mask' in fail_info:
                        fail_mask = np.unpackbits(np.frombuffer(zlib.decompress(fail_info['mask']), dtype=np.uint8))
                        fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
                        fail_idx = np.flatnonzero(fail_mask[:reconstructed.size])
                    else:
                        fail_idx = np.frombuffer(zlib.decompress(fail_info['idx']), dtype=np.int32)
                        fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
                    
                    reconstructed.flat[fail_idx] = fail_val
                    decompressed_dict[pressure_level] = reconstructed
                else:
                    # No fail values
                    delta = self.compressor.decompress(bitstream)
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