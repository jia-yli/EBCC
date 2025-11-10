#!/usr/bin/env python3
"""
Delta compression scheme for ERA5 pressure levels data.
Exploits correlation between adjacent pressure levels to improve compression efficiency.
"""

import numpy as np
import xarray as xr
import zlib
import pickle
import time
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
    
    def _encode_fail_values(self, fail_mask: np.ndarray, fail_idx: np.ndarray, 
                           fail_val: np.ndarray) -> bytes:
        """
        Encode fail values into compressed bytes.
        
        Args:
            fail_mask: Boolean mask of failing positions
            fail_idx: Indices of failing positions
            fail_val: Values at failing positions
            
        Returns:
            Encoded fail bytes with DELTAFAIL marker, or empty bytes if no fails
        """
        cmask = zlib.compress(np.packbits(fail_mask.ravel()).tobytes(), level=6)
        cidx = zlib.compress(fail_idx.tobytes(), level=6)
        cval = zlib.compress(fail_val.tobytes(), level=6)
        
        if len(cmask) <= len(cidx):
            fail_info = pickle.dumps({'mask': cmask, 'val': cval, 'len': fail_mask.size})
        else:
            fail_info = pickle.dumps({'idx': cidx, 'val': cval})
        
        return fail_info
    
    def _decode_fail_values(self, fail_info: bytes, data_shape: Tuple[int]) -> np.ndarray:
        fail_info = pickle.loads(fail_info)
        if 'mask' in fail_info:
            fail_mask = np.unpackbits(np.frombuffer(zlib.decompress(fail_info['mask']), dtype=np.uint8))
            fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
            fail_idx = np.flatnonzero(fail_mask[:fail_info['len']])
        else:
            fail_idx = np.frombuffer(zlib.decompress(fail_info['idx']), dtype=np.int32)
            fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
        
        return fail_idx, fail_val
    
    # def _binary_search_optimal_error_bound(self, delta: np.ndarray, prediction: np.ndarray,
    #                                       data: np.ndarray, error_bound: np.ndarray,
    #                                       ratio: float) -> Tuple[bytes, bytes, float]:
    #     """
    #     Binary search for optimal constant error bound for delta compression.
        
    #     Args:
    #         delta: Delta values (data - prediction)
    #         prediction: Predicted values
    #         data: Original data
    #         error_bound: Original error bounds
    #         ratio: Compression ratio parameter
            
    #     Returns:
    #         Tuple of (best_compressed_delta, best_fail_bytes, best_error_bound)
    #     """
    #     # Start from RMSE of delta
    #     rmse = np.sqrt(np.mean(delta**2))
        
    #     print(f"    Starting binary search from RMSE={rmse:.4f}")
        
    #     # Try compressing delta with constant error bound at RMSE
    #     const_error_bound = np.full_like(delta, rmse)
    #     compressed_delta_base = self.compressor.compress(delta, const_error_bound, ratio=ratio)
    #     decompressed_delta = self.compressor.decompress(compressed_delta_base)
    #     reconstructed = prediction + decompressed_delta
        
    #     # Find fail values where reconstructed exceeds original error bound
    #     fail_mask = np.abs(data - reconstructed) > error_bound * ratio
    #     fail_idx = np.flatnonzero(fail_mask).astype(np.int32)
    #     fail_val = data.flat[fail_idx].astype(np.float32) if fail_idx.size else np.array([], dtype=np.float32)
        
    #     fail_bytes = self._encode_fail_values(fail_mask, fail_idx, fail_val)
        
    #     current_length = len(compressed_delta_base) + len(fail_bytes)
    #     best_length = current_length
    #     best_error_bound = rmse
    #     best_compressed = compressed_delta_base
    #     best_fail_bytes = fail_bytes
        
    #     # Binary search: try halving error bound
    #     test_eb = rmse / 2
    #     while test_eb > 0:
    #         const_eb = np.full_like(delta, test_eb)
    #         comp_delta = self.compressor.compress(delta, const_eb, ratio=ratio)
    #         dec_delta = self.compressor.decompress(comp_delta)
    #         recon = prediction + dec_delta
            
    #         fmask = np.abs(data - recon) > error_bound * ratio
    #         fidx = np.flatnonzero(fmask).astype(np.int32)
    #         fval = data.flat[fidx].astype(np.float32) if fidx.size else np.array([], dtype=np.float32)
            
    #         fbytes = self._encode_fail_values(fmask, fidx, fval)
            
    #         test_length = len(comp_delta) + len(fbytes)
    #         if test_length < best_length:
    #             best_length = test_length
    #             best_error_bound = test_eb
    #             best_compressed = comp_delta
    #             best_fail_bytes = fbytes
    #             test_eb /= 2
    #         else:
    #             break
        
    #     # Binary search: try doubling error bound
    #     test_eb = rmse * 2
    #     for _ in range(10):  # Limit iterations
    #         const_eb = np.full_like(delta, test_eb)
    #         comp_delta = self.compressor.compress(delta, const_eb, ratio=ratio)
    #         dec_delta = self.compressor.decompress(comp_delta)
    #         recon = prediction + dec_delta
            
    #         fmask = np.abs(data - recon) > error_bound * ratio
    #         fidx = np.flatnonzero(fmask).astype(np.int32)
    #         fval = data.flat[fidx].astype(np.float32) if fidx.size else np.array([], dtype=np.float32)
            
    #         fbytes = self._encode_fail_values(fmask, fidx, fval)
            
    #         test_length = len(comp_delta) + len(fbytes)
    #         if test_length < best_length:
    #             best_length = test_length
    #             best_error_bound = test_eb
    #             best_compressed = comp_delta
    #             best_fail_bytes = fbytes
    #             test_eb *= 2
    #         else:
    #             break
        
    #     print(f"    Binary search result: eb={best_error_bound:.4f}, size={best_length:,} bytes")
        
    #     return best_compressed, best_fail_bytes, best_error_bound

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
        start_time = time.time()
        compressed_dict = {}
        pressure_values = [float(p) for p in pressure_levels]
        total_bytes = 0
        
        for i, pressure_level in enumerate(pressure_levels):
            if pressure_level not in data_dict:
                continue
                
            data = data_dict[pressure_level]
            error_bound = error_bound_dict[pressure_level]
            total_bytes += data.nbytes
            
            if i == 0:
                # Compress first level normally (reference level)
                compressed = self.compressor.compress(data, error_bound, ratio=ratio)
                compressed_dict[pressure_level] = {'method': 'direct', 'data': compressed}
                cr = data.nbytes / len(compressed)
                compressed_direct_data = pickle.dumps(compressed)
                print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(compressed):,} bytes {data.nbytes/len(compressed_direct_data):.2f}x) [DIRECT]")
            else:
                # Compare three approaches: fail-only, delta+fail, and direct
                prev_pressure_level = pressure_levels[i-1]
                
                prev_data = data_dict[prev_pressure_level]
                prediction = self.compute_prediction(prev_data, pressure_values[i], pressure_values[i-1])
                delta = data - prediction
                
                # Option 1: Encode fail values only (prediction without compression)
                # fail_mask_only = np.abs(data - prediction) > error_bound * ratio
                # fail_idx_only = np.flatnonzero(fail_mask_only).astype(np.int32)
                # fail_val_only = data.flat[fail_idx_only].astype(np.float32) if fail_idx_only.size else np.array([], dtype=np.float32)
                # fail_bytes_only = self._encode_fail_values(fail_mask_only, fail_idx_only, fail_val_only)
                
                # Option 2: Compress delta directly + fail values
                compressed_delta_direct = self.compressor.compress(delta, error_bound, ratio=ratio)
                decompressed_delta_direct = self.compressor.decompress(compressed_delta_direct)
                reconstructed_direct = prediction + decompressed_delta_direct
                
                fail_mask_delta = np.abs(data - reconstructed_direct) > error_bound * ratio
                fail_idx_delta = np.flatnonzero(fail_mask_delta).astype(np.int32)
                fail_val_delta = data.flat[fail_idx_delta].astype(np.float32) 
                if fail_idx_delta.size:
                    fail_bytes_delta = self._encode_fail_values(fail_mask_delta, fail_idx_delta, fail_val_delta)
                    delta_with_fail_data = {'method': 'delta', 'delta': compressed_delta_direct, 'fail': fail_bytes_delta}
                else:
                    delta_with_fail_data = {'method': 'delta', 'delta': compressed_delta_direct}

                compressed_dict[pressure_level] = delta_with_fail_data
                compressed_delta_with_fail = pickle.dumps(delta_with_fail_data) 
                cr = data.nbytes / len(compressed_delta_with_fail)
                print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(compressed_delta_with_fail):,} bytes {data.nbytes/len(compressed_delta_with_fail):.2f}x) [DELTA]")

                # Option 3: Direct compression
                # compressed_direct = self.compressor.compress(data, error_bound, ratio=ratio)
                
                # # Compare all three and select shortest
                # sizes = {
                #     'fail-only': len(fail_bytes_only),
                #     'delta+fail': len(compressed_delta_with_fail),
                #     'direct': len(compressed_direct)
                # }
                
                # print(f"Comparing: fail-only={sizes['fail-only']:,}, delta+fail={sizes['delta+fail']:,}, direct={sizes['direct']:,}")
                
                # best_method = min(sizes, key=sizes.get)
                
                # if best_method == 'fail-only':
                #     # Pack fail-only data
                #     fail_only_data = pickle.dumps({'fail': fail_bytes_only})
                #     compressed_dict[pressure_level] = b'DELTA' + fail_only_data
                #     cr = data.nbytes / len(fail_only_data)
                #     print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {len(fail_only_data):,} bytes) [DELTA-FAILONLY]")
                # elif best_method == 'delta+fail':
                #     compressed_dict[pressure_level] = b'DELTA' + compressed_delta_with_fail
                #     cr = data.nbytes / sizes['delta+fail']
                #     print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {sizes['delta+fail']:,} bytes) [DELTA]")
                # else:
                #     compressed_dict[pressure_level] = b'DIRECT' + compressed_direct
                #     cr = data.nbytes / sizes['direct']
                #     print(f"{pressure_level:>4} hPa: {cr:6.2f}x ({data.nbytes:,} -> {sizes['direct']:,} bytes) [DIRECT]")
        
        elapsed_time = time.time() - start_time
        throughput = total_bytes / elapsed_time / (1024**2)  # MB/s
        print(f"\nCompression completed in {elapsed_time:.2f}s, throughput: {throughput:.2f} MB/s")

        return pickle.dumps(compressed_dict)

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
        compressed_dict = pickle.loads(compressed_dict)
        decompressed_dict = {}
        pressure_values = [float(p) for p in pressure_levels]
        
        for i, pressure_level in enumerate(pressure_levels):
            if pressure_level not in compressed_dict:
                continue
                
            compressed = compressed_dict[pressure_level]
            
            if compressed['method'] == 'direct':
                # Direct decompression
                decompressed_dict[pressure_level] = self.compressor.decompress(compressed['data'])
            elif compressed['method'] == 'delta':
                # Delta decompression
                prev_pressure_level = pressure_levels[i-1]
                
                prev_data = decompressed_dict[prev_pressure_level]
                prediction = self.compute_prediction(prev_data, pressure_values[i], pressure_values[i-1])
                delta = self.compressor.decompress(compressed['delta'])
                reconstructed = prediction + delta
                
                # Restore fail values if present
                if 'fail' in compressed:
                    fail_idx, fail_val = self._decode_fail_values(compressed['fail'])
                    reconstructed.flat[fail_idx] = fail_val
                
                decompressed_dict[pressure_level] = reconstructed
                
        return decompressed_dict
