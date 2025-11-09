"""
Direct Python wrapper for EBCC compression that returns compressed bitstreams
instead of requiring HDF5 files.
"""

import os
import numpy as np
import h5py
import tempfile
import zlib
import pickle
from typing import Union
from .filter_wrapper import EBCC_Filter


class EBCCDirectWrapper:
    """
    Direct wrapper for EBCC compression that returns compressed bitstreams
    instead of requiring HDF5 files.
    """
    
    def __init__(self, filter_path: str = None):
        """
        Initialize the EBCC wrapper.
        
        Args:
            filter_path: Path to the compiled HDF5 filter plugin directory
        """
        if filter_path is None:
            # Default to the src directory relative to this file
            current_folder = os.path.dirname(os.path.realpath(__file__))
            filter_path = os.path.join(current_folder, '../src')
        
        self.filter_path = filter_path
        
        # Set HDF5 plugin path
        os.environ["HDF5_PLUGIN_PATH"] = os.path.join(filter_path, 'build/lib')
    
    def compress(self, data: np.ndarray, error_bound: Union[np.ndarray, float], 
                 ratio: float = 1.0) -> bytes:
        """
        Compress data with pointwise error bounds.
        
        Args:
            data: Input data array to compress
            error_bound: Error bound array (same shape as data) or scalar value
            ratio: Base compression ratio for EBCC
            
        Returns:
            Compressed bitstream as bytes
        """
        # Ensure data is numpy array
        data = np.asarray(data, dtype=np.float32)
        error_bound_array = np.asarray(error_bound, dtype=np.float32)
        
        # Stack data and error bound for pointwise compression
        # EBCC expects data and error bound as separate channels
        combined_data = np.stack([data, error_bound_array], axis=-3)
        
        # Create EBCC filter for pointwise error
        ebcc_filter = EBCC_Filter(
            base_cr=10,
            height=combined_data.shape[-2],
            width=combined_data.shape[-1],
            data_dim=len(combined_data.shape),
            residual_opt=("pointwise_max_error", ratio),  # Use ratio of 1.0 since we provide explicit bounds
            filter_path=self.filter_path
        )
        
        # Use temporary file to get compressed data; reuse it to read reconstructed data
        with tempfile.NamedTemporaryFile() as temp_file:
            with h5py.File(temp_file.name, 'w') as hdf5_file:
                hdf5_file.create_dataset('data', data=combined_data, **ebcc_filter)
            temp_file.seek(0)
            compressed_bytes = temp_file.read()
            
            # Read back reconstruction and append failing values
            with h5py.File(temp_file.name, 'r') as hf:
                data_hat = np.array(hf['data'])[..., 0, :, :]
            
            fail_mask = np.abs(data - data_hat) > error_bound_array * ratio
            fail_idx = np.flatnonzero(fail_mask).astype(np.int32)
            if fail_idx.size:
                fail_val = data.flat[fail_idx].astype(np.float32)
                cmask = zlib.compress(np.packbits(fail_mask.ravel()).tobytes(), level=6)
                cidx = zlib.compress(fail_idx.tobytes(), level=6)
                cval = zlib.compress(fail_val.tobytes(), level=6)
                if len(cmask) <= len(cidx):
                    fail_info = pickle.dumps({'mask': cmask, 'val': cval})
                else:
                    fail_info = pickle.dumps({'idx': cidx, 'val': cval})
                compressed_bytes += b'EBCCFAIL' + fail_info
        
        return compressed_bytes
    
    def decompress(self, bitstream: bytes) -> np.ndarray:
        """
        Decompress bitstream back to original data.
        
        Args:
            bitstream: Compressed bitstream
            
        Returns:
            Decompressed data array (without error bounds)
        """
        # Check for failing values marker
        if b'EBCCFAIL' in bitstream:
            pos = bitstream.index(b'EBCCFAIL')
            hdf5_bytes = bitstream[:pos]
            fail_info = pickle.loads(bitstream[pos + 8:])
            if 'mask' in fail_info:
                fail_mask = np.unpackbits(np.frombuffer(zlib.decompress(fail_info['mask']), dtype=np.uint8))
                fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
            else:
                fail_idx = np.frombuffer(zlib.decompress(fail_info['idx']), dtype=np.int32)
                fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
        else:
            hdf5_bytes = bitstream
            fail_idx = fail_val = None
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(hdf5_bytes)
            temp_file.seek(0)
            with h5py.File(temp_file.name, 'r') as hdf5_file:
                decompressed_data = np.array(hdf5_file['data'])[..., 0, :, :]
        
        if fail_val is not None:
            if 'fail_mask' in locals():
                fail_idx = np.flatnonzero(fail_mask[:decompressed_data.size])
            decompressed_data.flat[fail_idx] = fail_val
        
        return decompressed_data