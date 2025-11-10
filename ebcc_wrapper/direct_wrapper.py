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
                result = {'data': compressed_bytes, 'fail': fail_info}
            else:
                result = {'data': compressed_bytes}
        
        return pickle.dumps(result)
    
    def decompress(self, bitstream: bytes) -> np.ndarray:
        """
        Decompress bitstream back to original data.
        
        Args:
            bitstream: Compressed bitstream
            
        Returns:
            Decompressed data array (without error bounds)
        """
        # Check for failing values marker
        # The bitstream may be either:
        # 1) a pickled dict: {'data': hdf5_bytes, 'fail': pickle.dumps({...})?}
        # 2) legacy raw-hdf5-bytes + b'EBCCFAIL' + pickle.dumps(fail_info)
        hdf5_bytes = None
        fail_info = None

        result = pickle.loads(bitstream)
        hdf5_bytes = result['data']
        fail_info = result.get('fail', None)

        # Read HDF5 bytes from a temporary file
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(hdf5_bytes)
            temp_file.seek(0)
            with h5py.File(temp_file.name, 'r') as hdf5_file:
                decompressed_data = np.array(hdf5_file['data'])[..., 0, :, :]

        # If we have fail information, decompress and apply it
        if fail_info:
            fail_info = pickle.loads(fail_info)

            if 'mask' in fail_info:
                # mask was stored as packbits(bytes) then zlib-compressed
                mask_bytes = zlib.decompress(fail_info['mask'])
                mask_bits = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))
                # trim to the flattened data size
                mask_bits = mask_bits[:decompressed_data.size]
                fail_idx = np.flatnonzero(mask_bits)
                fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)
            else:
                fail_idx = np.frombuffer(zlib.decompress(fail_info['idx']), dtype=np.int32)
                fail_val = np.frombuffer(zlib.decompress(fail_info['val']), dtype=np.float32)

            # Only assign if we actually have indices and values
            if fail_idx is not None and fail_val is not None and fail_idx.size and fail_val.size:
                decompressed_data.flat[fail_idx] = fail_val

        return decompressed_data