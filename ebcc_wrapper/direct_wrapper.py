"""
Direct Python wrapper for EBCC compression that returns compressed bitstreams
instead of requiring HDF5 files.
"""

import os
import numpy as np
import h5py
import tempfile
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
        
        # Use temporary file to get compressed data
        with tempfile.NamedTemporaryFile() as temp_file:
            with h5py.File(temp_file.name, 'w') as hdf5_file:
                hdf5_file.create_dataset('data', data=combined_data, **ebcc_filter)
            
            temp_file.seek(0)
            compressed_bytes = temp_file.read()
        
        return compressed_bytes
    
    def decompress(self, bitstream: bytes) -> np.ndarray:
        """
        Decompress bitstream back to original data.
        
        Args:
            bitstream: Compressed bitstream
            
        Returns:
            Decompressed data array (without error bounds)
        """
        # Use temporary file to decompress
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(bitstream)
            temp_file.seek(0)
            
            with h5py.File(temp_file.name, 'r') as hdf5_file:
                # Read the data (first channel is the actual data, second is error bounds)
                combined_data = np.array(hdf5_file['data'])
                
                decompressed_data = combined_data[..., 0, :, :]
        
        return decompressed_data