import pickle
import numpy as np
from .image_compression import ErrorBoundedJP2KCodec


class DeltaErrorBoundedJP2KCodec:
    """
    Delta-based compression for [N, H, W] shaped data.
    
    Compression strategy:
    1. Compress first slice [0, H, W] with error_bound[0, H, W] using golden section search
    2. For each subsequent slice t:
       - Compute delta: current[t] - decompressed[t-1]
       - Compress delta with error_bound[t, H, W] using golden section search
    
    Each slice is independently optimized for best compression ratio.
    """
    
    def __init__(self):
        self.base_codec = ErrorBoundedJP2KCodec()
    
    def compress(self, data, error_bound, key_fail_u16='3'):
        assert data.ndim == 3, "Data must be 3D [N, H, W]"
        assert data.shape == error_bound.shape, "Data and error_bound must have same shape"
        assert data.dtype == np.float32, "Data must be float32"
        
        N, H, W = data.shape
        
        compressed_slices = []
        info = {
            "cratios": [],
            "compression_ratios_delta": [],
            "compression_ratios_base": [],
            "compression_ratios_original_delta": [],
            "delta_range_ratios": [],
            "original_delta_range_ratios": [],
            "error_bound_range_ratios": [],
        }
        
        # Compress first slice with golden section search
        first_slice = data[0:1, :, :]  # Keep 3D shape [1, H, W]
        first_eb = error_bound[0:1, :, :]
        
        (first_blob, first_info), first_cratio = self.base_codec.golden_section_search_best_compression(
            first_slice, first_eb, key_fail_u16=key_fail_u16
        )
        compressed_slices.append(('base', first_blob))
        # info
        info["cratios"].append(first_cratio)
        info["compression_ratios_delta"].append(first_slice.nbytes / len(first_blob))
        info["compression_ratios_base"].append(first_slice.nbytes / len(first_blob))
        info["compression_ratios_original_delta"].append(first_slice.nbytes / len(first_blob))
        info["delta_range_ratios"].append(1.0)  # No delta for first slice
        info["original_delta_range_ratios"].append(1.0)  # No delta for first slice
        error_bound_range = float(np.mean(first_eb))
        base_range = float(np.max(first_slice) - np.min(first_slice))
        info["error_bound_range_ratios"].append(error_bound_range / base_range)
        
        # Decompress first slice to use as reference
        prev_decompressed = self.base_codec.decompress(first_blob) # [1, H, W]
        
        # Compress remaining slices as deltas
        for t in range(1, N):
            current_slice = data[t:t+1]  # [H, W]
            current_eb = error_bound[t:t+1]  # [H, W]

            # Compute delta
            delta = current_slice - prev_decompressed  # [H, W]
            
            (delta_blob, delta_info), delta_cratio = self.base_codec.golden_section_search_best_compression(
                delta, current_eb, key_fail_u16=key_fail_u16
            )
            compressed_slices.append(('delta', delta_blob))

            # other info
            (base_blob, _), _ = self.base_codec.golden_section_search_best_compression(
                current_slice, current_eb, key_fail_u16=key_fail_u16
            )
            original_delta = data[t:t+1] - data[t-1:t] # [1, H, W] 
            (original_delta_blob, _), _ = self.base_codec.golden_section_search_best_compression(
                original_delta, current_eb, key_fail_u16=key_fail_u16
            )

            info["cratios"].append(delta_cratio)
            info["compression_ratios_delta"].append(current_slice.nbytes / len(delta_blob))
            info["compression_ratios_base"].append(current_slice.nbytes / len(base_blob))
            info["compression_ratios_original_delta"].append(current_slice.nbytes / len(original_delta_blob))
            delta_range = float(np.max(delta) - np.min(delta))
            original_delta_range = float(np.max(original_delta) - np.min(original_delta))
            base_range = float(np.max(current_slice) - np.min(current_slice))
            error_bound_range = float(np.mean(current_eb))
            info["delta_range_ratios"].append(delta_range / base_range)
            info["original_delta_range_ratios"].append(original_delta_range / base_range)
            info["error_bound_range_ratios"].append(error_bound_range / base_range)
            
            # Decompress delta and reconstruct current slice for next iteration
            delta_hat = self.base_codec.decompress(delta_blob)  # [1, H, W]
            prev_decompressed = prev_decompressed + delta_hat

        # Package payload
        payload = {
            'slices': compressed_slices,
        }
        
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), info
    
    def decompress(self, blob):
        """
        Decompress data compressed with delta encoding.
        
        Args:
            blob: pickled compressed payload
        
        Returns:
            data: np.ndarray of shape [N, H, W], dtype float32
        """
        payload = pickle.loads(blob)
        compressed_slices = payload['slices']
        
        N = len(compressed_slices)
        decompressed_slices = []
        
        # Decompress first slice
        slice_type, first_blob = compressed_slices[0]
        assert slice_type == 'base', "First slice must be base type"
        first_slice = self.base_codec.decompress(first_blob) # [1, H, W]
        decompressed_slices.append(first_slice)
        
        # Decompress and accumulate deltas
        for t in range(1, N):
            slice_type, delta_blob = compressed_slices[t]
            
            if slice_type == "delta":
                delta_hat = self.base_codec.decompress(delta_blob)  # [1, H, W]
                current_slice = decompressed_slices[t-1] + delta_hat
                decompressed_slices.append(current_slice)
            else:
                current_slice = self.base_codec.decompress(delta_blob)  # [1, H, W]
                decompressed_slices.append(current_slice)

        return np.concatenate(decompressed_slices, axis=0)