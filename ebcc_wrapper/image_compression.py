import os
import tempfile
import pickle

import numpy as np
import glymur
import blosc

class ErrorBoundedJP2KCodec:
    """
    Error-bounded JPEG2000 codec using glymur with global min-max normalization.

    - compress(data, error_bound, cratio)
      Steps:
        1) Cast/validate; shapes must match (N,H,W)
        2) Min-max scale 'data' to uint16; convert (error_bound*0.99) to the same uint16 units
        3) Encode uint16 by JPEG2000 with given cratio
        4) Decode, check absolute error in uint16 against bound
        5) Handle failures
        6) Final Checks
        7) Return pickled payload

    - decompress(blob)
      Steps:
        - Decode jpeg2000 -> uint16
        - Restore precise values via modular functions
        - Inverse-scale to original range -> float32 (N,H,W)
        - Apply failed values
        - Return float32 array
    """

    def __init__(self):
        self._u16_max = np.float32(65535.0)

    # ========= Min-max scaling =========

    def _minmax_scale_to_u16(self, x):
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max > x_min:
            scale = self._u16_max / np.float32(x_max - x_min)
            x_u16 = np.clip(np.round((x - x_min) * scale), 0, self._u16_max).astype(np.uint16)
        else:
            x_u16 = np.zeros_like(x, dtype=np.uint16)
        return x_u16, x_min, x_max

    def _minmax_scale_from_u16(self, x_u16, x_min, x_max):
        if x_max > x_min:
            scale_inv = np.float32(x_max - x_min) / self._u16_max
            x = (x_u16.astype(np.float32) * scale_inv) + np.float32(x_min)
        else:
            x = np.full(x_u16.shape, np.float32(x_min), dtype=np.float32)
        return x

    def _scale_err_to_u16(self, err, x_min, x_max):
        err = err * 0.99  # slight safety margin
        if x_max > x_min:
            scale = self._u16_max / np.float32(x_max - x_min)
            err_u16 = np.clip(np.floor(err * scale), 0, self._u16_max).astype(np.uint16)
        else:
            err_u16 = np.zeros_like(err, dtype=np.uint16)
        return err_u16

    # ========= JPEG2000 I/O =========

    def _encode_jp2k_u16(self, x_u16, cratio):
        """Encode uint16 (N,H,W) by treating N as components -> JP2K bytes."""
        if x_u16.dtype != np.uint16 or x_u16.ndim != 3:
            raise ValueError(f"Expected uint16 (N,H,W), got {x_u16.dtype} {x_u16.shape}")

        x_hw_c = np.transpose(x_u16, (1, 2, 0))  # (H, W, N)

        fd, fname = tempfile.mkstemp(suffix=".jp2")
        os.close(fd)
        os.remove(fname) # glymur needs a non-existing file

        glymur.Jp2k(fname, data=x_hw_c, cratios=[cratio])
        with open(fname, "rb") as f:
            bitstream = f.read()

        return bitstream

    def _decode_jp2k_u16(self, bitstream: bytes) -> np.ndarray:
        """Decode JP2K bytes -> uint16 (N,H,W)."""
        fd, fname = tempfile.mkstemp(suffix=".jp2")
        os.close(fd)
        with open(fname, "wb") as f:
            f.write(bitstream)
        jp2 = glymur.Jp2k(fname)

        arr_u16 = np.asarray(jp2[:], dtype=np.uint16)  # (H,W,N)
        return np.transpose(arr_u16, (2, 0, 1))  # (N,H,W)

    # ========= Failure handling =========

    # if eb >= 2^N, then the N LSB can be changed arbitrarily without violating the error bound.
    # Because max error from change = 2^N - 1 < eb.
    # Thus, we can quantize the residual to multiples of S = 2^k, where S is the largest power-of-two such that S*2 > eb.
    # But we do RNE instead, so the condition is S/2 <= eb.
    def _extract_failures_u16(self, data_u16, data_u16_jp2k_hat, err_u16, key_fail_u16):
        # dtype/shape checks
        assert (data_u16.dtype == np.uint16) and (data_u16_jp2k_hat.dtype == np.uint16) and (err_u16.dtype == np.uint16)
        assert data_u16.shape == data_u16_jp2k_hat.shape == err_u16.shape

        # get residual
        residual = data_u16.astype(np.int32) - data_u16_jp2k_hat.astype(np.int32) # [-65535, 65535]

        # quantize residual
        k = np.floor(np.log2(np.maximum(1, err_u16.astype(np.int32) * 2))).astype(np.int32)
        S = (1 << k).astype(np.int32)

        # quantize to nearest multiple of S (RNE)
        residual_q = np.rint(residual.astype(np.float32) / S).astype(np.int32) * S # ties-to-even

        # wrap-around to uint16
        residual_q_w = residual_q.astype(np.uint16) # map to [-32768, 32767]

        inside = np.abs(residual) <= err_u16.astype(np.int32)
        residual_q_w[inside] = 0

        data_u16_hat = (data_u16_jp2k_hat.astype(np.uint32) + residual_q_w.astype(np.uint32)).astype(np.uint16)

        bitstream_fail_u16_candidates = {}
        fail_info_u16 = {}
        fail_count_jp2k_hat = int(np.count_nonzero(np.abs(data_u16.astype(np.int32) - data_u16_jp2k_hat.astype(np.int32)) > err_u16.astype(np.int32)))

        # full residual
        if (key_fail_u16 is None) or (key_fail_u16 == '1'):
            residual_q_w_bitstream = blosc.compress(residual_q_w.tobytes(), typesize=2, cname="zstd", clevel=9, shuffle=blosc.BITSHUFFLE)
            bitstream_fail_u16_candidates['1'] = pickle.dumps(("1", residual_q_w_bitstream), protocol=pickle.HIGHEST_PROTOCOL)
            fail_info_u16['compression_ratio_fail_u16_1'] = fail_count_jp2k_hat * 2 / (len(residual_q_w_bitstream) + 1e-8)

        # sparse residuals
        if (key_fail_u16 is None) or (key_fail_u16 in ['2', '3', '4']):
            non_zero_mask = residual_q_w != 0
            non_zero_val = residual_q_w[non_zero_mask].ravel()
            non_zero_val_bitstream = blosc.compress(non_zero_val.tobytes(), typesize=2, cname="zstd", clevel=9, shuffle=blosc.BITSHUFFLE)

            if (key_fail_u16 is None) or (key_fail_u16 == '2'):
                # idx + val
                non_zero_idx = np.flatnonzero(non_zero_mask.ravel()).astype(np.int32)
                non_zero_idx_bitstream = blosc.compress(non_zero_idx, typesize=4, cname="zstd", clevel=9)
                bitstream_fail_u16_candidates['2'] = pickle.dumps(("2", non_zero_idx_bitstream, non_zero_val_bitstream), protocol=pickle.HIGHEST_PROTOCOL)
                fail_info_u16['compression_ratio_fail_u16_2'] = fail_count_jp2k_hat * 2 / (len(non_zero_idx_bitstream) + len(non_zero_val_bitstream) + 1e-8)
            if (key_fail_u16 is None) or (key_fail_u16 == '3'):
                # bitmask + val
                non_zero_bitmask = np.packbits(non_zero_mask.ravel())
                non_zero_bitmask_bitstream = blosc.compress(non_zero_bitmask.tobytes(), typesize=1, cname="zstd", clevel=9)
                bitstream_fail_u16_candidates['3'] = pickle.dumps(("3", non_zero_bitmask_bitstream, non_zero_val_bitstream), protocol=pickle.HIGHEST_PROTOCOL)
                fail_info_u16['compression_ratio_fail_u16_3'] = fail_count_jp2k_hat * 2 / (len(non_zero_bitmask_bitstream) + len(non_zero_val_bitstream) + 1e-8)
            if (key_fail_u16 is None) or (key_fail_u16 == '4'):
                # Use block id (uint16) + offset (uint16) + val
                non_zero_idx = np.flatnonzero(non_zero_mask.ravel()).astype(np.int32)
                block_size = 65536
                n_blocks = (non_zero_mask.size + block_size - 1) // block_size

                block_idx_count = np.bincount(non_zero_idx // block_size, minlength=n_blocks).astype(np.uint16)
                block_idx_offset = (non_zero_idx % block_size).astype(np.uint16)

                block_idx_count_bitstream = blosc.compress(block_idx_count, typesize=2, cname="zstd", clevel=9, shuffle=blosc.BITSHUFFLE)
                block_idx_offset_bitstream = blosc.compress(block_idx_offset, typesize=2, cname="zstd", clevel=9)

                bitstream_fail_u16_candidates['4'] = pickle.dumps(("4", block_idx_count_bitstream, block_idx_offset_bitstream, non_zero_val_bitstream), protocol=pickle.HIGHEST_PROTOCOL)
                fail_info_u16['compression_ratio_fail_u16_4'] = fail_count_jp2k_hat * 2 / (len(block_idx_count_bitstream) + len(block_idx_offset_bitstream) + len(non_zero_val_bitstream) + 1e-8)
                # print(fail_count_jp2k_hat * 2/len(block_idx_count_bitstream))
                # print(fail_count_jp2k_hat * 2/len(block_idx_offset_bitstream))
                # print(fail_count_jp2k_hat * 2/len(non_zero_val_bitstream))
        
        assert len(bitstream_fail_u16_candidates) > 0, "No failure handling methods generated."
        # choose best
        key_fail_u16, bitstream_fail_u16 = min(bitstream_fail_u16_candidates.items(), key=lambda kv: len(kv[1]))
        # import pdb;pdb.set_trace()
        assert (data_u16_hat == self._apply_failures_u16(data_u16_jp2k_hat, bitstream_fail_u16, data_u16.shape)).all()

        fail_count_hat = int(np.count_nonzero(np.abs(data_u16.astype(np.int32) - data_u16_hat.astype(np.int32)) > err_u16.astype(np.int32)))
        compression_ratio_fail_u16 = fail_count_jp2k_hat * 2 / (len(bitstream_fail_u16) + 1e-8)

        fail_info_u16.update({
            "fail_ratio_jp2k_hat": fail_count_jp2k_hat/data_u16.size,
            "fail_ratio_hat": fail_count_hat/data_u16.size,
            "key_fail_u16": key_fail_u16,
            "compression_ratio_fail_u16": compression_ratio_fail_u16,
        })

        return bitstream_fail_u16, data_u16_hat, fail_info_u16

    def _apply_failures_u16(self, data_u16_jp2k_hat, bitstream_fail_u16, shape):
        """
        Apply stored uint16 ring residuals to the JP2K reconstruction.
        """
        key_fail_u16, *payload = pickle.loads(bitstream_fail_u16)

        if key_fail_u16 == "1":
            # full residual
            (residual_q_w_bitstream,) = payload
            residual_q_w = np.frombuffer(blosc.decompress(residual_q_w_bitstream), dtype=np.uint16).reshape(shape)
        elif key_fail_u16 == "2":
            # sparse residuals via int32 mask
            non_zero_idx_bitstream, non_zero_val_bitstream = payload

            non_zero_idx = np.frombuffer(blosc.decompress(non_zero_idx_bitstream), dtype=np.int32)
            non_zero_val = np.frombuffer(blosc.decompress(non_zero_val_bitstream), dtype=np.uint16)

            residual_q_w = np.zeros(int(np.prod(shape)), dtype=np.uint16)
            residual_q_w[non_zero_idx] = non_zero_val
            residual_q_w = residual_q_w.reshape(shape)
        elif key_fail_u16 == "3":
            # sparse residuals via packed bitmask
            non_zero_bitmask_bitstream, non_zero_val_bitstream = payload

            non_zero_bitmask = np.frombuffer(blosc.decompress(non_zero_bitmask_bitstream), dtype=np.uint8)
            non_zero_val = np.frombuffer(blosc.decompress(non_zero_val_bitstream), dtype=np.uint16)

            mask = np.unpackbits(non_zero_bitmask)[:int(np.prod(shape))].astype(bool)

            residual_q_w = np.zeros(int(np.prod(shape)), dtype=np.uint16)
            residual_q_w[mask] = non_zero_val
            residual_q_w = residual_q_w.reshape(shape)
        elif key_fail_u16 == "4":
            block_idx_count_bitstream, block_idx_offset_bitstream, non_zero_val_bitstream = payload
            block_idx_count = np.frombuffer(blosc.decompress(block_idx_count_bitstream), dtype=np.uint16)
            block_idx_offset = np.frombuffer(blosc.decompress(block_idx_offset_bitstream), dtype=np.uint16)
            non_zero_val = np.frombuffer(blosc.decompress(non_zero_val_bitstream), dtype=np.uint16)

            block_size = 65536
            block_idx = np.repeat(np.arange(block_idx_count.size, dtype=np.int32), block_idx_count.astype(np.int32))
            non_zero_idx = block_idx * block_size + block_idx_offset.astype(np.int32)
            non_zero_val = np.frombuffer(blosc.decompress(non_zero_val_bitstream), dtype=np.uint16)

            residual_q_w = np.zeros(int(np.prod(shape)), dtype=np.uint16)
            residual_q_w[non_zero_idx] = non_zero_val
            residual_q_w = residual_q_w.reshape(shape)
        else:
            raise ValueError(f"Unknown key_fail_u16: {key_fail_u16}")

        data_u16_hat = (data_u16_jp2k_hat.astype(np.uint32) + residual_q_w.astype(np.uint32)).astype(np.uint16)

        return data_u16_hat

    def _extract_failures_fp32(self, data, data_fp32_hat, error_bound):
        """
        Final safety net in float32: if anything is still out-of-bound after u16 handling,
        store (flat) indices and original float32 values. Return compressed payload + info.
        """
        assert (data.dtype == np.float32) and (data_fp32_hat.dtype == np.float32) and (error_bound.dtype == np.float32)
        assert data.shape == data_fp32_hat.shape == error_bound.shape

        flat_data = data.ravel()
        flat_data_fp32_hat = data_fp32_hat.ravel()
        flat_error_bound = error_bound.ravel()

        mask = np.abs(flat_data - flat_data_fp32_hat) > flat_error_bound
        idx = np.flatnonzero(mask).astype(np.int32)
        vals = flat_data[idx].astype(np.float32)

        bitstream_fail_fp32 = pickle.dumps(
            (
                blosc.compress(idx.tobytes(), typesize=4, cname="zstd", clevel=9, shuffle=blosc.BITSHUFFLE), 
                blosc.compress(vals.tobytes(), typesize=4, cname="zstd", clevel=9, shuffle=blosc.BITSHUFFLE), 
            ), 
            protocol=pickle.HIGHEST_PROTOCOL
        )

        fail_count_fp32_hat = int(np.count_nonzero(np.abs(data - data_fp32_hat) > error_bound))
        compression_ratio = fail_count_fp32_hat * 4 / (len(bitstream_fail_fp32) + 1e-8)

        fail_info_fp32 = {
            "fail_count_fp32_hat": fail_count_fp32_hat,
            "fail_ratio_fp32_hat": fail_count_fp32_hat/data.size,
            "compression_ratio": compression_ratio,
        }
        return bitstream_fail_fp32, fail_info_fp32

    def _apply_failures_fp32(self, data_fp32_hat, bitstream_fail_fp32, shape):
        idx, vals = pickle.loads(bitstream_fail_fp32)
        idx = np.frombuffer(blosc.decompress(idx), dtype=np.int32)
        vals = np.frombuffer(blosc.decompress(vals), dtype=np.float32)

        data_fp32_hat = data_fp32_hat.reshape(-1)
        data_fp32_hat[idx] = vals.astype(np.float32)
        return data_fp32_hat.reshape(shape)

    # ========= Compression/Decompression =========

    def compress(self, data, error_bound, cratio, key_fail_u16='3'):
        assert (data.shape == error_bound.shape) and (data.ndim == 3), "data and error_bound must be float32 with shape (N,H,W)"
        assert (data.dtype == np.float32) and (error_bound.dtype == np.float32), "data and error_bound must be float32"

        data_u16, dmin, dmax = self._minmax_scale_to_u16(data)
        err_u16 = self._scale_err_to_u16(error_bound, dmin, dmax)

        bitstream_jp2k = self._encode_jp2k_u16(data_u16, cratio)
        data_u16_jp2k_hat = self._decode_jp2k_u16(bitstream_jp2k)

        bitstream_fail_u16, data_u16_hat, fail_info_u16 = self._extract_failures_u16(data_u16, data_u16_jp2k_hat, err_u16, key_fail_u16=key_fail_u16)
        print(f"U16 Failures: {fail_info_u16}")

        # final checks
        data_fp32_hat = self._minmax_scale_from_u16(data_u16_hat, dmin, dmax)
        bitstream_fail_fp32, fail_info_fp32 = self._extract_failures_fp32(data, data_fp32_hat, error_bound)
        print(f"FP32 Failures: {fail_info_fp32}")

        payload = {
            "min": dmin,
            "max": dmax,
            "shape": tuple(data.shape),
            "bs_jp2k": bitstream_jp2k,
            "bs_f_u16": bitstream_fail_u16,
            "bs_f_fp32": bitstream_fail_fp32,
        }
        total_size = len(bitstream_jp2k) + len(bitstream_fail_u16) + len(bitstream_fail_fp32)
        print(f"Final bistream: ratio of u16 failures = {len(bitstream_fail_u16) / total_size}, ratio of fp32 failures= {len(bitstream_fail_fp32) / total_size:}")
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def decompress(self, blob):
        payload = pickle.loads(blob)
        dmin, dmax = float(payload["min"]), float(payload["max"])
        shape = tuple(payload["shape"])
        bitstream_jp2k = payload["bs_jp2k"]
        bitstream_fail_u16 = payload["bs_f_u16"]
        bitstream_fail_fp32 = payload["bs_f_fp32"]

        data_u16_jp2k_hat = self._decode_jp2k_u16(bitstream_jp2k)
        assert data_u16_jp2k_hat.shape == shape

        data_u16_hat = self._apply_failures_u16(data_u16_jp2k_hat, bitstream_fail_u16, shape)
        data_fp32_hat = self._minmax_scale_from_u16(data_u16_hat, dmin, dmax)
        data_fp32_hat = self._apply_failures_fp32(data_fp32_hat, bitstream_fail_fp32, shape)

        assert (data_fp32_hat.shape == shape) and (data_fp32_hat.dtype == np.float32)

        return data_fp32_hat
