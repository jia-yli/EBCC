import os
current_folder = os.path.dirname(os.path.realpath(__file__))
os.environ["HDF5_PLUGIN_PATH"] = os.path.join(current_folder, '../src/build/lib')
os.environ["EBCC_LOG_LEVEL"] = 3

import time
import h5py
import xarray as xr
import numpy as np
from ebcc_wrapper import EBCC_Filter

def convert_nc_to_hdf5(nc_file, hdf5_file):
  """
  Convert a NetCDF (.nc) file to HDF5 (.h5) format.

  Parameters:
    nc_file (str): Path to the input NetCDF file.
    hdf5_file (str): Path to the output HDF5 file.
  """
  # Open the NetCDF file
  dataset = xr.open_dataset(nc_file)

  # Create an HDF5 file
  with h5py.File(hdf5_file, 'w') as hdf5_f:
    for var_name, da in dataset.data_vars.items():
      data = da.values  # Convert xarray DataArray to NumPy array
      hdf5_f.create_dataset(var_name, data=data)
    
    # Save the dimensions as attributes
    for dim_name, dim_value in dataset.sizes.items():
      hdf5_f.attrs[dim_name] = dim_value

    # Save global attributes
    for attr_name, attr_value in dataset.attrs.items():
      hdf5_f.attrs[attr_name] = str(attr_value)  # Convert to string to store in HDF5

def compress_hdf5_ebcc_uniform(input_hdf5, output_hdf5, ebcc_base_compression_ratio, ebcc_max_error):
  # compression and compression time
  with h5py.File(input_hdf5, 'r') as hdf5_in:
    compression_start_time = time.time()
    with h5py.File(output_hdf5, 'w') as hdf5_out:
      for var_name in hdf5_in.keys():
        data = np.array(hdf5_in[var_name])  # Read dataset
        ebcc_filter = EBCC_Filter(
          base_cr=ebcc_base_compression_ratio, # base compression ratio
          height=data.shape[-2],  # height of each 2D data chunk
          width=data.shape[-1],  # width of each 2D data chunk
          data_dim=len(data.shape), # data dimension, required to specify the HDF5 chunk shape
          residual_opt=("max_error_target", ebcc_max_error),
          filter_path=os.path.join(current_folder, '../src')) # directory to the compiled HDF5 filter plugin
        hdf5_out.create_dataset(var_name, data=data, **ebcc_filter)

      # Copy attributes
      for attr_name, attr_value in hdf5_in.attrs.items():
        hdf5_out.attrs[attr_name] = attr_value
    compression_end_time = time.time()
    compression_time = compression_end_time - compression_start_time

  # decompreession time
  decompression_start_time = time.time()
  with h5py.File(output_hdf5, 'r') as hdf5_out:
    for dataset_name in hdf5_out.keys():
      data = np.array(hdf5_out[dataset_name])
  decompression_end_time = time.time()
  decompression_time = decompression_end_time - decompression_start_time
  return compression_time, decompression_time


output_path = f'/capstor/scratch/cscs/ljiayong/workspace/EBCC'
variable = total_precipitation

ebcc_base_compression_ratio = 1000
ebcc_max_error = 1

input_hdf5_file_path = os.path.join(output_path, f'{variable}.hdf5')
output_hdf5_file_path = os.path.join(output_path, f'{variable}_compressed_ebcc_uniform_test.hdf5')
compression_time, decompression_time = compress_hdf5_ebcc_uniform(input_hdf5_file_path, output_hdf5_file_path, ebcc_base_compression_ratio, ebcc_max_error)

print(f"Compression Time: {compression_time} s, Decompression Time: {decompression_time} s")

input_size = os.path.getsize(input_hdf5_file_path)
output_size = os.path.getsize(output_hdf5_file_path)

print(f"File Sizes:\nInput: {input_size/1024/1024/1024:.4f} GiB\nOutput: {output_size/1024/1024/1024:.4f} GiB\n")
