import os
current_folder = os.path.dirname(os.path.realpath(__file__))
os.environ["HDF5_PLUGIN_PATH"] = os.path.join(current_folder, '../src/build/lib')

import h5py
import numpy as np
from ebcc_wrapper import EBCC_Filter

def compress_hdf5(input_hdf5, output_hdf5):
  """
  Reads an HDF5 file and saves it to a new file with a different compression method.

  Parameters:
    input_hdf5 (str): Path to the original HDF5 file.
    output_hdf5 (str): Path to the new HDF5 file.
    compression (str): Compression type ("gzip", "lzf", "szip").
  """
  with h5py.File(input_hdf5, 'r') as hdf5_in, h5py.File(output_hdf5, 'w') as hdf5_out:
    for var_name in hdf5_in.keys():
      data = np.array(hdf5_in[var_name])  # Read dataset

      ebcc_filter = EBCC_Filter(
        base_cr=100, # base compression ratio
        height=data.shape[-2],  # height of each 2D data chunk
        width=data.shape[-1],  # width of each 2D data chunk
        data_dim=len(data.shape), # data dimension, required to specify the HDF5 chunk shape
        residual_opt=("max_error_target", 1),
        filter_path=os.path.join(current_folder, 'src')) # directory to the compiled HDF5 filter plugin

      print(dict(ebcc_filter))

      # Save dataset with new compression
      # hdf5_out.create_dataset(var_name, data=data, **ebcc_filter)
      hdf5_out.create_dataset(var_name, data=data, compression='gzip')

    # Copy attributes
    for attr_name, attr_value in hdf5_in.attrs.items():
      hdf5_out.attrs[attr_name] = attr_value

def compute_compression_error(original_hdf5, compressed_hdf5):
  """
  Computes the point-wise error and maximum error between the original HDF5 file and the lossy-compressed HDF5 file.
  
  Parameters:
    original_hdf5 (str): Path to the original HDF5 file.
    compressed_hdf5 (str): Path to the lossy-compressed HDF5 file.
  """

  max_error = {}
  # Open HDF5 files
  with h5py.File(original_hdf5, 'r') as orig_file, h5py.File(compressed_hdf5, 'r') as comp_file:
    for dataset_name in orig_file.keys():
      orig_data = np.array(orig_file[dataset_name])
      comp_data = np.array(comp_file[dataset_name])

      # Compute absolute difference (point-wise error)
      error = np.abs(orig_data - comp_data)

      # Max absolute error
      dataset_max_error = np.max(error)
      max_error[dataset_name] = dataset_max_error

  for dataset_name in max_error.keys():
    print(f"Dataset: {dataset_name} | Max Error: {max_error[dataset_name]:.6f}")



input_hdf5_file_path = "/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/2m_temperature.hdf5"  # Replace with your desired output file path
output_hdf5_file_path = "/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/2m_temperature_ebcc.hdf5"  # Replace with your desired output file path
compress_hdf5(input_hdf5_file_path, output_hdf5_file_path)

input_size = os.path.getsize(input_hdf5_file_path)
output_size = os.path.getsize(output_hdf5_file_path)

print(f"File Sizes:\nInput: {input_size/1024/1024/1024:.4f} GiB\nOutput: {output_size/1024/1024/1024:.4f} GiB\n")

compute_compression_error(input_hdf5_file_path, output_hdf5_file_path)