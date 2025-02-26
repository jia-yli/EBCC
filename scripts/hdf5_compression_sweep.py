import os
current_folder = os.path.dirname(os.path.realpath(__file__))
os.environ["HDF5_PLUGIN_PATH"] = os.path.join(current_folder, '../src/build/lib')

import time
import xarray as xr
import h5py
import numpy as np
import pandas as pd
from ebcc_wrapper import JP2SPWV_Filter
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt

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

def compress_hdf5_lossless(input_hdf5, output_hdf5, filter_kwargs):
  # compression and compression time
  with h5py.File(input_hdf5, 'r') as hdf5_in:
    compression_start_time = time.time()
    with h5py.File(output_hdf5, 'w') as hdf5_out:
      for var_name in hdf5_in.keys():
        data = np.array(hdf5_in[var_name])  # Read dataset
        hdf5_out.create_dataset(var_name, data=data, **filter_kwargs)

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

def compute_compression_error(original_hdf5, compressed_hdf5):
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
  
  return max_error

def run_lossless_compression(output_path, variable):
  '''
  Compare lossless compressors
  '''
  filter_kwargs_all = {
    'no-compression' : dict(),
    'gzip-level0' : dict(compression='gzip', compression_opts=0),
    'gzip-level4' : dict(compression='gzip', compression_opts=4),
    'gzip-level9' : dict(compression='gzip', compression_opts=9),
    'lzf' : dict(compression='lzf'),
    'szip' : dict(compression='szip'),
  }
  results = {
    'scheme' : list(filter_kwargs_all.keys()),
    'compression_ratio' : [],
    'max_error' : [],
    'compression_time' : [],
    'decompression_time' : [],
  }
  for scheme in results['scheme']:
    input_hdf5_file_path = os.path.join(output_path, f'{variable}.hdf5')  # Replace with your desired output file path
    output_hdf5_file_path = os.path.join(output_path, f'{variable}_lossless_compression.hdf5') # Replace with your desired output file path
    compression_time, decompression_time = compress_hdf5_lossless(input_hdf5_file_path, output_hdf5_file_path, filter_kwargs_all[scheme])
    input_size = os.path.getsize(input_hdf5_file_path)
    output_size = os.path.getsize(output_hdf5_file_path)
    compression_ratio = input_size/output_size
    max_error = compute_compression_error(input_hdf5_file_path, output_hdf5_file_path)
    assert len(list(max_error.keys())) == 1
    max_error = max_error[list(max_error.keys())[0]]
    results['compression_ratio'].append(compression_ratio)
    results['max_error'].append(max_error)
    results['compression_time'].append(compression_time)
    results['decompression_time'].append(decompression_time)
  return results

def compress_hdf5_ebcc_uniform(input_hdf5, output_hdf5, ebcc_base_compression_ratio, ebcc_max_error):
  # compression and compression time
  with h5py.File(input_hdf5, 'r') as hdf5_in:
    compression_start_time = time.time()
    with h5py.File(output_hdf5, 'w') as hdf5_out:
      for var_name in hdf5_in.keys():
        data = np.array(hdf5_in[var_name])  # Read dataset
        jp2spwv_filter = JP2SPWV_Filter(
          base_cr=ebcc_base_compression_ratio, # base compression ratio
          height=data.shape[-2],  # height of each 2D data chunk
          width=data.shape[-1],  # width of each 2D data chunk
          data_dim=len(data.shape), # data dimension, required to specify the HDF5 chunk shape
          residual_opt=("max_error_target", ebcc_max_error),
          filter_path=os.path.join(current_folder, '../src')) # directory to the compiled HDF5 filter plugin
        hdf5_out.create_dataset(var_name, data=data, **jp2spwv_filter)

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

def run_ebcc_uniform(output_path, variable, ebcc_base_compression_ratio, ebcc_max_error):
  '''
  EBCC compressor with uniform max error bound
  '''
  input_hdf5_file_path = os.path.join(output_path, f'{variable}.hdf5')
  output_hdf5_file_path = os.path.join(output_path, f'{variable}_compressed_ebcc_uniform_cr_{ebcc_base_compression_ratio}_err_{ebcc_max_error}.hdf5')
  compression_time, decompression_time = compress_hdf5_ebcc_uniform(input_hdf5_file_path, output_hdf5_file_path, ebcc_base_compression_ratio, ebcc_max_error)
  input_size = os.path.getsize(input_hdf5_file_path)
  output_size = os.path.getsize(output_hdf5_file_path)
  compression_ratio = input_size/output_size
  max_error = compute_compression_error(input_hdf5_file_path, output_hdf5_file_path)
  assert len(list(max_error.keys())) == 1
  max_error = max_error[list(max_error.keys())[0]]
  results = {
    'ebcc_base_compression_ratio' : ebcc_base_compression_ratio,
    'ebcc_max_error' : ebcc_max_error, 
    'compression_ratio' : compression_ratio,
    'max_error' : max_error,
    'compression_time' : compression_time,
    'decompression_time' : decompression_time,
  }
  return results

def plot_compression_error_dist(output_path, variable, ebcc_base_compression_ratio, ebcc_max_error):
  '''
  Plot EBCC Compressor Error Dist
  '''
  original_hdf5 = os.path.join(output_path, f'{variable}.hdf5')
  compressed_hdf5 = os.path.join(output_path, f'{variable}_compressed_ebcc_uniform_cr_{ebcc_base_compression_ratio}_err_{ebcc_max_error}.hdf5')
  with h5py.File(original_hdf5, 'r') as orig_file, h5py.File(compressed_hdf5, 'r') as comp_file:
    for dataset_name in orig_file.keys():
      orig_data = np.array(orig_file[dataset_name])
      comp_data = np.array(comp_file[dataset_name])

      # Compute absolute difference (point-wise error)
      error = np.abs(orig_data - comp_data)

      # Plot
      fig = plt.figure(figsize=(12, 6))
      plt.hist(error.flatten(), bins=40, alpha=0.7, edgecolor=None)
      plt.xlabel(f"{variable} Compression Error")
      plt.ylabel(f"Count")
      plt.title(f"EBCC {variable} Compression Error Dist\nBase Compression Ratio {ebcc_base_compression_ratio}, Max Error Bound {ebcc_max_error}")
      plt.grid(True)
      output_path = f"./results/{variable}_ebcc_uniform_compression_error_cr_{ebcc_base_compression_ratio}_err_{ebcc_max_error}.png"
      plt.savefig(output_path, dpi=500, bbox_inches="tight")
      plt.close()

if __name__ == '__main__':
  # variable_lst = [
  #   "10m_u_component_of_wind",
  #   "10m_v_component_of_wind",
  #   "2m_temperature",
  #   "total_precipitation"
  # ]
  variable_lst = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
  ]
  # global value
  for variable_idx in range(len(variable_lst)):
    variable = variable_lst[variable_idx]
    output_path = f'/capstor/scratch/cscs/ljiayong/workspace/EBCC'
    os.makedirs(output_path, exist_ok = True)

    # '''
    # Step 1: NetCDF to HDF5 without compression
    # '''
    # nc_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    # hdf5_file = os.path.join(output_path, f'{variable}.hdf5')
    # convert_nc_to_hdf5(nc_file, hdf5_file)

    # '''
    # Step 2: Run Lossless Compressors
    # '''
    # results = run_lossless_compression(output_path, variable)
    # results_df = pd.DataFrame(results)
    # results_df.to_csv(f'./results/{variable}_lossless_compression.csv', index=False)

    # '''
    # Step 3: Run EBCC with Uniform Error Bound
    # '''
    # ebcc_base_compression_ratio_lst = [1, 10, 100, 1000]
    # ebcc_max_error_lst = [0.01, 0.1, 1]

    # param_combinations = list(itertools.product([output_path], [variable], ebcc_base_compression_ratio_lst, ebcc_max_error_lst))
      
    # # MP
    # with mp.Pool(processes=32) as pool:
    #   results = pool.starmap(run_ebcc_uniform, param_combinations)

    # # for loop
    # # results = []
    # # for params in param_combinations:
    # #   results.append(run_ebcc_uniform(*params))
    
    # # Convert results to a structured DataFrame
    # results_df = pd.DataFrame(results)

    # results_df.to_csv(f'./results/{variable}_ebcc_uniform_compression.csv', index=False)

    '''
    Step 4: Plot Compression Error Distribution
    '''
    ebcc_base_compression_ratio_lst = [1, 10, 100, 1000]
    ebcc_max_error_lst = [0.01, 0.1, 1]
    param_combinations = list(itertools.product([output_path], [variable], ebcc_base_compression_ratio_lst, ebcc_max_error_lst))
    for params in param_combinations:
      plot_compression_error_dist(*params)





