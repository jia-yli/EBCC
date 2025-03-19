import os
current_folder = os.path.dirname(os.path.realpath(__file__))
os.environ["HDF5_PLUGIN_PATH"] = os.path.join(current_folder, '../src/build/lib')

import time
import h5py
import xarray as xr
import numpy as np
import pandas as pd
from ebcc_wrapper import EBCC_Filter
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata

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
      data = da.values[0:4] # Convert xarray DataArray to NumPy array
      hdf5_f.create_dataset(var_name, data=data)

def spatial_interpolation(data, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, start_idx=0, end_idx=None):
  data = data[start_idx:end_idx]
  num_time_steps = data.shape[0]
  data_interpolated = np.empty((num_time_steps, lon_target_grid.shape[0], lon_target_grid.shape[1]))
  points = np.column_stack((lat_source_grid.ravel(), lon_source_grid.ravel()))
  for t_idx in range(num_time_steps):
    values = data[t_idx].ravel()
    data_interpolated[t_idx] = griddata(points, values, (lat_target_grid, lon_target_grid), method='linear')
  return data_interpolated

def interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_file, output_file):
  # Load reanalysis and ensemble datasets
  ds_reanalysis = xr.open_dataset(reanalysis_file)
  ds_ensemble = xr.open_dataset(ensemble_file)

  # Extract coordinates from reanalysis dataset (target grid)
  time_target = ds_reanalysis['valid_time'].values
  lat_target = ds_reanalysis['latitude'].values
  lon_target = ds_reanalysis['longitude'].values

  # Extract coordinates and spread variable from ensemble dataset (source grid)
  time_source = ds_ensemble['valid_time'].values
  lat_source = ds_ensemble['latitude'].values
  lon_source = ds_ensemble['longitude'].values

  # shape: (Time, Latitude, Longitude)
  assert len(list(ds_ensemble.data_vars)) == 1
  for var_name, da in ds_ensemble.data_vars.items():
    data_source = da.values  # Convert xarray DataArray to NumPy array
    # Step 1: Interpolate Spatial Dims
    # handle longitude wrap-up at 360
    # Src
    lon_source_extended = np.concatenate((lon_source, lon_source[0:1] + 360), axis=0)
    lat_source_grid, lon_source_grid = np.meshgrid(lat_source, lon_source_extended, indexing='ij')
    data_extended = np.concatenate((data_source, data_source[:, :, 0:1]), axis=2)
    # Dst
    lat_target_grid, lon_target_grid = np.meshgrid(lat_target, lon_target, indexing='ij')

    num_time_steps = 4
    num_jobs = (data_extended.shape[0] + num_time_steps - 1) // num_time_steps

    # mp
    with mp.Pool(processes=32) as pool:  # Adjust processes as needed
      results = [pool.apply_async(spatial_interpolation,
        (data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0]))
      ) for idx in range(num_jobs)]
      results = [result.get() for result in results]
    
    # for loop
    # results = [spatial_interpolation(
    #   data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0])
    # ) for idx in range(num_jobs)]

    results = np.concatenate(results, axis=0)

    # Step 2: Interpolate Temporal Dim
    ds_interp_space = xr.Dataset(
      {
        var_name: (['valid_time', 'latitude', 'longitude'], results)
      },
      coords={
        'valid_time': time_source,
        'latitude': lat_target,
        'longitude': lon_target
      }
    )
    # Interpolate in time to match reanalysis time grid
    ds_interp_time = ds_interp_space.interp(
      valid_time=time_target, 
      method="linear")
    ds_output = ds_interp_time.ffill(dim="valid_time")

    # Save to new hdf5 file
    with h5py.File(output_file, 'w') as hdf5_f:
      for var_name, da in ds_output.data_vars.items():
        data = da.values[0:4] # Convert xarray DataArray to NumPy array
        hdf5_f.create_dataset(var_name, data=data)

def compress_hdf5_ebcc_pointwise(input_hdf5, input_uncertainty_hdf5, output_hdf5, ebcc_base_compression_ratio, ebcc_pointwise_max_error_ratio, is_pointwise):
  # compression and compression time
  with h5py.File(input_hdf5, 'r') as hdf5_in:
    compression_start_time = time.time()

    with h5py.File(output_hdf5, 'w') as hdf5_out:
      for var_name in hdf5_in.keys():
        data = np.array(hdf5_in[var_name])  # Read dataset
        if is_pointwise:
          error_bound = np.ones_like(data)
          if input_uncertainty_hdf5 is not None:
            with h5py.File(input_uncertainty_hdf5, 'r') as hdf5_uncertainty_in:
              error_bound = np.array(hdf5_uncertainty_in[var_name])
          data = np.stack([data, error_bound], axis=-3)
          ebcc_filter = EBCC_Filter(
            base_cr=ebcc_base_compression_ratio, # base compression ratio
            height=data.shape[-2],  # height of each 2D data chunk
            width=data.shape[-1],  # width of each 2D data chunk
            data_dim=len(data.shape), # data dimension, required to specify the HDF5 chunk shape
            residual_opt=("pointwise_max_error", ebcc_pointwise_max_error_ratio))
          hdf5_out.create_dataset(var_name, data=data, **ebcc_filter)
        else:
          ebcc_filter = EBCC_Filter(
            base_cr=ebcc_base_compression_ratio, # base compression ratio
            height=data.shape[-2],  # height of each 2D data chunk
            width=data.shape[-1],  # width of each 2D data chunk
            data_dim=len(data.shape), # data dimension, required to specify the HDF5 chunk shape
            residual_opt=("max_error_target", ebcc_pointwise_max_error_ratio)) # directory to the compiled HDF5 filter plugin
          hdf5_out.create_dataset(var_name, data=data, **ebcc_filter)

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

def check_error_pointwise(original_hdf5, uncertainty_hdf5, compressed_hdf5, ebcc_pointwise_max_error_ratio, is_pointwise):
  check_passed = True
  # Open HDF5 files
  with h5py.File(original_hdf5, 'r') as orig_file, h5py.File(compressed_hdf5, 'r') as comp_file:
    for dataset_name in orig_file.keys():
      orig_data = np.array(orig_file[dataset_name])
      comp_data = np.array(comp_file[dataset_name])

      if is_pointwise:
        comp_data = comp_data[:, 0, :, :]

      assert (orig_data.shape == comp_data.shape)

      # Compute absolute difference (point-wise error)
      error = np.abs(orig_data - comp_data)

      if is_pointwise:
        if uncertainty_hdf5 is not None:
          with h5py.File(uncertainty_hdf5, 'r') as uncertainty_file:
            error_bound = np.array(uncertainty_file[dataset_name])*ebcc_pointwise_max_error_ratio
        else:
          error_bound = np.ones_like(orig_data) * ebcc_pointwise_max_error_ratio
        check_passed = check_passed and ((error <= error_bound).all())
      else:
        check_passed = check_passed and ((error <= ebcc_pointwise_max_error_ratio).all())

  return check_passed

def run_ebcc_pointwise(output_path, variable, ebcc_base_compression_ratio, ebcc_pointwise_max_error_ratio, is_pointwise):
  '''
  EBCC compressor with pointwise max error bound
  '''
  input_hdf5_file_path = os.path.join(output_path, f'{variable}.hdf5')
  input_uncertainty_file_path = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
  output_hdf5_file_path = os.path.join(output_path, f'{variable}_compressed_ebcc_pointwise_cr_{ebcc_base_compression_ratio}_ratio_{ebcc_pointwise_max_error_ratio}_pointwise_{is_pointwise}.hdf5')
  compression_time, decompression_time = compress_hdf5_ebcc_pointwise(input_hdf5_file_path, input_uncertainty_file_path, output_hdf5_file_path, ebcc_base_compression_ratio, ebcc_pointwise_max_error_ratio, is_pointwise)
  input_size = os.path.getsize(input_hdf5_file_path)
  output_size = os.path.getsize(output_hdf5_file_path)
  compression_ratio = input_size/output_size
  check_passed = check_error_pointwise(input_hdf5_file_path, input_uncertainty_file_path, output_hdf5_file_path, ebcc_pointwise_max_error_ratio, is_pointwise)
  results = {
    'is_pointwise': is_pointwise,
    'ebcc_base_compression_ratio' : ebcc_base_compression_ratio,
    'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
    'compression_ratio' : compression_ratio,
    'check_passed' : check_passed,
    'compression_time' : compression_time,
    'decompression_time' : decompression_time,
  }
  return results

def plot_compression_error_dist(output_path, variable, ebcc_base_compression_ratio, ebcc_pointwise_max_error_ratio, is_pointwise):
  '''
  Plot EBCC Compressor Error Dist
  '''
  original_hdf5 = os.path.join(output_path, f'{variable}.hdf5')
  uncertainty_hdf5 = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
  compressed_hdf5 = os.path.join(output_path, f'{variable}_compressed_ebcc_pointwise_cr_{ebcc_base_compression_ratio}_ratio_{ebcc_pointwise_max_error_ratio}_pointwise_{is_pointwise}.hdf5')
  with h5py.File(original_hdf5, 'r') as orig_file, h5py.File(compressed_hdf5, 'r') as comp_file:
    for dataset_name in orig_file.keys():
      orig_data = np.array(orig_file[dataset_name])
      comp_data = np.array(comp_file[dataset_name])

      if is_pointwise:
        comp_data = comp_data[:, 0, :, :]

      assert (orig_data.shape == comp_data.shape)

      # Compute absolute difference (point-wise error)
      error = np.abs(orig_data - comp_data)

      if is_pointwise:
        if uncertainty_hdf5 is not None:
          with h5py.File(uncertainty_hdf5, 'r') as uncertainty_file:
            error_bound = np.array(uncertainty_file[dataset_name])*ebcc_pointwise_max_error_ratio
        else:
          error_bound = np.ones_like(orig_data) * ebcc_pointwise_max_error_ratio
        error_to_error_bound = error/error_bound
        # Plot error_bound
        fig = plt.figure(figsize=(12, 6))
        plt.hist(error_bound.flatten(), bins=40, alpha=0.7, edgecolor=None)
        plt.xlabel(f"{variable} Max Error Bound\nError Bound, Min: {error_bound.min()}, Max: {error_bound.max()}, Avg: {error_bound.mean()}\nData, Min: {orig_data.min()}, Max: {orig_data.max()}, Avg: {orig_data.mean()}")
        plt.ylabel(f"Count")
        plt.title(f"EBCC {variable} Max Error Bound Dist\nBase Compression Ratio {ebcc_base_compression_ratio}, Max Error Bound to Uncertainty {ebcc_pointwise_max_error_ratio}")
        plt.grid(True)
        output_path = f"./results/{variable}_ebcc_error_bound_cr_{ebcc_base_compression_ratio}_ratio_{ebcc_pointwise_max_error_ratio}.png"
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()
        # Plot error/error_bound
        fig = plt.figure(figsize=(12, 6))
        plt.hist(error_to_error_bound.flatten(), bins=40, alpha=0.7, edgecolor=None)
        plt.xlabel(f"{variable} Error/Max Error Bound Distribution\nError to Error Bound, Min: {error_to_error_bound.min()}, Max: {error_to_error_bound.max()}, Avg: {error_to_error_bound.mean()}")
        plt.ylabel(f"Count")
        plt.title(f"EBCC {variable} Error/Max Error Bound Distribution\nBase Compression Ratio {ebcc_base_compression_ratio}, Max Error Bound to Uncertainty {ebcc_pointwise_max_error_ratio}")
        plt.grid(True)
        output_path = f"./results/{variable}_ebcc_error_to_error_bound_cr_{ebcc_base_compression_ratio}_ratio_{ebcc_pointwise_max_error_ratio}.png"
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()
      else:
        dist = error
        # Plot
        fig = plt.figure(figsize=(12, 6))
        plt.hist(error.flatten(), bins=40, alpha=0.7, edgecolor=None)
        plt.xlabel(f"{variable} Compression Error")
        plt.ylabel(f"Count")
        plt.title(f"EBCC {variable} Compression Error Dist\nBase Compression Ratio {ebcc_base_compression_ratio}, Max Error Bound {ebcc_pointwise_max_error_ratio}")
        plt.grid(True)
        output_path = f"./results/{variable}_ebcc_uniform_compression_error_cr_{ebcc_base_compression_ratio}_err_{ebcc_pointwise_max_error_ratio}.png"
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
  variable_lst = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    # "total_precipitation"
  ]

  # global value
  for variable_idx in range(len(variable_lst)):
    variable = variable_lst[variable_idx]
    output_path = f'/capstor/scratch/cscs/ljiayong/workspace/EBCC/test'
    os.makedirs(output_path, exist_ok = True)

    '''
    Step 1: NetCDF to HDF5 without compression
    '''
    print(f'[INFO] Converting NetCDF to HDF5 for Variable {variable} ......')
    nc_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    hdf5_file = os.path.join(output_path, f'{variable}.hdf5')
    convert_nc_to_hdf5(nc_file, hdf5_file)


    '''
    Step 2: Interpolate Ensemble Spread
    '''
    print(f'[INFO] Interpolating Ensemble Spread for Variable {variable} ......')
    reanalysis_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    ensemble_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/ensemble_spread/{variable}.nc'
    output_file = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
    interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_file, output_file)

    '''
    Param Combinations
    '''
    ebcc_base_compression_ratio_lst = [1, 10, 100, 1000]
    ebcc_pointwise_max_error_ratio_lst = [0.1, 0.5, 1]

    # ebcc_base_compression_ratio_lst = [10]
    # ebcc_pointwise_max_error_ratio_lst = [0.1]

    param_combinations = list(itertools.product([output_path], [variable], ebcc_base_compression_ratio_lst, ebcc_pointwise_max_error_ratio_lst, [1]))
    
    '''
    Step 3: Run EBCC with Pointwise Error Bound
    '''
    # MP
    print(f'[INFO] Running EBCC for Variable {variable} ......')
    with mp.Pool(processes=8) as pool:
      results = pool.starmap(run_ebcc_pointwise, param_combinations)

    # for loop
    # results = []
    # for params in param_combinations:
    #   print(f'[INFO] Starting Pointwise Error Compression with Param: {params}')
    #   results.append(run_ebcc_pointwise(*params))
    
    # Convert results to a structured DataFrame
    results_df = pd.DataFrame(results)

    results_df.to_csv(f'./results/{variable}_ebcc_pointwise_compression.csv', index=False)

    '''
    Step 4: Plot Compression Error Distribution
    '''
    print(f'[INFO] Ploting EBCC for Variable {variable} ......')
    for params in param_combinations:
      plot_compression_error_dist(*params)





