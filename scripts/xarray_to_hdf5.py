import os
import xarray as xr
import h5py
import numpy as np

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

def compare_nc_hdf5(nc_file, hdf5_file):
  """
  Compare the NetCDF (.nc) and HDF5 (.h5) files for size and content consistency.
  
  Parameters:
    nc_file (str): Path to the NetCDF file.
    hdf5_file (str): Path to the HDF5 file.
  """

  # Open NetCDF file
  nc_data = xr.open_dataset(nc_file)

  # Open HDF5 file
  with h5py.File(hdf5_file, 'r') as hdf5_f:
    match = True  # Flag to check if all data matches

    for var_name, da in nc_data.data_vars.items():
      nc_values = da.values  # Extract NumPy array from xarray
      hdf5_values = np.array(hdf5_f[var_name])  # Extract NumPy array from HDF5

      # Check if data is the same
      if not np.array_equal(nc_values, hdf5_values):
        print(f"Mismatch found in variable: {var_name}")
        match = False

    if match:
      print("All variables match exactly between NetCDF and HDF5!")
    else:
      print("Some variables do not match!")

nc_file_path = "/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/2m_temperature.nc"
hdf5_file_path = "/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/2m_temperature.hdf5"  # Replace with your desired output file path
convert_nc_to_hdf5(nc_file_path, hdf5_file_path)
compare_nc_hdf5(nc_file_path, hdf5_file_path)

nc_size = os.path.getsize(nc_file_path)
hdf5_size = os.path.getsize(hdf5_file_path)

print(f"File Sizes:\nNetCDF: {nc_size/1024/1024/1024:.4f} GiB\nHDF5: {hdf5_size/1024/1024/1024:.4f} GiB\n")