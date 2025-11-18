import os
import bisect
import random
import itertools
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import pandas as pd

def _build_single_level_npy_storage(era5_root, npy_root, product_type, year, month, variable):
  npy_path = os.path.join(npy_root, f"single_level/{product_type}/{year}/{month}/{variable}.npy")
  if os.path.exists(npy_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "variable": variable,
      "status": "npy_exists",
    }

  nc_path = os.path.join(era5_root, f"single_level/{product_type}/{year}/{month}/{variable}.nc")
  if not os.path.exists(nc_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "variable": variable,
      "status": "nc_missing",
    }
  dataset = xr.open_dataset(nc_path, engine="netcdf4")
  assert len(dataset.data_vars) == 1
  var_short = list(dataset.data_vars)[0]

  data = dataset[var_short].values

  # make sure subdirs exist
  os.makedirs(os.path.dirname(npy_path), exist_ok=True)

  shape = data.shape
  np.save(npy_path, data)
  return {
    "product_type": product_type,
    "year": year,
    "month": month,
    "variable": variable,
    "status": "npy_built",
    "shape": shape,
  }

def _build_pressure_level_npy_storage(era5_root, npy_root, product_type, year, month, pressure_level, variable):
  npy_path = os.path.join(npy_root, f"pressure_level/{product_type}/{year}/{month}/{pressure_level}/{variable}.npy")
  if os.path.exists(npy_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "pressure_level": pressure_level,
      "variable": variable,
      "status": "npy_exists",
    }

  nc_path = os.path.join(era5_root, f"pressure_level/{product_type}/{year}/{month}/{pressure_level}/{variable}.nc")
  if not os.path.exists(nc_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "pressure_level": pressure_level,
      "variable": variable,
      "status": "nc_missing",
    }
  dataset = xr.open_dataset(nc_path, engine="netcdf4")
  assert len(dataset.data_vars) == 1
  var_short = list(dataset.data_vars)[0]

  data = dataset[var_short].values

  # make sure subdirs exist
  os.makedirs(os.path.dirname(npy_path), exist_ok=True)

  shape = data.shape
  np.save(npy_path, data)
  return {
    "product_type": product_type,
    "year": year,
    "month": month,
    "pressure_level": pressure_level,
    "variable": variable,
    "status": "npy_built",
    "shape": shape,
  }

def build_npy_storage(
  era5_root,
  npy_root,
  product_types,
  years,
  months,
  single_level_variables=[],
  pressure_levels=[],
  pressure_level_variables=[],
  num_processes=8,
):
  # single level
  pool = Pool(processes=num_processes)
  results = []
  for args in itertools.product([era5_root], [npy_root], product_types, years, months, single_level_variables):
    if num_processes > 1:
      result = pool.apply_async(_build_single_level_npy_storage, args=args)
    else:
      result = _build_single_level_npy_storage(*args)
    results.append(result)
  
  # pressure level
  for args in itertools.product([era5_root], [npy_root], product_types, years, months, pressure_levels, pressure_level_variables):
    if num_processes > 1:
      result = pool.apply_async(_build_pressure_level_npy_storage, args=args)
    else:
      result = _build_pressure_level_npy_storage(*args)
    results.append(result)
  
  pool.close()

  for idx in tqdm(range(len(results))):
    result = results[idx]
    if num_processes > 1:
      result = result.get()

    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_npy_storage_log.csv"), index=False)
  
  pool.join()

if __name__ == "__main__":
  era5_root = "/capstor/scratch/cscs/ljiayong/datasets/ERA5_large"
  npy_root = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
  # npy_root = "/capstor/scratch/cscs/ljiayong/cache/era5_npy"

  # product_types = ["reanalysis", "ensemble_members", "ensemble_mean", "ensemble_spread"]
  product_types = ["reanalysis", "ensemble_spread"]
  years = [str(y) for y in range(2024, 2025)]
  # months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
  months = ["12"]

  single_level_variables = [
    # "100m_u_component_of_wind",
    # "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "2m_dewpoint_temperature",
    "2m_temperature",
    # "ice_temperature_layer_1",
    # "ice_temperature_layer_2",
    # "ice_temperature_layer_3",
    # "ice_temperature_layer_4",
    # "maximum_2m_temperature_since_previous_post_processing",
    "mean_sea_level_pressure",
    # "minimum_2m_temperature_since_previous_post_processing",
    # "sea_surface_temperature",
    # "skin_temperature",
    # "surface_pressure",
    # "total_precipitation",

    "geopotential",
    "soil_type",
    "land_sea_mask",
  ]

  pressure_levels = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
  ]

  pressure_level_variables = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
  ]

  build_npy_storage(
    era5_root = era5_root,
    npy_root = npy_root,
    product_types = product_types,
    years = years,
    months = months,
    single_level_variables = single_level_variables,
    pressure_levels = pressure_levels,
    pressure_level_variables = pressure_level_variables,
    num_processes = 32,
  )

