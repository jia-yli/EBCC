import os
import bisect
import random
import itertools
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import pandas as pd

def _interpolate_single_level_npy_storage(npy_root, product_type, year, month, variable):
  out_npy_path = os.path.join(npy_root, f"single_level/interpolated_{product_type}/{year}/{month}/{variable}.npy")
  if os.path.exists(out_npy_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "variable": variable,
      "status": "out_npy_exists",
    }

  in_npy_path = os.path.join(npy_root, f"single_level/{product_type}/{year}/{month}/{variable}.npy")
  if not os.path.exists(in_npy_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "variable": variable,
      "status": "in_npy_missing",
    }
  in_npy = np.load(in_npy_path)

  n_time, H, W = in_npy.shape

  # interpolate time to 3x, with const filling at the end
  arr_t = np.zeros((3*n_time, H, W), dtype=np.float32)
  arr1_t = in_npy # [n_time, H, W]
  arr2_t = np.concatenate((in_npy[1:], in_npy[-1:]), axis=0) # [n_time, H, W]
  arr_t[0::3] = arr1_t
  arr_t[1::3] = (2*arr1_t + arr2_t) / 3
  arr_t[2::3] = (arr1_t + 2*arr2_t) / 3

  # interpolate h to 2x, while keeping the original first and last
  arr_h = np.zeros((3*n_time, 2*H-1, W), dtype=np.float32)
  arr1_h = arr_t[:, :-1, :]  # [3*n_time, H-1, W]
  arr2_h = arr_t[:, 1:, :]   # [3*n_time, H-1, W]
  arr_h[:, 0::2, :] = arr_t
  arr_h[:, 1::2, :] = (arr1_h + arr2_h) / 2

  # interpolate w to 2x, while wrapping around
  arr_w = np.zeros((3*n_time, 2*H-1, 2*W), dtype=np.float32)
  arr1_w = arr_h  # [3*n_time, 2*H-1, W]
  arr2_w = np.concatenate((arr_h[:, :, 1:], arr_h[:, :, 0:1]), axis=2)  # [3*n_time, 2*H-1, W]
  arr_w[:, :, 0::2] = arr_h
  arr_w[:, :, 1::2] = (arr1_w + arr2_w) / 2

  out_npy = arr_w
  assert (in_npy == out_npy[0::3, 0::2, 0::2]).all()

  # make sure subdirs exist
  os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
  np.save(out_npy_path, out_npy)
  return {
    "product_type": product_type,
    "year": year,
    "month": month,
    "variable": variable,
    "status": "npy_built",
    "in_shape": in_npy.shape,
    "out_shape": out_npy.shape,
  }

def _interpolate_pressure_level_npy_storage(npy_root, product_type, year, month, pressure_level, variable):
  out_npy_path = os.path.join(npy_root, f"pressure_level/interpolated_{product_type}/{year}/{month}/{pressure_level}/{variable}.npy")
  if os.path.exists(out_npy_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "pressure_level": pressure_level,
      "variable": variable,
      "status": "out_npy_exists",
    }

  in_npy_path = os.path.join(npy_root, f"pressure_level/{product_type}/{year}/{month}/{pressure_level}/{variable}.npy")
  if not os.path.exists(in_npy_path):
    return {
      "product_type": product_type,
      "year": year,
      "month": month,
      "pressure_level": pressure_level,
      "variable": variable,
      "status": "in_npy_missing",
    }
  in_npy = np.load(in_npy_path)

  n_time, n_pressure, H, W = in_npy.shape
  assert n_pressure == 1, "Pressure level data should have only one pressure level dimension."

  # interpolate time to 3x, with const filling at the end
  arr_t = np.zeros((3*n_time, n_pressure, H, W), dtype=np.float32)
  arr1_t = in_npy # [n_time, n_pressure, H, W]
  arr2_t = np.concatenate((in_npy[1:], in_npy[-1:]), axis=0) # [n_time, n_pressure, H, W]
  arr_t[0::3] = arr1_t
  arr_t[1::3] = (2*arr1_t + arr2_t) / 3
  arr_t[2::3] = (arr1_t + 2*arr2_t) / 3

  # interpolate h to 2x, while keeping the original first and last
  arr_h = np.zeros((3*n_time, n_pressure, 2*H-1, W), dtype=np.float32)
  arr1_h = arr_t[:, :, :-1, :]  # [3*n_time, n_pressure, H-1, W]
  arr2_h = arr_t[:, :, 1:, :]   # [3*n_time, n_pressure, H-1, W]
  arr_h[:, :, 0::2, :] = arr_t
  arr_h[:, :, 1::2, :] = (arr1_h + arr2_h) / 2

  # interpolate w to 2x, while wrapping around
  arr_w = np.zeros((3*n_time, n_pressure, 2*H-1, 2*W), dtype=np.float32)
  arr1_w = arr_h  # [3*n_time, n_pressure, 2*H-1, W]
  arr2_w = np.concatenate((arr_h[:, :, :, 1:], arr_h[:, :, :, 0:1]), axis=3)  # [3*n_time, n_pressure, 2*H-1, W]
  arr_w[:, :, :, 0::2] = arr_h
  arr_w[:, :, :, 1::2] = (arr1_w + arr2_w) / 2

  out_npy = arr_w
  assert (in_npy == out_npy[0::3, :, 0::2, 0::2]).all()

  # make sure subdirs exist
  os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
  np.save(out_npy_path, out_npy)
  return {
    "product_type": product_type,
    "year": year,
    "month": month,
    "pressure_level": pressure_level,
    "variable": variable,
    "status": "npy_built",
    "in_shape": in_npy.shape,
    "out_shape": out_npy.shape,
  }

def interpolate_npy_array(
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
  for args in itertools.product([npy_root], product_types, years, months, single_level_variables):
    if num_processes > 1:
      result = pool.apply_async(_interpolate_single_level_npy_storage, args=args)
    else:
      result = _interpolate_single_level_npy_storage(*args)
    results.append(result)

  # pressure level
  for args in itertools.product([npy_root], product_types, years, months, pressure_levels, pressure_level_variables):
    if num_processes > 1:
      result = pool.apply_async(_interpolate_pressure_level_npy_storage, args=args)
    else:
      result = _interpolate_pressure_level_npy_storage(*args)
    results.append(result)

  pool.close()

  for idx in tqdm(range(len(results))):
    result = results[idx]
    if num_processes > 1:
      result = result.get()

    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interpolate_npy_array_log.csv"), index=False)
  
  pool.join()

if __name__ == "__main__":
  npy_root = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
  # npy_root = "/capstor/scratch/cscs/ljiayong/cache/era5_npy"

  product_types = ["ensemble_spread"]
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

  interpolate_npy_array(
    npy_root = npy_root,
    product_types = product_types,
    years = years,
    months = months,
    single_level_variables = single_level_variables,
    pressure_levels = pressure_levels,
    pressure_level_variables = pressure_level_variables,
    num_processes = 32,
  )

