import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from ebcc_wrapper import ErrorBoundedJP2KCodec
from multiprocessing import Pool

def run_scan(
  era5_path,
  year,
  month,
  variable,
  ratio = 1,
  cratio = 30
):
  reanalysis_file = os.path.join(era5_path, f"single_level/reanalysis/{year}/{month}/{variable}.npy")
  spread_file = os.path.join(era5_path, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.npy")

  if not (os.path.isfile(reanalysis_file) and os.path.isfile(spread_file)):
    print(f"[WARN] Skipping {year}-{month} {variable}: missing files.")
    return []

  data = np.load(reanalysis_file)
  spread = np.load(spread_file)
  error_bound = spread * ratio

  codec = ErrorBoundedJP2KCodec()

  steps = 6
  data = data[::steps]
  error_bound = error_bound[::steps]

  data_size = data.nbytes

  # print(f"Running {cratio} for {variable} ...")
  compression_start_time = time.time()
  blob, info = codec.compress(data, error_bound, cratio=cratio, key_fail_u16=None)
  compression_end_time = time.time()
  compression_time = compression_end_time - compression_start_time

  decompression_start_time = time.time()
  data_hat = codec.decompress(blob)
  decompression_end_time = time.time()
  decompression_time = decompression_end_time - decompression_start_time

  check_passed = np.all(np.abs(data_hat - data) <= error_bound)
  mse = float(np.mean(((data_hat - data)/(np.max(data)-np.min(data))) ** 2))
  rmse = float(np.sqrt(mse))
  compressed_size = len(blob)
  compression_ratio = data_size / compressed_size
  # print(f"  cratio: {cratio}, compression_ratio: {compression_ratio:.2f}, mse: {mse:.6e}, rmse: {rmse:.6e}, check_passed: {check_passed}")

  result = {
    "variable": variable,
    "year": year,
    "month": month,
    "ratio": ratio,
    "cratio": cratio,
    "data_size": data_size,
    "check_passed": check_passed,
    "compressed_size_jp2k": info["compressed_size_jp2k"],
    "compressed_size_fail_u16": info["compressed_size_fail_u16"],
    "compressed_size_fail_fp32": info["compressed_size_fail_fp32"],
    "compressed_size": compressed_size,
    "compression_ratio": compression_ratio,
    "mse": mse,
    "rmse": rmse,
    "compression_ratio_u16_1": info.get("compression_ratio_fail_u16_1"),
    "compression_ratio_u16_2": info.get("compression_ratio_fail_u16_2"),
    "compression_ratio_u16_3": info.get("compression_ratio_fail_u16_3"),
    "compression_ratio_u16_4": info.get("compression_ratio_fail_u16_4"),
    "compression_ratio_fail_fp32": info.get("compression_ratio_fail_fp32"),
    "key_fail_u16": info["key_fail_u16"],
    "fail_ratio_jp2k_hat": info["fail_ratio_jp2k_hat"],
    "fail_ratio_hat": info["fail_ratio_hat"],
    "fail_ratio_fp32_hat": info["fail_ratio_fp32_hat"],
    "compression_time": compression_time,
    "decompression_time": decompression_time,
    "compression_throughput": data_size / compression_time / (1024**2),
    "decompression_throughput": data_size / decompression_time / (1024**2),
  }
  return result

def run_search(
  era5_path,
  year,
  month,
  variable,
  ratio,
  mode,
):
  reanalysis_file = os.path.join(era5_path, f"single_level/reanalysis/{year}/{month}/{variable}.npy")
  spread_file = os.path.join(era5_path, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.npy")

  if not (os.path.isfile(reanalysis_file) and os.path.isfile(spread_file)):
    print(f"[WARN] Skipping {year}-{month} {variable}: missing files.")
    return []

  data = np.load(reanalysis_file)
  spread = np.load(spread_file)
  error_bound = spread * ratio

  codec = ErrorBoundedJP2KCodec()

  steps = 6
  data = data[::steps]
  error_bound = error_bound[::steps]

  data_size = data.nbytes


  # print(f"Running search for {variable} ...")
  compression_start_time = time.time()
  if mode == "full_scale":
    (blob, info), best_cratio = codec.golden_section_search_best_compression(data, error_bound)
  elif mode == "half_resolution":
    (blob, info), best_cratio = codec.golden_section_search_best_compression(data[..., ::2, ::2], error_bound[..., ::2, ::2])
  else:
    raise ValueError(f"Unknown mode: {mode}")
  blob, info = codec.compress(data, error_bound, best_cratio)
  compression_end_time = time.time()
  compression_time = compression_end_time - compression_start_time

  decompression_start_time = time.time()
  data_hat = codec.decompress(blob)
  decompression_end_time = time.time()
  decompression_time = decompression_end_time - decompression_start_time

  check_passed = np.all(np.abs(data_hat - data) <= error_bound)
  mse = float(np.mean(((data_hat - data)/(np.max(data)-np.min(data))) ** 2))
  rmse = float(np.sqrt(mse))
  compressed_size = len(blob)
  compression_ratio = data_size / compressed_size

  result = {
    "variable": variable,
    "year": year,
    "month": month,
    "ratio": ratio,
    "mode": mode,
    "best_cratio": best_cratio,
    "data_size": data_size,
    "check_passed": check_passed,
    "compressed_size_jp2k": info["compressed_size_jp2k"],
    "compressed_size_fail_u16": info["compressed_size_fail_u16"],
    "compressed_size_fail_fp32": info["compressed_size_fail_fp32"],
    "compressed_size": compressed_size,
    "compression_ratio": compression_ratio,
    "mse": mse,
    "rmse": rmse,
    "compression_ratio_fail_fp32": info.get("compression_ratio_fail_fp32"),
    "key_fail_u16": info["key_fail_u16"],
    "fail_ratio_jp2k_hat": info["fail_ratio_jp2k_hat"],
    "fail_ratio_hat": info["fail_ratio_hat"],
    "fail_ratio_fp32_hat": info["fail_ratio_fp32_hat"],
    "compression_time": compression_time,
    "decompression_time": decompression_time,
    "compression_throughput": data_size / compression_time / (1024**2),
    "decompression_throughput": data_size / decompression_time / (1024**2),
  }
  return result

def main():
  era5_path = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
  year = '2024'
  month = '10'
  ratio = 1
  variables = [
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
  ]

  num_processes = 16
  pool = Pool(processes=num_processes)
  results = []
  for variable in variables:
    for cratio in range(10, 61, 5):
      args = (era5_path, year, month, variable, ratio, cratio)
      if num_processes > 1:
        result = pool.apply_async(run_scan, args=args)
      else:
        result = run_scan(*args)
      results.append(result)
  pool.close()
  for idx in tqdm(range(len(results))):
    result = results[idx]
    if num_processes > 1:
      result = result.get()

    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/scan_cratio_single_level_{year}_{month}_{ratio}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
  
  pool.join()

  pool = Pool(processes=num_processes)
  results = []
  for variable in variables:
    for mode in ["full_scale", "half_resolution"]:
      args = (era5_path, year, month, variable, ratio, mode)
      if num_processes > 1:
        result = pool.apply_async(run_search, args=args)
      else:
        result = run_search(*args)
      results.append(result)
  pool.close()
  for idx in tqdm(range(len(results))):
    result = results[idx]
    if num_processes > 1:
      result = result.get()

    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/golden_section_search_single_level_{year}_{month}_{ratio}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

  pool.join()

if __name__ == "__main__":
  main()
