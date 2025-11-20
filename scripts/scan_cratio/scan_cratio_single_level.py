import os
import time

import numpy as np
import pandas as pd

from ebcc_wrapper import ErrorBoundedJP2KCodec

def run_scan(
  era5_path,
  year,
  month,
  variable,
  ratio = 1,
  cratio_start = 10,
  cratio_stop = 100,
  cratio_step = 5,
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

  steps = 8
  data = data[:steps]
  error_bound = error_bound[:steps]

  data_size = data.nbytes

  results = []
  for cratio in range(cratio_start, cratio_stop + 1, cratio_step):
    print(f"Running {cratio} for {variable} ...")
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
    results.append(result)
  return results

def main():
  era5_path_np = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
  year = '2024'
  month = '12'
  ratio = 0.5
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
  output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"scan_cratio_single_level_{year}_{month}_{ratio}.csv")

  results = []
  for variable in variables:
    scan_results = run_scan(
      era5_path=era5_path_np,
      year=year,
      month=month,
      variable=variable,
      ratio=ratio,
    )
    results.extend(scan_results)
    pd.DataFrame(results).to_csv(output_csv, index=False)


if __name__ == "__main__":
  main()
