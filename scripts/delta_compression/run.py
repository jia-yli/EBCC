import os
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from ebcc_wrapper import DeltaErrorBoundedJP2KCodec
from multiprocessing import Pool

def load_data(era5_root, year, month, variable, pressure_levels, interval):
    """Load all pressure level data for given year, month, and variable."""
    data_lst = []
    spread_lst = []
    time_steps = 8
    for level in pressure_levels:
        data = np.load(os.path.join(era5_root, f"pressure_level/reanalysis/{year}/{month}/{level}/{variable}.npy"))
        spread = np.load(os.path.join(era5_root, f"pressure_level/interpolated_ensemble_spread/{year}/{month}/{level}/{variable}.npy"))

        data = data[::interval][:time_steps]
        spread = spread[::interval][:time_steps]

        data_lst.append(data)
        spread_lst.append(spread)
    data = np.concatenate(data_lst, axis=1)
    spread = np.concatenate(spread_lst, axis=1)
    return data, spread

def delta_compress(data, spread, delta_on="time", mode="pointwise"):
    """
    Compress on axis by delta
    """
    codec = DeltaErrorBoundedJP2KCodec()
    # axis to do delta compression
    n_time, n_pressure, H, W = data.shape
    if delta_on == "time":
        num_slices = n_pressure
        data = np.transpose(data, (1,0,2,3))  # [Pressure, Time, H, W]
        spread = np.transpose(spread, (1,0,2,3))
    else:
        num_slices = n_time
    # error bound setup
    if mode == "pointwise":
        error_bound = spread
    else:
        mode, target = mode.split("_")
        assert mode == "rel"
        target = float(target)
        data_range = np.max(data) - np.min(data)
        error_bound = np.full_like(data, target * data_range, dtype=np.float32)
    results = []

    for slice_idx in range(1):
        print(f"Compressing {delta_on} dimension for slice {slice_idx}/{num_slices}...")
        slice_data = data[slice_idx]
        slice_eb = error_bound[slice_idx]
        
        # Compress with delta (first as base, rest as delta)
        compressed_blob, info = codec.compress(slice_data, slice_eb)
        
        # Decompress to verify
        decompressed = codec.decompress(compressed_blob)
        
        # Calculate metrics
        compression_ratio = slice_data.nbytes / len(compressed_blob)
        check_passed = np.all(np.abs(slice_data - decompressed) <= slice_eb)
        passed_ratio = np.mean(np.abs(slice_data - decompressed) <= slice_eb)
        
        results.append({
            'delta_on': delta_on,
            'mode': mode,
            'slice_idx': slice_idx,
            'compression_ratio': compression_ratio,
            'check_passed': check_passed,
            'passed_ratio': passed_ratio,
            **info,
        })
        
        print(f"Slice {slice_idx}: CR={compression_ratio:.2f}, Check Passed={check_passed}, Passed Ratio={passed_ratio:.6f}")
    
    return results

def visualize_delta_input(data, start_time, time_delta, pressure_levels, output_dir):
    """
    Visualize input data and deltas for delta compression with coastlines.
    Plots slices and deltas for both time (data[0, :, :, :]) and pressure (data[:, 0, :, :]).
    """
    os.makedirs(output_dir, exist_ok=True)
    n_time, n_pressure, nlat, nlon = data.shape

    # ERA5 latitude: [90, -90]
    lat_coord = np.linspace(90, -90, nlat, endpoint=True)
    # ERA5 longitude: [0, 360) 
    lon_coord = np.linspace(0, 360, nlon, endpoint=False)

    # Visualize 1st time step across all pressure levels
    for i in range(n_pressure):
        curr_slice = data[0, i, :, :]  # [H, W]
        data_vmin, data_vmax = np.min(data[0]), np.max(data[0])

        current_time = (start_time + 0 * time_delta).isoformat()
        pressure_level = pressure_levels[i]

        # Plot current slice
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot data
        contour = ax.pcolormesh(
            lon_coord, lat_coord, curr_slice, 
            transform=ccrs.PlateCarree(), cmap="coolwarm",
            vmin=data_vmin, vmax=data_vmax
        )
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        cbar = plt.colorbar(contour, orientation="horizontal", pad=0.05)
        # cbar.set_label(f"Reanalysis @ {pressure_level} hPa")
        # Title
        plt.title(f"Reanalysis on {pressure_level} hPa at {current_time}")
        # Save the plot
        output_path = os.path.join(output_dir, f"pressure/reanalysis_{current_time.replace(':','').replace('-','')}_{pressure_level}hPa.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()
    
    delta = data[0, 1:, :, :] - data[0, :-1, :, :]
    delta_vmin, delta_vmax = np.min(delta), np.max(delta)
    for i in range(1, n_pressure):
        curr_slice = delta[i-1, :, :]  # [H, W]

        current_time = (start_time + 0 * time_delta).isoformat()
        pressure_level = pressure_levels[i]

        # Plot current slice
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot data
        contour = ax.pcolormesh(
            lon_coord, lat_coord, curr_slice, 
            transform=ccrs.PlateCarree(), cmap="coolwarm",
            vmin=delta_vmin, vmax=delta_vmax
        )
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        cbar = plt.colorbar(contour, orientation="horizontal", pad=0.05)
        # cbar.set_label(f"Reanalysis @ {pressure_level} hPa")
        # Title
        plt.title(f"Reanalysis Delta on {pressure_level} hPa at {current_time}")
        # Save the plot
        output_path = os.path.join(output_dir, f"pressure/reanalysis_delta_{current_time.replace(':','').replace('-','')}_{pressure_level}hPa.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()
    
    # Visualize 1st pressure level across all time steps
    for i in range(n_time):
        curr_slice = data[i, 0, :, :]  # [H, W]
        data_vmin, data_vmax = np.min(data[:,0]), np.max(data[:,0])

        current_time = (start_time + i * time_delta).isoformat()
        pressure_level = pressure_levels[0]

        # Plot current slice
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot data
        contour = ax.pcolormesh(
            lon_coord, lat_coord, curr_slice, 
            transform=ccrs.PlateCarree(), cmap="coolwarm",
            vmin=data_vmin, vmax=data_vmax
        )
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        cbar = plt.colorbar(contour, orientation="horizontal", pad=0.05)
        # cbar.set_label(f"Reanalysis @ {pressure_level} hPa")
        # Title
        plt.title(f"Reanalysis on {pressure_level} hPa at {current_time}")
        # Save the plot
        output_path = os.path.join(output_dir, f"time/reanalysis_{current_time.replace(':','').replace('-','')}_{pressure_level}hPa.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()

    delta = data[1:, 0, :, :] - data[:-1, 0, :, :]
    delta_vmin, delta_vmax = np.min(delta), np.max(delta)
    for i in range(1, n_time):
        curr_slice = delta[i-1, :, :]  # [H, W]

        current_time = (start_time + i * time_delta).isoformat()
        pressure_level = pressure_levels[0]

        # Plot current delta
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot data
        contour = ax.pcolormesh(
            lon_coord, lat_coord, curr_slice, 
            transform=ccrs.PlateCarree(), cmap="coolwarm",
            vmin=delta_vmin, vmax=delta_vmax
        )
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        cbar = plt.colorbar(contour, orientation="horizontal", pad=0.05)
        # cbar.set_label(f"Reanalysis @ {pressure_level} hPa")
        # Title
        plt.title(f"Reanalysis Delta on {pressure_level} hPa at {current_time}")
        # Save the plot
        output_path = os.path.join(output_dir, f"time/reanalysis_delta_{current_time.replace(':','').replace('-','')}_{pressure_level}hPa.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close()

def main(variable, pressure_level_type, interval):
    # Configuration
    era5_root = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
    year = "2024"
    month = "12"
    if pressure_level_type == "aurora":
        pressure_levels = ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
    elif pressure_level_type == "top6":
        pressure_levels = ["1",  "2", "3", "5", "7", "10"]
    elif pressure_level_type == "bottom6":
        pressure_levels = ["875", "900", "925", "950", "975", "1000"]
    # [
    #     "1",  "2", "3", 
    #     "5", "7", "10",
    #     "20", "30", "50",
    #     "70", "100", "125",
    #     "150", "175", "200",
    #     "225", "250", "300",
    #     "350", "400", "450",
    #     "500", "550", "600",
    #     "650", "700", "750",
    #     "775", "800", "825",
    #     "850", "875", "900",
    #     "925", "950", "975",
    #     "1000"
    # ]  

    data, spread = load_data(era5_root, year, month, variable, pressure_levels, interval)

    visualize_delta_input(data, 
        start_time=pd.Timestamp(f"{year}-{month}-01T00:00:00Z"), 
        time_delta=pd.Timedelta(hours=interval),
        pressure_levels=pressure_levels,
        output_dir=os.path.join(os.path.dirname(__file__), f"results/{variable}_{pressure_level_type}_{interval}/visualizations")
    )

    results = []
    args = itertools.product(
        [data], [spread],
        ["time", "pressure"],
        ["pointwise", "rel_0.01"],
    )

    results = []
    for arg in args:
        res = delta_compress(*arg)
        results.extend(res)

    # Save results to DataFrame
    df = pd.DataFrame(results)
    output_csv = os.path.join(os.path.dirname(__file__), f"results/{variable}_{pressure_level_type}_{interval}/delta_compression_results.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    variables = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
    pressure_level_type = ["aurora", "top6", "bottom6"]
    intervals = [1, 6]

    num_processes = 8
    pool = Pool(processes=num_processes)
    results = []
    for args in itertools.product(variables, pressure_level_type, intervals):
        if num_processes > 1:
            result = pool.apply_async(main, args=args)
        else:
            result = main(*args)
        results.append(result)

    pool.close()

    for idx in tqdm(range(len(results))):
        result = results[idx]
        if num_processes > 1:
            result = result.get()
        results[idx] = result

    pool.join()