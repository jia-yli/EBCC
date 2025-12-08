# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && pip install microsoft-aurora joblib && pip install -e .
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from joblib import Memory

from ebcc_wrapper import ErrorBoundedJP2KCodec
from aurora import Aurora, Batch, Metadata

memory = Memory("/capstor/scratch/cscs/ljiayong/cache/EBCC/aurora_delta_compression")

@memory.cache
def load_all_data(era5_root, year, month, pressure_levels, num_steps):
     # Load all variables
    all_data = {}
    all_spread = {}
    
    # Surface variables
    for var in ["t2m", "u10", "v10", "msl", "z", "slt", "lsm"]:
        single_level_filenames = {
            "t2m": "2m_temperature",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "msl": "mean_sea_level_pressure",
            "z": "geopotential",
            "slt": "soil_type",
            "lsm": "land_sea_mask",
        }
        variable = single_level_filenames[var]
        data = np.load(os.path.join(era5_root, f"single_level/reanalysis/{year}/{month}/{variable}.npy"))
        spread = np.load(os.path.join(era5_root, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.npy"))
        
        # Take every 6th timestep
        data = data[::6][:num_steps][..., :-1, :]
        spread = spread[::6][:num_steps][..., :-1, :]
        all_data[var] = data
        all_spread[var] = spread
    
    # Atmospheric variables
    for var in ["temperature", "u_component_of_wind", "v_component_of_wind", 
                "specific_humidity", "geopotential"]:
        data_lst = []
        spread_lst = []
        
        for level in pressure_levels:
            data = np.load(os.path.join(era5_root, f"pressure_level/reanalysis/{year}/{month}/{level}/{var}.npy"))
            spread = np.load(os.path.join(era5_root, f"pressure_level/interpolated_ensemble_spread/{year}/{month}/{level}/{var}.npy"))
            
            # Take every 6th timestep (6-hour intervals) for num_steps
            data = data[::6][:num_steps][..., :-1, :]
            spread = spread[::6][:num_steps][..., :-1, :]
            
            data_lst.append(data)
            spread_lst.append(spread)
        
        data = np.concatenate(data_lst, axis=1)  # [Time, Pressure, H, W]
        spread = np.concatenate(spread_lst, axis=1)
        all_data[var] = data
        all_spread[var] = spread
    return all_data, all_spread

def create_batch_from_data(surf_vars, static_vars, atmos_vars, 
                           lat, lon, time_start, pressure_levels, step_idx):
    """Create Aurora Batch from numpy arrays at given step."""
    # Get data for steps step_idx and step_idx+1 (two consecutive 6-hour steps)
    return Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars["2t"][step_idx:step_idx+2][None]).float(),
            "10u": torch.from_numpy(surf_vars["10u"][step_idx:step_idx+2][None]).float(),
            "10v": torch.from_numpy(surf_vars["10v"][step_idx:step_idx+2][None]).float(),
            "msl": torch.from_numpy(surf_vars["msl"][step_idx:step_idx+2][None]).float(),
        },
        static_vars={
            "z": torch.from_numpy(static_vars["z"]).float(),
            "slt": torch.from_numpy(static_vars["slt"]).float(),
            "lsm": torch.from_numpy(static_vars["lsm"]).float(),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars["t"][step_idx:step_idx+2][None]).float(),
            "u": torch.from_numpy(atmos_vars["u"][step_idx:step_idx+2][None]).float(),
            "v": torch.from_numpy(atmos_vars["v"][step_idx:step_idx+2][None]).float(),
            "q": torch.from_numpy(atmos_vars["q"][step_idx:step_idx+2][None]).float(),
            "z": torch.from_numpy(atmos_vars["z"][step_idx:step_idx+2][None]).float(),
        },
        metadata=Metadata(
            lat=torch.from_numpy(lat).float(),
            lon=torch.from_numpy(lon).float(),
            time=(time_start + timedelta(hours=6*(step_idx+1)),),
            atmos_levels=tuple(int(level) for level in pressure_levels),
        ),
    )

def compress_all_variables_at_step(codec, all_data, all_spread, step):
    """Compress all variables at a given step together."""
    # Collect all data and error bounds at this step
    total_bytes = 0
    compressed_blobs = []
    
    for var_name, data in all_data.items():
        slice_data = data[step]  # Shape varies by variable type
        slice_eb = all_spread[var_name][step]
        
        (compressed_blob, info), cratio = codec.golden_section_search_best_compression(slice_data[None], slice_eb[None])
        compressed_blobs.append((var_name, compressed_blob, slice_data.shape))
        
        total_bytes += slice_data.nbytes
    
    # Calculate total compressed size
    total_compressed = sum(len(blob) for _, blob, _ in compressed_blobs)
    compression_ratio = total_bytes / total_compressed
    
    return compressed_blobs, compression_ratio

def decompress_all_variables_at_step(codec, compressed_blobs):
    """Decompress all variables at a given step."""
    decompressed = {}
    for var_name, blob, shape in compressed_blobs:
        dec = codec.decompress(blob)[0]
        decompressed[var_name] = dec
    return decompressed

def aurora_predictive_compression(era5_root, year, month, num_steps):
    """
    Compress weather data using Aurora predictions.
    Steps 0-1: Direct compression of all variables
    Steps 2+: Compress residual (data - prediction) for all variables
    """
    codec = ErrorBoundedJP2KCodec()
    pressure_levels = ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
    
    # Load model
    model = Aurora(use_lora=False)
    model.load_checkpoint_local("/capstor/scratch/cscs/ljiayong/workspace/aurora/aurora-0.25-pretrained.ckpt")
    model.eval()
    model = model.to("cuda")
    
    print(f"\n{'='*60}")
    print(f"Loading all variables")
    print(f"{'='*60}")
    
    all_data, all_spread = load_all_data(era5_root, year, month, pressure_levels, num_steps)
    
    # Map variable names to Aurora names
    var_map = {
        "t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl",
        "temperature": "t", "u_component_of_wind": "u",
        "v_component_of_wind": "v", "specific_humidity": "q", "geopotential": "z"
    }
    
    # Get coordinates
    n_time, n_pressure, nlat, nlon = all_data["temperature"].shape
    lat = np.linspace(90, -90, nlat, endpoint=False)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    time_start = datetime(int(year), int(month), 1, 0, 0, 0)
    
    compression_ratios = []
    decompressed_data = {k: v.copy() for k, v in all_data.items()}
    
    # Steps 0 and 1: Direct compression of all variables
    for step in range(min(2, num_steps)):
        print(f"\nStep {step}: Direct compression of all variables")
        
        compressed_blobs, cr = compress_all_variables_at_step(
            codec, all_data, all_spread, step
        )
        
        # Decompress to maintain state
        decompressed = decompress_all_variables_at_step(codec, compressed_blobs)
        for var_name, dec_data in decompressed.items():
            decompressed_data[var_name][step] = dec_data
        
        compression_ratios.append(cr)
        print(f"  Overall compression ratio: {cr:.2f}")
    
    # Steps 2+: Predictive compression
    with torch.inference_mode():
        for step in range(2, num_steps):
            print(f"\nStep {step}: Predictive compression of all variables")
            
            # Prepare data for Aurora
            surf_vars = {
                "2t": decompressed_data["t2m"],
                "10u": decompressed_data["u10"],
                "10v": decompressed_data["v10"],
                "msl": decompressed_data["msl"],
            }
            
            static_vars = {
                "z": all_data["z"][0],
                "slt": all_data["slt"][0],
                "lsm": all_data["lsm"][0],
            }
            
            atmos_vars = {
                "t": decompressed_data["temperature"],
                "u": decompressed_data["u_component_of_wind"],
                "v": decompressed_data["v_component_of_wind"],
                "q": decompressed_data["specific_humidity"],
                "z": decompressed_data["geopotential"],
            }
            
            # Create batch from decompressed data at step-2 and step-1
            batch = create_batch_from_data(
                surf_vars, static_vars, atmos_vars,
                lat, lon, time_start, pressure_levels, step-2
            )

            # Run Aurora prediction
            pred = model.forward(batch)
            
            # Extract predictions and compute residuals
            residuals = {}
            residuals_spread = {}
            total_bytes = 0
            compressed_blobs = []
            
            # Surface variables
            for var_name in ["t2m", "u10", "v10", "msl"]:
                aurora_name = var_map[var_name]
                pred_data = pred.surf_vars[aurora_name][0, 0].cpu().numpy() # [1, 1, H, W]
                true_data = all_data[var_name][step]
                residual = true_data - pred_data
                
                slice_eb = all_spread[var_name][step]
                compressed_blob, info = codec.golden_section_search_best_compression(residual[None], slice_eb[None])
                compressed_blobs.append((var_name, compressed_blob, residual.shape))
                
                total_bytes += true_data.nbytes
            
            # Atmospheric variables
            for var_name in ["temperature", "u_component_of_wind", "v_component_of_wind",
                           "specific_humidity", "geopotential"]:
                aurora_name = var_map[var_name]
                pred_data = pred.atmos_vars[aurora_name][0, 0].cpu().numpy()
                true_data = all_data[var_name][step]
                residual = true_data - pred_data
                
                slice_eb = all_spread[var_name][step]
                compressed_blob, info = codec.golden_section_search_best_compression(residual[None], slice_eb[None])
                compressed_blobs.append((var_name, compressed_blob, residual.shape))
                
                total_bytes += true_data.nbytes
            
            # Calculate compression ratio for all variables at this step
            total_compressed = sum(len(blob) for _, blob, _ in compressed_blobs)
            cr = total_bytes / total_compressed
            compression_ratios.append(cr)
            
            # Decompress residuals and reconstruct
            for var_name, blob, shape in compressed_blobs:
                decompressed_residual = codec.decompress(blob)[0]
                
                # Get prediction
                if var_name in ["t2m", "u10", "v10", "msl"]:
                    aurora_name = var_map[var_name]
                    pred_data = pred.surf_vars[aurora_name][0, 0].cpu().numpy()
                else:
                    aurora_name = var_map[var_name]
                    pred_data = pred.atmos_vars[aurora_name][0, 0].cpu().numpy()
                
                # Reconstruct
                decompressed_data[var_name][step] = pred_data + decompressed_residual
            
            print(f"  Overall compression ratio: {cr:.2f}")
    
    return compression_ratios

def main():
    era5_root = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"
    year = "2024"
    month = "12"
    
    num_steps = 8  # Number of 6-hour steps
    
    compression_ratios = aurora_predictive_compression(
        era5_root, year, month, num_steps
    )
    
    # Print and save results
    print(f"\n{'='*60}")
    print("Compression ratios for each time step:")
    print(f"{'='*60}")
    for i, cr in enumerate(compression_ratios):
        print(f"  Step {i}: {cr:.2f}")
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'step': range(len(compression_ratios)),
        'compression_ratio': compression_ratios
    })
    output_csv = os.path.join(output_dir, "aurora_compression_all_variables.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    main()