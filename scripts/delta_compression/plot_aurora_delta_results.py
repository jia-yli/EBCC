import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_compression_results(results_dir):
    """Load all compression ratio CSV files."""
    data = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('_compression_ratios.csv') and filename != 'overall_compression_ratios.csv':
            var_name = filename.replace('_compression_ratios.csv', '')
            df = pd.read_csv(os.path.join(results_dir, filename))
            data[var_name] = df['compression_ratio'].values
    
    # Load overall compression ratios
    overall_df = pd.read_csv(os.path.join(results_dir, 'overall_compression_ratios.csv'))
    overall_ratios = overall_df['compression_ratio'].values
    
    return data, overall_ratios

def plot_compression_by_category(data, output_path, title, var_names):
    """Plot compression ratios for a specific category of variables."""
    # Filter data for specified variables
    filtered_data = {k: v for k, v in data.items() if k in var_names}
    if not filtered_data:
        print(f"No data found for {title}")
        return
    
    num_steps = len(next(iter(filtered_data.values())))
    num_vars = len(filtered_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, num_steps * 1.5), 8))
    
    # Set bar width and positions
    bar_width = 0.8 / num_vars
    x = np.arange(num_steps)
    
    # Plot bars for each variable
    colors = plt.cm.tab20(np.linspace(0, 1, num_vars))
    
    for i, (var_name, ratios) in enumerate(sorted(filtered_data.items())):
        offset = (i - num_vars/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, ratios, bar_width, label=var_name, color=colors[i])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_steps)])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def plot_overall_compression(overall_ratios, output_path):
    """Plot overall compression ratios across all steps."""
    num_steps = len(overall_ratios)
    
    fig, ax = plt.subplots(figsize=(max(10, num_steps * 0.8), 6))
    
    x = np.arange(num_steps)
    bars = ax.bar(x, overall_ratios, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Overall Compression Ratio (All Variables)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_steps)])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    # Define paths
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, "results_aurora")
    output_dir = os.path.join(script_dir, "plots_aurora")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading compression results...")
    data, overall_ratios = load_compression_results(results_dir)
    
    # Define variable categories
    single_level_vars = ["t2m", "u10", "v10", "msl"]
    pressure_level_vars = ["temperature", "u_component_of_wind", "v_component_of_wind", 
                          "specific_humidity", "geopotential"]
    
    # Plot single level variables
    print("\nPlotting single level variables...")
    plot_compression_by_category(
        data,
        os.path.join(output_dir, "single_level_compression.png"),
        "Compression Ratios - Single Level Variables",
        single_level_vars
    )
    
    # Plot pressure level variables
    print("\nPlotting pressure level variables...")
    plot_compression_by_category(
        data,
        os.path.join(output_dir, "pressure_level_compression.png"),
        "Compression Ratios - Pressure Level Variables",
        pressure_level_vars
    )
    
    # Plot overall compression
    print("\nPlotting overall compression...")
    plot_overall_compression(
        overall_ratios,
        os.path.join(output_dir, "overall_compression.png")
    )
    
    print("\n" + "="*60)
    print("All plots saved successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
