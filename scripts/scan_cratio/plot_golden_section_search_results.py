import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
df = pd.read_csv('/users/ljiayong/projects/EBCC/scripts/scan_cratio/results/golden_section_search_single_level_2024_10_1.csv')

# Define option names
round_option_names = {0: 'original', 1: 'no_mask_inside', 2: 'no_trunc'}
shuffle_option_names = {0: 'byte_shuffle', 1: 'bitshuffle'}
variables = df['variable'].unique()

# Define output directory
output_dir = '/users/ljiayong/projects/EBCC/scripts/scan_cratio/results'
os.makedirs(output_dir, exist_ok=True)

# For each value_shuffle_option (0 and 1), plot compression ratio vs variable
# Each variable has 3 bars for different round_options

for shuffle_opt in [0, 1]:
    fig2, ax = plt.subplots(1, 1, figsize=(9, 6))
    data_subset = df[df['value_shuffle_option'] == shuffle_opt]
    
    # Get unique round options
    round_options = sorted(data_subset['round_option'].unique())
    n_rounds = len(round_options)
    
    # Set up bar positions
    x = np.arange(len(variables))
    width = 0.25
    
    for i, round_opt in enumerate(round_options):
        round_data = data_subset[data_subset['round_option'] == round_opt]
        # Ensure data is in same order as variables
        values = [round_data[round_data['variable'] == var]['compression_ratio'].values[0] 
                  if len(round_data[round_data['variable'] == var]) > 0 else 0 
                  for var in variables]
        
        offset = (i - n_rounds/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=round_option_names[round_opt])
        
        # Add values on top of bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title(f'Compression Ratio by Variable ({shuffle_option_names[shuffle_opt]})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/compression_ratio_by_variable_s{shuffle_opt}.png', dpi=300, bbox_inches='tight')
    plt.close()