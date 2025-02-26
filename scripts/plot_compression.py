import pandas as pd
import matplotlib.pyplot as plt

def plot_lossless_compression(variable, value_name, unit=None):
  csv_file = f'./results/{variable}_lossless_compression.csv'
  # Read CSV file
  df = pd.read_csv(csv_file)
  
  # Create bar chart
  plt.figure(figsize=(10, 6))
  # bars = plt.bar(df['scheme'], df[value_name], color='skyblue', width=0.5)
  bars = plt.bar(df['scheme'], df[value_name], color='blue', width=0.5)
  
  # Add values on top of bars
  for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10)
  
  # Labels and title
  if unit:
    plt.ylabel(f'{value_name} [{unit}]')
  else:
    plt.ylabel(value_name)
  plt.xticks(rotation=45, ha='right')
  plt.title(f'{variable} {value_name} vs Scheme')
  
  # Show plot
  plt.tight_layout()
  plt.grid(True)
  # Save the plot
  output_path = f"./results/{variable}_lossless_{value_name}.png"
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()

def plot_ebcc_uniform_compression(variable, value_name, unit=None):
  csv_file = f'./results/{variable}_ebcc_uniform_compression.csv'
  # Read CSV file
  df = pd.read_csv(csv_file)
  
  # Create bar chart
  plt.figure(figsize=(10, 6))

  for ebcc_max_error, group in df.groupby('ebcc_max_error'):
    plt.plot(group['ebcc_base_compression_ratio'], group[value_name], marker='o', linestyle='-', label=str(ebcc_max_error))

    for x, y in zip(group['ebcc_base_compression_ratio'], group[value_name]):
      plt.text(x, y, round(y, 2), ha='center', va='bottom', fontsize=10)
  
  # Labels and title
  plt.xscale('log')
  plt.xlabel('EBCC Base Compression Ratio')
  if unit:
    plt.ylabel(f'{value_name} [{unit}]')
  else:
    plt.ylabel(value_name)
  plt.title(f'{variable} {value_name} vs EBCC Base Compression Ratio')
  plt.legend(title='max_error')
  
  # Show plot
  plt.tight_layout()
  plt.grid(True)
  # Save the plot
  output_path = f"./results/{variable}_ebcc_uniform_compression_{value_name}.png"
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()


if __name__ == "__main__":
  variable_lst = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation"
  ]
  # global value
  for variable_idx in range(len(variable_lst)):
    variable = variable_lst[variable_idx]
    plot_lossless_compression(variable, 'compression_ratio')
    plot_lossless_compression(variable, 'compression_time', 's')
    plot_lossless_compression(variable, 'decompression_time', 's')

    plot_ebcc_uniform_compression(variable, 'compression_ratio')
    plot_ebcc_uniform_compression(variable, 'max_error')
    plot_ebcc_uniform_compression(variable, 'compression_time', 's')
    plot_ebcc_uniform_compression(variable, 'decompression_time', 's')
  
