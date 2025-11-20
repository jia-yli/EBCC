import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_all_variables_one_fig(df, ycol, ylabel, title, outpath):
  plt.figure(figsize=(8, 6))
  for var, g in df.groupby('variable'):
    plt.plot(g['cratio'], g[ycol], label=var)
  plt.xlabel('JPEG2000 cratio')
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend(fontsize=8)
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(outpath, dpi=500)
  plt.close()


def plot_per_variable_multi_lines(df, ycols, ylabel, title, outdir, fname_suffix):
  for var, g in df.groupby('variable'):
    plt.figure(figsize=(8, 6))
    for y in ycols:
      plt.plot(g['cratio'], g[y], label=y)
    plt.xlabel('JPEG2000 cratio')
    plt.ylabel(ylabel)
    plt.title(f"{var} {title}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(outdir, f"{var}_{fname_suffix}.png")
    plt.savefig(fname, dpi=500)
    plt.close()


def plot_sizes(df, outdir):
  # compressed_size_fail = compressed_size_fail_u16 + compressed_size_fail_fp32
  df = df.copy()
  df['compressed_size_fail'] = df['compressed_size_fail_u16'] + df['compressed_size_fail_fp32']
  # per variable plot: compressed_size, compressed_size_jp2k, compressed_size_fail
  plot_per_variable_multi_lines(
    df,
    ['compressed_size', 'compressed_size_jp2k', 'compressed_size_fail'],
    'Size [Byte]',
    'Compressed Sizes',
    outdir,
    'compressed_size'
  )
  # ratio compressed_size_fail / compressed_size across variables one figure
  df['compressed_size_fail_ratio'] = df['compressed_size_fail'] / df['compressed_size']
  outpath = os.path.join(outdir, 'ratio_compressed_failed_points_size.png')
  plot_all_variables_one_fig(
    df, 
    'compressed_size_fail_ratio', 
    'Ratio', 
    'Ratio of Compressed Failed Points in Final Bitstream', 
    outpath
  )


def plot_fail_ratio_jp2k(df, outdir):
  outpath = os.path.join(outdir, 'ratio_failed_points_count_jp2k.png')
  plot_all_variables_one_fig(
    df, 
    'fail_ratio_jp2k_hat', 
    'Ratio', 
    'Ratio of Failed Points in Data after JPEG2000 Compression', 
    outpath
  )


def plot_failed_points_compression_ratio(df, outdir):
  plot_per_variable_multi_lines(
    df, 
    [f'compression_ratio_u16_{i}' for i in range(1, 5)], 
    'Compression Ratio', 
    'Failed Points Compression Ratio', 
    outdir, 
    'failed_points_compression_ratio'
  )


def plot_overall_compression_ratio(df, outdir):
  outpath = os.path.join(outdir, 'compression_ratio.png')
  plot_all_variables_one_fig(
    df, 
    'compression_ratio', 
    'Compression Ratio',
    'Compression Ratio',
    outpath
  )


def plot_rmse(df, outdir):
  outpath = os.path.join(outdir, 'rmse.png')
  plot_all_variables_one_fig(df, 'rmse', 'Normalized RMSE', 'Normalized RMSE', outpath)


def main():
  ratio = 1
  df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'scan_cratio_single_level_2024_12_{ratio}.csv'))
  df = df.sort_values(['variable', 'cratio']).reset_index(drop=True)
  outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'plots_{ratio}')
  os.makedirs(outdir, exist_ok=True)

  plot_fail_ratio_jp2k(df, outdir)
  plot_sizes(df, outdir)
  plot_failed_points_compression_ratio(df, outdir)
  plot_overall_compression_ratio(df, outdir)
  plot_rmse(df, outdir)


if __name__ == '__main__':
  main()
