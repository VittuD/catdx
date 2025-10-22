import pandas as pd
import os

# Paths
input_csv = '/scratch/catdx/all_files_with_partition.csv'
output_csv = '/scratch/catdx/all_files_with_partition_corrected.csv'

# Read the original CSV (skip header) assuming 4 columns per line
df_orig = pd.read_csv(input_csv, header=None, skiprows=1, names=['file_name', 'CO', 'shape1', 'shape2'])

# Extract partition from the path (element after the third '/')
df_orig['partition'] = df_orig['file_name'].apply(lambda x: x.split('/')[3] if isinstance(x, str) else '')

# Keep only the necessary columns
df_corrected = df_orig[['file_name', 'CO', 'partition']]

# Strip away /workspace/RVENetCropRsz/ from file_name
df_corrected['file_name'] = df_corrected['file_name'].apply(lambda x: x.replace('/workspace/RVENetCropRsz/', '') if isinstance(x, str) else x)

# Save corrected CSV
df_corrected.to_csv(output_csv, index=False)

# Provide the download link
output_csv
