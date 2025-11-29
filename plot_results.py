#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results.csv'
df = pd.read_csv(csv_file)

# Create figure
plt.figure(figsize=(10, 6))

# Plot each vectorization width
vec_sizes = df['vec_size_bytes'].unique()
for vec_size in sorted(vec_sizes):
	data = df[df['vec_size_bytes'] == vec_size]
	label = f'{vec_size}B'
	plt.semilogx(data['buffer_size_bytes'], data['bandwidth_GB_per_s'], 
	             marker='o', label=label, linewidth=2, markersize=6)

# Formatting
plt.xlabel('Array Size (bytes)', fontsize=14)
plt.ylabel('Bandwidth (GB/s)', fontsize=14)
# plt.title('Memory Bandwidth vs Array Size', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save high quality PNG
plt.savefig('bandwidth_plot.png', dpi=300, bbox_inches='tight')
print('Plot saved: bandwidth_plot.png')

# plt.show()
