# GPU Memory Bandwidth Benchmark

Measure memory bandwidth across different vectorized loading strategies on NVIDIA GPUs.

## Quick Start
```bash
nvcc -O3 -std=c++17 -arch=sm_90 memcpy_benchmark.cu -o memcpy_benchmark.exe
./memcpy_benchmark.exe
python3 plot_results.py
```

For more options such as changing block size and output filename
```bash
./memcpy_benchmark.exe 512 benchmark_results_512.csv
python3 plot_results.py benchmark_results_512.csv
```

## NVIDIA GPU Bandwidth Plots

The figures below show the measured memory bandwidth on A100, H100, H200 GPUs.
All results use a block size of 1024 and evaluate vectorized load sizes from 1–16 bytes.

* **H100**
![H100 Bandwidth](figures/bandwidth_plot_blocksize1024_h100.png)

* **H200**
![H200 Bandwidth](figures/bandwidth_plot_blocksize1024_h200.png)

