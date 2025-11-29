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
```

## H100 Bandwidth Plot at 256 block size
![H100 Bandwidth](bandwidth_plot_h100.png)
