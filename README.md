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

## H100 Bandwidth Plot

The figure below shows the measured memory bandwidth on an H100 GPU using a block size of 256.  
Several vectorized loading strategies were evaluated, with element sizes ranging from 1 byte up to 16 bytes.

![H100 Bandwidth](bandwidth_plot_h100.png)

