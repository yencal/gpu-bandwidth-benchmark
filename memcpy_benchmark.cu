#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdlib>

// ============================================================================
// Error Checking
// ============================================================================

#define CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line
		          << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK_LAST_CUDA() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
{
	cudaError_t const err{cudaGetLastError()};
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line
		          << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// ============================================================================
// Kernel
// ============================================================================

template <typename T, typename VecT>
__global__ void memcpy_kernel(
	T* __restrict__ dst,
	const T* __restrict__ src,
	size_t n_elements)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	size_t total_bytes = n_elements * sizeof(T);
	size_t n_vecs = total_bytes / sizeof(VecT);

	// Phase 1: Vectorized copy
	VecT* dst_vec = reinterpret_cast<VecT*>(dst);
	const VecT* src_vec = reinterpret_cast<const VecT*>(src);

	for (size_t i = idx; i < n_vecs; i += stride)
	{
		dst_vec[i] = src_vec[i];
	}

  // Phase 2: Handle remainder elements
	size_t remainder_start_element = (n_vecs * sizeof(VecT)) / sizeof(T);
	for (size_t i = remainder_start_element + idx; i < n_elements; i += stride)
	{
  	dst[i] = src[i];
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

template <typename T>
void initialize_buffer(T* buffer, size_t n)
{
	for (size_t i = 0; i < n; ++i)
	{
		buffer[i] = static_cast<T>(
			i % static_cast<size_t>(std::numeric_limits<T>::max()));
	}
}

template <typename T>
void verify_buffer(const T* buffer, size_t n)
{
	for (size_t i = 0; i < n; ++i)
	{
		T value = static_cast<T>(
			i % static_cast<size_t>(std::numeric_limits<T>::max()));
		if (buffer[i] != value)
		{
			std::cerr << "Verification failed at index: " << i << std::endl;
			std::cerr << "Expected: " << static_cast<int>(value) 
			          << ". Got: " << static_cast<int>(buffer[i]) << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult
{
	size_t unit_size_bytes;
	size_t vec_size_bytes;
	size_t buffer_size_bytes;
	size_t threads_per_block;
	float latency_ms;
	float bandwidth_GB_per_s;
};

template <typename T, typename VecT>
BenchmarkResult measure_memcpy(
	size_t n_elements,
	T* d_src,
	T* d_dst,
	T* h_dst,
	size_t block_size,
	cudaStream_t& stream,
	size_t num_warmup = 20,
	size_t num_iterations = 20)
{
	size_t size_bytes = n_elements * sizeof(T);
	size_t n_vecs = size_bytes / sizeof(VecT);

	int device_id;
	int max_blocks_x;
	CHECK_CUDA(cudaGetDevice(&device_id));
	CHECK_CUDA(cudaDeviceGetAttribute(&max_blocks_x, cudaDevAttrMaxGridDimX, device_id));

	dim3 threads_per_block{static_cast<unsigned int>(block_size)};
	dim3 blocks_per_grid{static_cast<unsigned int>(
		std::min((n_vecs + (threads_per_block.x - 1)) / threads_per_block.x,
		         static_cast<size_t>(max_blocks_x)))};

	// Verify correctness first
	CHECK_CUDA(cudaMemset(d_dst, 0, size_bytes));
	memcpy_kernel<T, VecT><<<blocks_per_grid, threads_per_block, 0, stream>>>(
		d_dst, d_src, n_elements);
	CHECK_LAST_CUDA();
	CHECK_CUDA(cudaMemcpy(h_dst, d_dst, size_bytes, cudaMemcpyDeviceToHost));
	verify_buffer<T>(h_dst, n_elements);

	// Warmup runs
	for (size_t i = 0; i < num_warmup; ++i)
	{
		memcpy_kernel<T, VecT><<<blocks_per_grid, threads_per_block, 0, stream>>>(
			d_dst, d_src, n_elements);
	}
	CHECK_CUDA(cudaStreamSynchronize(stream));

	// Timed runs
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	CHECK_CUDA(cudaEventRecord(start, stream));
	for (size_t i = 0; i < num_iterations; ++i)
	{
		memcpy_kernel<T, VecT><<<blocks_per_grid, threads_per_block, 0, stream>>>(
			d_dst, d_src, n_elements);
	}
	CHECK_CUDA(cudaEventRecord(stop, stream));
	CHECK_CUDA(cudaEventSynchronize(stop));

	float total_ms;
	CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));

	BenchmarkResult result;
	result.unit_size_bytes = sizeof(T);
	result.vec_size_bytes = sizeof(VecT);
	result.buffer_size_bytes = size_bytes;
	result.threads_per_block = block_size;
	result.latency_ms = total_ms / num_iterations;
	// Bandwidth includes read + write (factor of 2)
	result.bandwidth_GB_per_s = (2.0f * size_bytes / 1e9) / (result.latency_ms / 1000.0f);
	return result;
}

std::vector<BenchmarkResult> run_benchmark(size_t block_size)
{
	using T = int8_t;

	// Create stream
	cudaStream_t stream;
	CHECK_CUDA(cudaStreamCreate(&stream));

	std::vector<BenchmarkResult> results;

	std::cout << "Running benchmarks with block size: " << block_size << std::endl;
	std::cout << "Progress: " << std::flush;

	for (size_t power = 8; power <= 27; ++power)
	{
		size_t n_elements = 1ULL << power;
		size_t size_bytes = n_elements * sizeof(T);

		// Allocate host memory
		T* h_src = new T[n_elements];
		T* h_dst = new T[n_elements];
		initialize_buffer<T>(h_src, n_elements);

		// Allocate device memory
		T* d_src;
		T* d_dst;
		CHECK_CUDA(cudaMalloc(&d_src, size_bytes));
		CHECK_CUDA(cudaMalloc(&d_dst, size_bytes));
		CHECK_CUDA(cudaMemcpy(d_src, h_src, size_bytes, cudaMemcpyHostToDevice));

		// Test different vectorization widths
		results.push_back(measure_memcpy<T, uint8_t>(
			n_elements, d_src, d_dst, h_dst, block_size, stream));
		results.push_back(measure_memcpy<T, uint16_t>(
			n_elements, d_src, d_dst, h_dst, block_size, stream));
		results.push_back(measure_memcpy<T, uint32_t>(
			n_elements, d_src, d_dst, h_dst, block_size, stream));
		results.push_back(measure_memcpy<T, uint64_t>(
			n_elements, d_src, d_dst, h_dst, block_size, stream));
		results.push_back(measure_memcpy<T, uint4>(
			n_elements, d_src, d_dst, h_dst, block_size, stream));

		// Cleanup
		delete[] h_src;
		delete[] h_dst;
		CHECK_CUDA(cudaFree(d_src));
		CHECK_CUDA(cudaFree(d_dst));

		std::cout << "." << std::flush;
	}

	std::cout << " Done!" << std::endl;

	// Destroy stream
	CHECK_CUDA(cudaStreamDestroy(stream));

	return results;
}

void write_csv(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
	std::ofstream csv(filename);
	if (!csv)
	{
		std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Write header
	csv << "unit_size_bytes,vec_size_bytes,buffer_size_bytes,threads_per_block,"
	    << "latency_ms,bandwidth_GB_per_s" << std::endl;

	// Write results
	for (const auto& r : results)
	{
		csv << r.unit_size_bytes << ","
		    << r.vec_size_bytes << ","
		    << r.buffer_size_bytes << ","
		    << r.threads_per_block << ","
		    << r.latency_ms << ","
		    << r.bandwidth_GB_per_s << std::endl;
	}

	csv.close();
	std::cout << "Results written to: " << filename << std::endl;
}

void print_device_info()
{
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

	std::cout << "\n=== GPU Information ===" << std::endl;
	std::cout << "Device: " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Global Memory: " << prop.totalGlobalMem / (1ULL << 30) << " GB" << std::endl;
	std::cout << "L2 Cache: " << prop.l2CacheSize / (1ULL << 20) << " MB" << std::endl;
	std::cout << "Memory Clock: " << prop.memoryClockRate / 1000.0f << " MHz" << std::endl;
	std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;

	// Calculate theoretical peak memory bandwidth
	float memory_clock_mhz = prop.memoryClockRate / 1000.0f;
	float memory_bus_width = prop.memoryBusWidth;
	float peak_bandwidth_gb_per_s = 2.0f * memory_clock_mhz * (memory_bus_width / 8.0f) / 1000.0f;
	std::cout << "Theoretical Peak Bandwidth: " << peak_bandwidth_gb_per_s << " GB/s" << std::endl;
	std::cout << std::endl;
}

int main(int argc, char** argv)
{
	// Parse command line arguments
	size_t block_size = 256;  // default
	std::string csv_filename = "benchmark_results.csv";

	if (argc > 1)
	{
		block_size = std::atoi(argv[1]);
		if (block_size == 0 || block_size > 1024)
		{
			std::cerr << "Error: Block size must be between 1 and 1024" << std::endl;
			std::cerr << "Usage: " << argv[0] << " [block_size] [output.csv]" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (argc > 2)
	{
		csv_filename = argv[2];
	}

	// Print GPU information
	print_device_info();

	// Run benchmark
	auto results = run_benchmark(block_size);

	// Write results to CSV
	write_csv(results, csv_filename);

	std::cout << "\nBenchmark complete!" << std::endl;
	std::cout << "Total tests: " << results.size() << std::endl;

	return EXIT_SUCCESS;
}
