#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

// #define CUDA_CHECK(call)

#ifndef NumElem
#define NumElem 512
#endif

#include <sys/resource.h>
#include <sys/times.h>

float GetTime(void) {
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim = ru.ru_utime;
  return ((float)tim.tv_sec + (float)tim.tv_usec / 1000000.0) * 1000.0;
}


void InitV(int N, double *v);
int Test(int N, double *v, double sum, double *res);

// Block-reduce kernel: each block reduces a strided subset of the input and writes one per-block sum.
constexpr int BLOCK_SIZE = 512;
__global__ void block_reduce_kernel(const double *d_v, double *d_block_sums, int N) {
  typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  double thread_sum = 0.0;
  // Stride across the array to cover all elements
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < N; i += stride) {
    thread_sum += d_v[i];
  }
  double block_sum = BlockReduceT(temp_storage).Sum(thread_sum);
  if (tid == 0) {
    d_block_sums[bid] = block_sum;
  }
}

int main() {
  unsigned int N;
  unsigned int numBytesV;
  unsigned int nBlocks, nThreads;
  int test;
  float SeqTime, elapsedTime;
  float t1, t2;

  cudaEvent_t start, stop;

  double *h_v;
  double *d_v;

  double SUM, SumSeq;
  int count, gpu, tmp;

  N = 1024 * 1024 * 16;

  nThreads = NumElem;  // This value must match the number of elements handled by the kernel

  // Maximum number of block threads = 65535
  nBlocks = NumElem * 2;

  numBytesV = N * sizeof(double);

  // Select GPU randomly
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  tmp = rand();
  gpu = (tmp >> 3) % count;
  cudaSetDevice(gpu);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Use pinned host memory for faster H2D copies
  cudaMallocHost((double **)&h_v, numBytesV);
  // h_v = (double *)malloc(numBytesV);

  // Initialize the vectors
  InitV(N, h_v);

  // Allocate memory on the device
  cudaMalloc((double **)&d_v, numBytesV);

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_v, h_v, numBytesV, cudaMemcpyHostToDevice));

  // Determine kernel launch params
  nThreads = BLOCK_SIZE;
  // Limit number of blocks to a reasonable upper bound (e.g., 1024) to avoid overly large intermediate array
  unsigned int maxBlocks = 1024;
  nBlocks = (N + nThreads - 1) / nThreads;
  if (nBlocks > maxBlocks) nBlocks = maxBlocks;

  // Allocate per-block sums array
  double *d_block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_block_sums, nBlocks * sizeof(double)));

  // Launch block-reduce kernel and then finalize with DeviceReduce on the small per-block array.
  // Time the whole operation (kernel + final device reduce)
  CUDA_CHECK(cudaEventRecord(start, 0));

  block_reduce_kernel<<<nBlocks, nThreads>>>(d_v, d_block_sums, N);
  CUDA_CHECK(cudaGetLastError());

  // Use CUB DeviceReduce::Sum to reduce the small per-block array to a single double
  double *d_result = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));

  // Query temporary storage size for reducing nBlocks elements
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_block_sums, d_result, nBlocks);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run final reduce
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_block_sums, d_result, nBlocks);

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // copy final result back to host
  CUDA_CHECK(cudaMemcpy(&SUM, d_result, sizeof(double), cudaMemcpyDeviceToHost));

  // free temporary device storage
  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_result));
  CUDA_CHECK(cudaFree(d_block_sums));

  // Free device memory
  cudaFree(d_v);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nBLOCK-REDUCE VERSION\n");
  printf("GPU used: %d\n", gpu);
  printf("Vector Size: %d\n", N);
  printf("Total Time %4.6f ms\n", elapsedTime);
  printf("Bandwidth %4.3f GB/s\n",
         (N * sizeof(double)) / (1000000 * elapsedTime));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  t1 = GetTime();
  test = Test(N, h_v, SUM, &SumSeq);
  t2 = GetTime();
  SeqTime = t2 - t1;
  printf("Speedup: x%2.3f \n", SeqTime / elapsedTime);

  if (test)
    printf("TEST PASS, Time seq: %f ms\n", SeqTime);
  else {
    printf("ERROR: %f(GPU) : %f(CPU) : %f(diff) : %f(error) \n", SumSeq, SUM,
           abs(SumSeq - SUM), abs(SumSeq - SUM) / SumSeq);
    printf("TEST FAIL\n");
  }

  // Free host memory
  cudaFreeHost(h_v);
  return 0;
}
