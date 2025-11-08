#include <cub/cub.cuh>
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

  // Use CUB DeviceReduce to compute the sum entirely on device
  double *d_result = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
  // Query temporary storage size
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_v, d_result, N);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run sum-reduction
  CUDA_CHECK(cudaEventRecord(start, 0));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_v, d_result, N);

  // copy final result back to host
  CUDA_CHECK(
      cudaMemcpy(&SUM, d_result, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  // free temporary device storage
  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_result));
  // Free device memory
  CUDA_CHECK(cudaFree(d_v));

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nKERNEL 09\n");
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
  printf("CPU Time seq: %f ms\n", SeqTime);
  printf("GPU Time: %f ms\n", elapsedTime);
  printf("Speedup: x%2.3f \n", SeqTime / elapsedTime);

  if (test)
    printf("TEST PASS, Time seq: %f ms\n", SeqTime);
  else {
    printf("ERROR: %f(GPU) : %f(CPU) : %f(diff) : %f(error) \n", SumSeq, SUM,
           abs(SumSeq - SUM), abs(SumSeq - SUM) / SumSeq);
    printf("TEST FAIL\n");
  }
}
