#include <stdio.h>
#include <stdlib.h>

#ifndef NumElem
#define NumElem 512
#endif


#include <sys/times.h>
#include <sys/resource.h>

float GetTime(void)        
{
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((float)tim.tv_sec + (float)tim.tv_usec / 1000000.0)*1000.0;
}

__global__ void Kernel05(double *g_idata, double *g_odata) {
  __shared__ double sdata[NumElem];
  unsigned int s;

  // Each thread loads 2 elements from global memory,
  // sums them and stores the result in shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
      __syncthreads();
    }

  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}


void InitV(int N, double *v);
int Test(int N, double *v, double sum, double *res);

int main(int argc, char** argv) {
  unsigned int N;
  unsigned int numBytesV, numBytesW, numBytesX;
  unsigned int nBlocks, nThreads;
  int test;
  float SeqTime, elapsedTime;
  float t1,t2; 

  cudaEvent_t start, stop;

  double *h_v, *h_x;
  double *d_v, *d_w, *d_x;

  double SUM, SumSeq;
  int i;
  int count, gpu, tmp;

  N = 1024 * 1024 * 16;
  nThreads = NumElem;  // This value must match the number of elements handled by the kernel

  // Maximum number of block threads = 65535
  nBlocks = N/nThreads;  // Only works well if N is a multiple of nThreads
  
  numBytesV = N * sizeof(double);
  numBytesW = nBlocks * sizeof(double);
  numBytesX = (nBlocks/nThreads) * sizeof(double);

  // Select GPU randomly
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  tmp = rand();
  gpu = (tmp>>3) % count;
  cudaSetDevice(gpu);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate memory on the host
  h_v = (double*) malloc(numBytesV);
  h_x = (double*) malloc(numBytesX);

  // Obtain [pinned] memory on the host
  //cudaMallocHost((double**)&h_v, numBytesV);
  //cudaMallocHost((double**)&h_x, numBytesX);

  // Initialize the vectors
  InitV(N, h_v);


  // Allocate memory on the device
  cudaMalloc((double**)&d_v, numBytesV);
  cudaMalloc((double**)&d_w, numBytesW);
  cudaMalloc((double**)&d_x, numBytesX);

  // Copy data from host to device
  cudaMemcpy(d_v, h_v, numBytesV, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

  nBlocks = nBlocks/2;
  // Run the kernel
  Kernel05<<<nBlocks, nThreads>>>(d_v, d_w);
  Kernel05<<<nBlocks/(2*nThreads), nThreads>>>(d_w, d_x);

  // Copy the partial result back to the host
  cudaMemcpy(h_x, d_x, numBytesX, cudaMemcpyDeviceToHost);


  SUM = 0.0;
  for (i=0; i<(nBlocks/(2*nThreads)); i++)
    SUM = SUM + h_x[i];


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Free device memory
  cudaFree(d_v);
  cudaFree(d_w);
 
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nKERNEL 05\n");
  printf("GPU used: %d\n", gpu);
  printf("Vector Size: %d\n", N);
  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);
  printf("Total Time %4.6f ms\n", elapsedTime);
  printf("Bandwidth %4.3f GB/s\n", (N * sizeof(double)) / (1000000 * elapsedTime));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  t1=GetTime();
  test = Test(N, h_v, SUM, &SumSeq);
  t2=GetTime();
  SeqTime = t2 - t1;
  printf("Speedup: x%2.3f \n", SeqTime/elapsedTime);

  if (test)
    printf ("TEST PASS, Time seq: %f ms\n", SeqTime);
  else {
    printf ("ERROR: %f(GPU) : %f(CPU) : %f(diff) : %f(error) \n", SumSeq, SUM, abs(SumSeq-SUM), abs(SumSeq - SUM)/SumSeq);
    printf ("TEST FAIL\n");
  }
}

