#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

int main()
{
  cudaError_t error;
  cudaDeviceProp dev;
  int dev_cnt = 0;

  // return device numbers with compute capability >= 1.0
  error = cudaGetDeviceCount (&dev_cnt);
  if(error != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  printf("Number of devices: %d\n",dev_cnt);

  // Get properties of each device
  for(int i = 0; i < dev_cnt; i++)
  {
     error = cudaGetDeviceProperties(&dev, i);
     if(error != cudaSuccess)
     {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
     }
     printf("\nDevice %d:\n", i);
     printf("name: %s\n",dev.name);
     printf("Compute capability %d.%d\n",dev.major, dev.minor);
     printf("total global memory(KB): %ld\n", dev.totalGlobalMem/1024);
     printf("shared mem per block: %d\n",dev.sharedMemPerBlock);
     printf("regs per block: %d\n", dev.regsPerBlock);
     printf("warp size: %d\n", dev.warpSize);
     printf("max threads per block: %d\n",dev.maxThreadsPerBlock);
     printf("max thread dim z:%d y:%d x:%d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
     printf("max grid size z:%d y:%d x:%d\n", dev.maxGridSize[0],dev.maxGridSize[1], dev.maxGridSize[2]);
     printf("clock rate(KHz):\n",dev.clockRate);
     printf("total constant memory (bytes): %ld\n",dev.totalConstMem);
     printf("multiprocessor count %d\n",dev.multiProcessorCount);
     printf("integrated: %d\n",dev.integrated);
     printf("async engine count: %d\n",dev.asyncEngineCount);
     printf("memory bus width: %d\n",dev.memoryBusWidth);
     printf("memory clock rate (KHz): %d\n",dev.memoryClockRate);
     printf("L2 cache size (bytes): %d\n", dev.l2CacheSize);
     printf("max threads per SM: %d\n", dev.maxThreadsPerMultiProcessor);
     printf("Texture alignment: %d\n", dev.textureAlignment);
  }
}