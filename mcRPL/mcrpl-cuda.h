
#ifndef MCRPL_CUDA_H
#define MCRPL_CUDA_H
#include "mcrpl-basicTypes.h"
int InitCUDA(int myid)
{
   int devCount = 0;
int dev = 0;

   cudaGetDeviceCount(&devCount);
   if (devCount == 0) 
   {
      fprintf(stderr, "There is no device supporting CUDA.\n");
      return false;
   }

   for (dev = 0; dev < devCount; ++ dev)
   {
      cudaDeviceProp prop;

      if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
      {

         if (prop.major >= 1) break;
      }
	     
}

   if (dev == devCount)
{
     fprintf(stderr, "There is no device supporting CUDA.\n");
     return false;
   }

cudaSetDevice(myid);
 cudaDeviceProp prop1;
 cudaGetDeviceProperties(&prop1,myid);
fprintf(stdout, "设备上多处理器的数量: %d\n", prop1.multiProcessorCount);
   return true;
}
#endif