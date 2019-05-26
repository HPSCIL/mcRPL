#ifndef ERRORHELPER_H_H
#define ERRORHELPER_H_H

#include <iostream>
#include <string>
#include <stdlib.h>
#include "CuPRL.h"
using namespace std;

namespace mcPRL
{
#define printError(msg) \
	{ \
		cerr <<"Error:"<<msg << " " << __FILE__ << ":" << __LINE__ << endl; \
	}

	inline void __checkCudaErrors(cudaError err, const char *file, const int line)
	{
		if (cudaSuccess != err)
		{
			cerr << file << " Line" << line <<" : CUDA Runtime API error" << int(err) << " : " << cudaGetErrorString(err) << endl;
			exit(EXIT_FAILURE);
		}
	}

#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
}








#endif