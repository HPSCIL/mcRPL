#ifndef CUSTARFM_H
#define CUSTARFM_H
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include"mcrpl-DevicePara.h"
#include"mcrpl-DeviceParamType.h"
class cuSTARFM
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para);
};
class testFocal
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{
		int nbrsize = nbrinfo.nbrsize;
		int* nbrcood = nbrinfo.nbrcood;
		int nwidth = rasterinfo.width;
		int nheight = rasterinfo.height;
		int nrow=nIndex/nwidth;
		int ncolumn=nIndex%nwidth;
		int nbrIndex=nIndex;
		short sum1=0;
		int sum2=0;
		long dim=nwidth*nheight;
		for(int i = 0; i < nbrsize; i++)
		{
			nbrIndex+=nbrcood[i * 2] + nbrcood[i * 2 + 1] * nwidth;
			sum1+=cuGetDataAs<int>(nbrIndex,0,focalIn,DataInType[0],dim);
			sum2+=cuGetDataAs<int>(nbrIndex,1,focalIn,DataInType[1],dim);
		}
		cuupdateCellAs<short>(nIndex,0,focalOut,DataOutType[0],dim,sum1);
		cuupdateCellAs<int>(nIndex,1,focalOut,DataOutType[1],dim,sum2);
	}
};
#endif