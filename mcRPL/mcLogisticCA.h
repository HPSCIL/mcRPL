#ifndef CULOGISTICCA_H
#define CULOGISTICCA_H
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include"mcrpl-DevicePara.h"
#include"mcrpl-DeviceParamType.h"
static const char* _aInputNames[5] = {
	"LANDUSE",
	"ELEV",
	"SLOPE",
	"DIST2CITYCTR",
	"DIST2TRANSP"
};
__device__ static const float _aFactorCoeffs[5] = {
	6.4640,   // constant
	43.5404,  // elev
	1.9150,   // slope
	41.3441,  // dist2cityctr
	12.5878   // dist2transp
};

__device__ static const float _aLandUseCoeffs[9] = {0.0,     // water
	0.0,     // urban
	-9.8655, // barren
	-8.7469, // forest
	-9.2688, // shrubland
	-8.0321, // woody
	-9.1693, // herbaceous
	-8.9420, // cultivated
	-9.4500  // wetland
};
class cuglbProb
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{
		int *pLU=(int*)focalIn[0]+nIndex;
		float *pIncell;
		float *glbProVal=(float*)focalOut[0]+nIndex;
		int landuseVal=*pLU;
		if(landuseVal!=CUDEFAULT_NODATA_INT&&landuseVal<9)
		{
			
			if(landuseVal<=2)
			{
				*glbProVal=0.0;
			}
			else
			{
				float z=_aFactorCoeffs[0]+_aLandUseCoeffs[landuseVal-1];
				//float z=_aFactorCoeffs[0];
				for(int i=1;i<5;i++)
				{
					pIncell=(float*)(focalIn[i])+nIndex;
					if(fabs(*pIncell-CUDEFAULT_NODATA_FLOAT)<=1e-6)
					{
						z=-9999;
						break;
					}
					else
					{
						z+=*pIncell*_aFactorCoeffs[i];
					}
				}
				if(fabs(z+9999)>1e-6)
				{
					*glbProVal=1.0/(1.0+exp(z));
				}
			}
		}
		
	}
};
class jointProb
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para);
};
class cuTransition
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{

	}
};
class constarined
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para);
};
class DistDecayProb
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para);
	

};
void resetMax();
float jointMax();
void resetSum();
float distDecySum();
void maxJointProb(float maxVal);
void resetNumUrbanized();
void sumDistDecayProb(float sumDistDecayProb);
void convertLimit( long nToUrbanizePerYear);
unsigned long long numUrbanized();
void devicePrt(int *p);
#endif