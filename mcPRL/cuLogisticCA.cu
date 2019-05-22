#include "OperatorDevice.h"
#include"LocalOperatorDevice.h"
#include<device_atomic_functions.h>
#include<sm_20_atomic_functions.h>
#include"cuLogisticCA.h"
__device__   float _maxJointProb=0;
__device__   float _sumDistDecayProb=0;
__device__ long _nToUrbanizePerYear;
__device__ unsigned long long _nUrbanized=0;
__device__ int *pAtest;
__device__ unsigned short *pMax;
__device__ unsigned short *pMin;
__device__ unsigned short minZoneID;
__device__ unsigned short maxZoneID;
__device__ long *pCount;
__device__ long *pTotals;
__device__  inline void atomicFloatAdd(float *address, float val)
{
      int i_val = __float_as_int(val);
      int tmp0 = 0;
      int tmp1;

      while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
      {
              tmp0 = tmp1;
              i_val = __float_as_int(val + __int_as_float(tmp1));
      }
}
#ifdef cuZonal
__device__ void cucopy::operator()(void **focalIn,char **DataInType,void **focalOut,char **DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
{
	unsigned short val=*((unsigned short*)focalIn[0]+nIndex);
	unsigned short zone=*((unsigned short*)focalIn[1]+nIndex);
	if(minZoneID > zone) 
	{
		minZoneID = zone;
		atomicExch(&minZoneID,zone);
	}
	if(maxZoneID < zone) 
	{
		atomicExch(&maxZoneID,zone);
	}
	if(val>pMax[zone])
	{
		atomicExch(pMax+zone,val);
	}
	if(val<pMin[zone])
	{
		atomicExch(pMin+zone,val);
	}
	atomicAdd(pCount+zone,1);
	atomicAdd(pTotals+zone,(long)val);
}
#endif
__device__ double SlopeMPI::operator()(short* focal, int x, int y,nbrInfo<double>nbrinfo,rasterInfo rasterinfo, float noDataValue)
	{
		if(threadIdx.x%25==0)
		{
			pAtest[10]=9;
			printf("%d",pAtest[8]);
		}
		int focalValue = *focal;
		if (focalValue == noDataValue)
			return noDataValue;
		int nbrsize = nbrinfo.nbrsize;
		int* nbrcood = nbrinfo.nbrcood;
	//	double* weights = nbrinfo.weights;
		int width = rasterinfo.width;
	//	int height = rasterinfo.height;
	//	double xz = 0;
		for(int i = 0; i < nbrsize; i++)
		{
			int curnbr = focal[nbrcood[i * 2] + nbrcood[i * 2 + 1] * width];
			if(noDataValue==curnbr)
			{
				float slopevalue=noDataValue;
				return slopevalue;
			}
		}
		float demDiff;
		 demDiff = static_cast<float>((focal[-1-width] + focal[-1] + focal[-1] + focal[-1+width]) -
                                 (focal[1-width] + focal[1] + focal[1]+ focal[1+width]));
		 float az = demDiff / (8.0 * fabs(rasterinfo.cellWidth));
		 demDiff = static_cast<float>((focal[-1+width] + focal[width] + focal[width] + focal[1+width]) -
                                 (focal[-1-width] + focal[-width] + focal[-width] + focal[1-width]));
    float bz = demDiff / (8.0 *fabs(rasterinfo.cellHeight));
	double slopevalue = atan(sqrt(az*az + bz*bz))*57.29578;
	if(slopevalue>4)
		return 0;
 return slopevalue;
	}
__device__ void jointProb:: operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{
	
		short *pUrban=(short*)focalIn[0]+nIndex;
		//float *pIncell;
		float *pGlbProb=(float*)focalIn[2]+nIndex;
		unsigned char *pExcluded=(unsigned char*)focalIn[1]+nIndex;
		float *pJointProb=(float*)focalOut[0]+nIndex;
		int nbrsize = nbrinfo.nbrsize;
		int* nbrcood = nbrinfo.nbrcood;
		int width = rasterinfo.width;
		int height = rasterinfo.height;
		int nurbanCount=0;
		unsigned char  excludedVal=*pExcluded;
		*pJointProb=CUDEFAULT_NODATA_FLOAT;
		if(fabs(*pGlbProb-CUDEFAULT_NODATA_FLOAT)>CUEPSION&&     
			*pUrban!=CUDEFAULT_NODATA_SHORT&&
			*pExcluded!=CUDEFAULT_NODATA_SHORT)
		{
			if(excludedVal==0)
			{
				excludedVal=1;
			}
			else
			{
				excludedVal=0;
			}
			for(int i = 0; i < nbrsize; i++)
			{
				int cx=nIndex%width+ nbrcood[i * 2];
				int cy=nIndex/width+nbrcood[i * 2 + 1];
				if (cx < 0 || cx >= width || cy < 0 || cy >= height)
				{
					continue;
				}
				
				if(nbrcood[i * 2]==0&&nbrcood[i * 2 + 1]==0)
                {
                   continue;
                 }
				int curnbr = pUrban[nbrcood[i * 2] + nbrcood[i * 2 + 1] * width];
				if(curnbr==1)
				{
					 nurbanCount++;
				}
			}
			*pJointProb = (float)nurbanCount/8.0;
			*pJointProb=excludedVal*(*pGlbProb)*(*pJointProb);
			if(*pJointProb>_maxJointProb)
			{
				atomicExch(&_maxJointProb,*pJointProb);
			}
		}
	}
__device__ void constarined::operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{
		short *pUrban=(short*)focalIn[0]+nIndex;
		float *pDist=(float*)focalIn[1]+nIndex;
		short *pNextVal=(short*)focalOut[0]+nIndex;
		if(*pUrban!=CUDEFAULT_NODATA_SHORT)
		{
			char nextVal=*pUrban;
			*pNextVal=nextVal;
			//float distDecayProbVal=*pDist;
			if(*pNextVal==0)
			{
				if(fabs(*pDist-CUDEFAULT_NODATA_FLOAT)>CUEPSION)
				{
				//	curandState state;
				//	curand_init(1234,nIndex,0,&state);
				//	float frandom=curand_uniform(&state);
					float frandom=*((float*)focalIn[2]+nIndex);
					float constarinedProVal=*pDist*_nToUrbanizePerYear/_sumDistDecayProb;
				//	printf("%f",frandom);
					if(constarinedProVal>frandom)
					//if(constarinedProVal>0.5)
					{
						*pNextVal=1;
						atomicAdd(&_nUrbanized,1);
					}
				}
			}
		}
	}
__device__ void DistDecayProb::operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{
		float *pDist=(float*)focalOut[0]+nIndex;
		float *pJointProb=(float*)focalIn[0]+nIndex;
		if(fabs(*pJointProb-CUDEFAULT_NODATA_FLOAT)>CUEPSION)
		{
			*pDist=*pJointProb*exp(-CUDISPERSION*(1-*pJointProb/_maxJointProb));
			atomicAdd(&_sumDistDecayProb,*pDist);
			
		}
	}

float jointMax()
{
	float nMax;
	cudaMemcpyFromSymbol(&nMax,_maxJointProb, sizeof(float));
	return nMax;
}
float distDecySum()
{
	float nSum;
	cudaMemcpyFromSymbol( &nSum,_sumDistDecayProb, sizeof(float));
	return nSum;
}
unsigned long long numUrbanized()
{
	unsigned long long nUrbanized;
	cudaMemcpyFromSymbol( &nUrbanized,_nUrbanized, sizeof(unsigned long long));
	return nUrbanized;
}
void resetSum()
{
	float nSum=0;
	cudaMemcpyToSymbol(_sumDistDecayProb, &nSum, sizeof(float));
}
void resetMax()
{
	float nMax=0;
	cudaMemcpyToSymbol(_maxJointProb, &nMax, sizeof(float));
}
void resetNumUrbanized()
{
	unsigned long long nSum=0;
	cudaMemcpyToSymbol(_nUrbanized, &nSum, sizeof(unsigned long long));
}
void maxJointProb(float maxVal)
{
	cudaMemcpyToSymbol(_maxJointProb, &maxVal, sizeof(float));
}
void sumDistDecayProb(float sumDistDecayProb)
{
	cudaMemcpyToSymbol(_sumDistDecayProb, &sumDistDecayProb, sizeof(float));
}
void convertLimit( long nToUrbanizePerYear)
{
	cudaMemcpyToSymbol(_nToUrbanizePerYear, &nToUrbanizePerYear, sizeof(long));
}
void devicePrt(int *p)
{
	cudaMemcpyToSymbol(pAtest, &p, sizeof(p));
}
