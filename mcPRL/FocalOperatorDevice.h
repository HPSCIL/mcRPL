#ifndef FOCALOPERATORDEVICE_H_H
#define FOCALOPERATORDEVICE_H_H

#include"DevicePara.h"
//#include "CuPRL.h"
#include "DeviceParamType.h"
#include <curand.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include"cuSTARFM.h"
#include"cuLogisticCA.h"
//extern __device__   float _maxJointProb;
//extern __device__   float _sumDistDecayProb;
//extern __device__ long _nToUrbanizePerYear;
//extern __device__ unsigned long long _nUrbanized;


template<class DataInType,class DataOutType>
class FocalStatisticsSum
{
public:
	__device__ DataOutType operator()(DataInType* input, int x, int y, int width, int height, int* nbrcoord, int nbrsize, int boundhandle,int nodatahandle)
	{
		DataOutType outvalue = 0;

		if (nodatahandle == 0)
		{
			for (int idxnbr = 0; idxnbr < nbrsize; idxnbr++)
			{

				int nbridx_x = x + nbrcoord[idxnbr * 2];
				int nbridx_y = y + nbrcoord[idxnbr * 2 + 1];

				if (nbridx_x >= 0 && nbridx_x < width && nbridx_y >= 0 && nbridx_y < height)
				{
					outvalue += input[nbridx_y*width + nbridx_x];
				}
				else
				{
					if (boundhandle == 0)
					{
						outvalue += input[y*width + x];
					}
				}
			}

		}
		return outvalue;
	}
};
template<class DataInType, class DataOutType>
class FocalStatisticsMean
{
public:
	__device__ DataOutType operator()(DataInType* input, int x, int y, int width, int height, int* nbrcoord, int nbrsize, int boundhandle,int nodatahandle)
	{
		DataOutType outvalue = 0;

		for (int idxnbr = 0; idxnbr < nbrsize; idxnbr++)
		{
			int nbridx_x = x + nbrcoord[idxnbr * 2];
			int nbridx_y = y + nbrcoord[idxnbr * 2 + 1];

			if (nbridx_x >= 0 && nbridx_x < width && nbridx_y >= 0 && nbridx_y < height)
			{
				outvalue += input[nbridx_y*width + nbridx_x];
			}
			else
			{
				if (boundhandle == 0)
				{
					outvalue += input[y*width + x];
				}
			}
		}
		outvalue /= nbrsize;
		return outvalue;
	}
};
template<class DataInType,class DataOutType>
class FocalStatisticsMaximum
{
public:
	__device__ DataOutType operator()(DataInType* input, int x, int y, int width, int height, int* nbrcoord, int nbrsize,int boundHandle,int nodatahandle)
	{

		DataOutType outvalue = input[y*width + x];

		for (int idxnbr = 0; idxnbr < nbrsize; idxnbr++)
		{

			int	nbridx_x = x + nbrcoord[idxnbr * 2];
			int nbridx_y = y + nbrcoord[idxnbr * 2 + 1];

			if (nbridx_x >= 0 && nbridx_x < width && nbridx_y >= 0 && nbridx_y < height)
			{
				if (outvalue < input[nbridx_y*width + nbridx_x])
					outvalue = input[nbridx_y*width + nbridx_x];
			}
		}
		return outvalue;
	}
};

template<class DataInType,class DataOutType>
class FocalStatisticsMinimum
{
public:
	__device__ DataOutType operator()(DataInType* input, int x, int y, int width, int height, int* nbrcoord, int nbrsize)
	{

		DataOutType outvalue = input[y*width + x];

		for (int idxnbr = 0; idxnbr < nbrsize; idxnbr++)
		{

			int	nbridx_x = x + nbrcoord[idxnbr * 2];
			int nbridx_y = y + nbrcoord[idxnbr * 2 + 1];

			if (nbridx_x >= 0 && nbridx_x < width && nbridx_y >= 0 && nbridx_y < height)
			{
				if (outvalue > input[nbridx_y*width + nbridx_x])
					outvalue = input[nbridx_y*width + nbridx_x];
			}
		}
		return outvalue;
	}
};

template<class DataInType,class DataOutType>
class FocalStatisticsRange
{
public:
	__device__ DataOutType operator()(DataInType* input, int x, int y, int width, int height, int* nbrcoord, int nbrsize,int boundHandle)
	{

		DataOutType minvalue = input[y*width + x];
		DataOutType maxvalue = input[y*width + x];

		for (int idxnbr = 0; idxnbr < nbrsize; idxnbr++)
		{

			int	nbridx_x = x + nbrcoord[idxnbr * 2];
			int nbridx_y = y + nbrcoord[idxnbr * 2 + 1];

			if (nbridx_x >= 0 && nbridx_x < width && nbridx_y >= 0 && nbridx_y < height)
			{
				if (maxvalue<input[nbridx_y*width + nbridx_x])
					maxvalue = input[nbridx_y*width + nbridx_x];

				if (minvalue > input[nbridx_y*width + nbridx_x])
					minvalue = input[nbridx_y*width + nbridx_x];
			}
		}
		return maxvalue - minvalue;
	}
};



class SlopeCal
{
public:
	__device__ double operator()(short* focal, int x, int y,nbrInfo<float>nbrinfo,rasterInfo rasterinfo, float noDataValue)
	{
		int focalValue = *focal;
		if (focalValue == noDataValue)
			return noDataValue;
		int nbrsize = nbrinfo.nbrsize;
		int* nbrcood = nbrinfo.nbrcood;
		float* weights = nbrinfo.weights;
		int width = rasterinfo.width;
		int height = rasterinfo.height;
		double xz = 0;
		for (int i = 0; i < nbrsize / 2; i++)
		{
			int cx = x + nbrcood[i * 2];
			int cy = y + nbrcood[i * 2 + 1];
			if (cx < 0 || cx >= width || cy < 0 || cy >= height)
			{
				xz += focalValue*weights[i];
				continue;
			}
			int curnbr = focal[nbrcood[i * 2] + nbrcood[i * 2 + 1] * width];
			if (curnbr == noDataValue)
				xz += focalValue*weights[i];
			else
				xz += curnbr*weights[i];
		}
		xz /= (nbrsize*rasterinfo.cellWidth / 2);
		double yz = 0;
		for (int i = nbrsize / 2; i < nbrsize; i++)
		{
			int cx = x + nbrcood[i * 2];
			int cy = y + nbrcood[i * 2 + 1];
			if (cx < 0 || cx >= width || cy < 0 || cy >= height)
			{
				yz += focalValue*weights[i];
				continue;
			}
			int curnbr = focal[nbrcood[i * 2] + nbrcood[i * 2 + 1] * width];
			if (curnbr == noDataValue)
				yz += focalValue*weights[i];
			else
				yz += curnbr*weights[i];
		}
		yz /= (nbrsize*rasterinfo.cellHeight / 2);
		double slopevalue = atan(sqrt(xz*xz + yz*yz))*57.29578;
		return slopevalue;
	}

};
class SlopeMPI
{
public:
	__device__ double operator()(short* focal, int x, int y,nbrInfo<double>nbrinfo,rasterInfo rasterinfo, float noDataValue);
};
//class SlopeMPI
//{
//public:
//	__device__ double operator()(short* focal, int x, int y,nbrInfo<float>nbrinfo,rasterInfo rasterinfo, float noDataValue)
//	{
//
//	}
//};
/*-------------------------------------------------------------------------
//focalÄ£°å
template<class DataInType,class DataOutType,class WeightType>
class FocalStatistics
{
public:
	__device__ DataOutType operator()(DataInType* focal, int x, int y, int width, int height, int* nbrcoord, int* weights, int nbrsize, int boundhandle)
	{

	}
};
---------------------------------------------------------------------------*/



template<class DataInType,class DataOutType,class WeightType,class Oper>
__global__ void G_FocalStatistics(DataInType *input, DataOutType *output, int width, int height, int *nbrcoord, WeightType *weights, int nbrsize, Oper op, int boundhandle)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;


	output[y_idx*width + x_idx] = op(input, x_idx, y_idx, width, height, nbrcoord, weights, nbrsize, boundhandle);
}


template<class DataType,class Oper>
__global__ void G_FocalStatistics(DataType *input, DataType *output, int width, int height, int *nbrcoord, int nbrsize, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;


	output[y_idx*width + x_idx] = op(input, x_idx, y_idx, width, height, nbrcoord, nbrsize);
}



template<class DataInType,class DataOutType, class Oper>
__global__ void G_FocalOperator(DataInType *input, DataOutType *output, int width, int height, int *nbrcoord, int nbrsize, DataInType noDataValue,double cellSize, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;


	output[y_idx*width + x_idx] = op(input, x_idx, y_idx, width, height, nbrcoord, nbrsize, noDataValue, cellSize);
}
template<class DataInType, class DataOutType,class WeightType, class Oper>
__global__ void G_FocalOperator(DataInType *input, DataOutType *output, int width, int height, int *nbrcoord, WeightType* weights,int nbrsize, DataInType noDataValue, double cellWidth,double cellHeight, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	nbrInfo<double>nbrinfo;
	nbrinfo.weights = weights;
	nbrinfo.nbrcood = nbrcoord;
	nbrinfo.nbrsize = nbrsize;

	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;


	output[y_idx*width + x_idx] = op(input + x_idx + y_idx*width, x_idx, y_idx, nbrinfo, rasterinfo, noDataValue);
}




/*--------------------------------------------------------------------------------------
template<class DataInType,class DataOutType,class Oper>
class UserOperator
{
public:
__device__ DataOutType operator()(DataInType* value,int cellX,int cellY,int *nbrcoord,int nbrsize,int boundHandle){};
};
---------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class Oper>
__global__ void G_FocalOper(DataInType *input, DataOutType *output, int width, int height, int *nbrcoords, int nbrsize, DataInType nodataIn, DataOutType nodataOut,int boundHandle,int nodataHandle, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	
	if (isNoData(input[y_idx*width + x_idx], nodataIn))
	{
		output[y_idx*width + x_idx] = nodataOut;
	}
	else
	{
		output[y_idx*width + x_idx] = op(input + x_idx + y_idx*width, x_idx, y_idx, width, height, nbrcoords, nbrsize, boundHandle, nodataHandle);
	}

}

template<class DataInType,class DataOutType,class WeightType,class Oper>
__global__ void G_FocalOperWeight(DataInType *input, DataOutType *output, int width, int height, int *nbrcoords, WeightType* weights, int nbrsize, DataInType nodataIn, DataOutType nodataOut, int boundHandle, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	nbrInfo<WeightType>nbrinfo;
	nbrinfo.weights = weights;
	nbrinfo.nbrcood = nbrcoords;
	nbrinfo.nbrsize = nbrsize;


	if (isNoData(input[y_idx*width + x_idx], nodataIn))
	{
		output[y_idx*width + x_idx] = nodataOut;
	}
	else
	{
		output[y_idx*width + x_idx] = op(input + x_idx + y_idx*width, x_idx, y_idx, width, height, nbrinfo, boundHandle);
	}
}

template<class DataInType,class DataOutType,class WeightType,class Oper>
__global__ void G_FocalOperWeight(DataInType *input, DataOutType *output, int width, int height, double cellWidth, double cellHeight, int *nbrcoords, WeightType* weights, int nbrsize, DataInType nodataIn, DataOutType nodataOut, int boundHandle, Oper op)
{

	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	nbrInfo<WeightType>nbrinfo;
	nbrinfo.weights = weights;
	nbrinfo.nbrcood = nbrcoords;
	nbrinfo.nbrsize = nbrsize;

	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;

	if (isNoData(input[y_idx*width + x_idx], nodataIn))
	{
		output[y_idx*width + x_idx] = nodataOut;
	}
	else
	{
		output[y_idx*width + x_idx] = op(input + x_idx + y_idx*width, x_idx, y_idx, nbrinfo, rasterinfo);
	}



}
template<class Oper>
__global__ void G_FocalMutiOperator (void **d_pDataIn,int *d_pDataInType,void **d_pDataOut,int *d_pDataOutType,int brminIRow,int brmaxIRow,int brminICol,int brmaxICol,int width, int height,int *nbrcoords,double *weights, int  nbrsize, double cellWidth,double cellHeight,double *para, Oper op)
{/*
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >brmaxICol||x_idx<brminICol)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >brmaxIRow||y_idx<brminIRow)
		return;*/
	int  x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	nbrInfo<double>nbrinfo;
	nbrinfo.weights = weights;
	nbrinfo.nbrcood = nbrcoords;
	nbrinfo.nbrsize = nbrsize;
	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;
	for(long long i=x_idx;i<width*height;i=i+blockDim.x*gridDim.x)
	{
		int row=i/width;
		int col=i%width;
		if (col >brmaxICol||col<brminICol)
		continue;
	    if (row>brmaxIRow||row<brminIRow)
		continue;
	op(d_pDataIn,d_pDataInType,d_pDataOut,d_pDataOutType,i, nbrinfo, rasterinfo,para);
	}
}
template<class Oper>
__global__ void G_LocalMutiOperator (void **d_pDataIn,int *d_pDataInType,void **d_pDataOut,int *d_pDataOutType,int width, int height, double cellWidth,double cellHeight,double *para, Oper op)
{/*
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >brmaxICol||x_idx<brminICol)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >brmaxIRow||y_idx<brminIRow)
		return;*/
	int  x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;
	for(long long i=x_idx;i<width*height;i=i+blockDim.x*gridDim.x)
	{
	op(d_pDataIn,d_pDataInType,d_pDataOut,d_pDataOutType,i, rasterinfo,para);
	}
}
template<class Oper>
__global__ void G_ZonalMutiOperator (void **d_pDataIn,int *d_pDataInType,void **d_pDataOut,int *d_pDataOutType,int width, int height, double cellWidth,double cellHeight,double *para, Oper op)
{/*
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >brmaxICol||x_idx<brminICol)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >brmaxIRow||y_idx<brminIRow)
		return;*/
	int  x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;
	for(long long i=x_idx;i<width*height;i=i+blockDim.x*gridDim.x)
	{
	op(d_pDataIn,d_pDataInType,d_pDataOut,d_pDataOutType,i, rasterinfo,para);
	}
}
class Copy
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,rasterInfo rasterinfo)
	{
		short *pB=(short*)focalIn[0]+nIndex;
		int width=rasterinfo.width;
		int height=rasterinfo.height;
		long dim=width*height;
		short pb=cuGetDataAs<short>(nIndex,0,focalIn,DataInType[0],dim);
		//short *pA=(short*)focalOut[0]+nIndex;
		//*pA=pb;
		cuupdateCellAs<short>(nIndex,0,focalOut,DataOutType[0],dim,pb);
	}
};
#endif