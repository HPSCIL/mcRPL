#ifndef GLOBALOPERATORDEVICE_H_H
#define GLOBALOPERATORDEVICE_H_H


#include "CuPRL.h"

template<class DataInType,class DataOutType>
class GlobalEucDistance
{
public:
	__device__ DataOutType operator()(int value,int x,int y,int *srcCoord,int srcSize,double cellSize)
	{
		DataOutType mindist = 0;
		for (int idxsrc = 0; idxsrc < srcSize; idxsrc++)
		{
			int dist_x = srcCoord[idxsrc * 2] - x;
			int dist_y = srcCoord[idxsrc * 2 + 1] - y;
			DataOutType distance = sqrt(dist_x*dist_x + dist_y*dist_y);
			if (distance < mindist)
			{
				mindist = distance;
			}
		}

		return mindist*cellSize;
	}
};

/*
template<class DataOutType>
class GlobalEucDirection
{
public:
	__device__ DataOutType operator()(int x, int y, int *srcCoord, int srcSize, double cellSize)
	{
		DataOutType mindist = 0;
		int mindist_x = 0;
		int mindist_y = 0;
		for (int idxsrc = 0; idxsrc < srcSize; idxsrc++)
		{
			int dist_x = srcCoord[idxsrc * 2] - x;
			int dist_y = srcCoord[idxsrc * 2 + 1] - y;
			DataOutType distance = sqrt(dist_x*dist_x + dist_y*dist_y);
			if (distance < mindist)
			{
				mindist = distance;
				mindist_x = dist_x;
				mindist_y = dist_y;
			}
		}

		if (mindist_x == 0 && mindist_y == 0)
		{

		}


		return mindist*cellSize;
	}
};
*/


class EucAlloCal
{
public:
	__device__ int operator()(int* input, int x, int y, rasterInfo rasterinfo, int noDataValue,rasterCell* pRasterCell,int cellnum)
	{
		int width = rasterinfo.width;
		int height = rasterinfo.height;

		if (*input != noDataValue)
		{
			return *input;
		}

		double mindist = pow(width*rasterinfo.cellWidth, 2) + pow(height*rasterinfo.cellHeight, 2);
		int minzonel = noDataValue;
		
		for (int i = 0; i < cellnum; i++)
		{
			double x_dist = abs(x - pRasterCell[i].x) * rasterinfo.cellWidth;
			double y_dist = abs(y - pRasterCell[i].y) * rasterinfo.cellHeight;
			double dist = sqrt(x_dist*x_dist + y_dist*y_dist);
			if (dist < mindist)
			{
				mindist = dist;
				minzonel = pRasterCell[i].value;
			}
		}
		return minzonel;

	}
};



template<class DataOutType,class Oper>
__global__ void G_GlobalEucCal(DataOutType *output, int* srcCoord,int srcSize, int width, int height,double cellSize,Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	output[y_idx*width + x_idx] = op(x_idx, y_idx, srcCoord, srcSize, cellSize);

}


template<class DataInType,class DataOutType, class Oper>
__global__ void G_GlobalEucCal(DataInType *input,DataOutType *output, int width, int height, double cellWidth,double cellHeight,DataInType noDataValue,Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	
	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;

	output[y_idx*width + x_idx] = op(input + y_idx*width + x_idx, x_idx, y_idx, rasterinfo, noDataValue);

}

template<class DataInType, class DataOutType, class Oper>
__global__ void G_GlobalEucCal(DataInType *input, DataOutType *output, int width, int height, double cellWidth, double cellHeight, DataInType noDataValue,rasterCell* pRasterCell,int cellnum, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;

	output[y_idx*width + x_idx] = op(input + y_idx*width + x_idx, x_idx, y_idx, rasterinfo, noDataValue, pRasterCell, cellnum);

}





#endif