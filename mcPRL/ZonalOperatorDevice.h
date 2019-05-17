#ifndef ZONALOPERATORDEVICE_H_H
#define ZONALOPERATORDEVICE_H_H

#include "CuPRL.h"
#include "DeviceParamType.h"


class ZonelAreaCal
{
public:
	__device__ double operator()(int* value, int x, int y, rasterInfo rasterinfo, int noDataValue)
	{
		if (*value == noDataValue)
			return noDataValue;
		double area = rasterinfo.cellHeight*rasterinfo.cellWidth;
		return area;
	}
};


template<class DataInType,class DataOutType,class Oper>
__global__ void G_ZonelStatisticSumRow(DataInType* input, DataOutType* output,double* zonelvalues,int zonelnum, int width, int height, double cellWidth, double cellHeight, DataInType noDataValue, Oper op)
{
	int y_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (y_idx >= height)
		return;

	rasterInfo rasterinfo;
	rasterinfo.cellHeight = cellHeight;
	rasterinfo.cellWidth = cellWidth;
	rasterinfo.height = height;
	rasterinfo.width = width;

	for (int idxW = 0; idxW < width; idxW++)
	{
		DataOutType rsl = op(input + idxW + y_idx*width, idxW, y_idx, rasterinfo, noDataValue);
		if (rsl == noDataValue)
			continue;
		for (int idxZonel = 0; idxZonel < zonelnum; idxZonel++)
		{
			if (fabs((zonelvalues[idxZonel]-input[idxW + y_idx*width]))<1e-6)
			{
				output[y_idx*zonelnum + idxZonel] += rsl;
			}
		}
	}
}
template<class DataType>
__global__ void G_ZonelStatisticSumCol(DataType* input,int width,int height)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;

	for (int idxH = 1; idxH < height; idxH++)
	{
		input[x_idx] += input[x_idx + idxH*width];
	}

}




#endif