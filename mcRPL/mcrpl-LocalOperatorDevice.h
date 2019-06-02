#ifndef LOCALOPERATORDEVICE_H_H
#define LOCALOPERATORDEVICE_H_H

#include "mcrpl-CuPRL.h"
//#include <device_functions.h>
#include"mcrpl-DevicePara.h"

//-------------------------------testblock--------------------------------------------
template<class DataType>
__global__ void G_sin(DataType *input, DataType *output, int width, int height)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	output[y_idx*width + x_idx] = sin(double(input[y_idx*width + x_idx]));
}
//-----------------------------------------------------------------------------------


//----device arithmetic function--------
class cucopy
{
public:
	__device__ void operator()(void **focalIn,char **DataInType,void **focalOut,char **DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo);
};
template<class DataInType,class DataOutType>
class CudaAbs
{
public:
	__device__ DataOutType operator()(DataInType value){ return fabs(double(value)); };
};


//----device power function-------------
template<class DataInType,class DataOutType>
class CudaSqr
{
public:
	__device__ DataOutType operator()(DataInType value){ return value*value; };
};

template<class DataInType,class DataOutType>
class CudaSqrt
{
public:
	__device__ DataOutType operator()(DataInType value){ return sqrt(double(value)); };
};

template<class DataInType,class DataOutType,class ParamType>
class CudaPow
{
public:
	__device__ DataOutType operator()(DataInType value, ParamType powvalue){ return pow(value, powvalue); };
};



//----device trigonometric function-----
template<class DataInType,class DataOutType>
class CudaSin
{
public:
	__device__ DataOutType operator()(DataInType value){ return sin(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaCos
{
public:
	__device__ DataInType operator()(DataOutType value){ return cos(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaTan
{
public:
	__device__ DataOutType operator()(DataInType value){ return tan(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaAsin
{
public:
	__device__ DataOutType operator()(DataInType value){ return asin(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaAcos
{
public:
	__device__ DataOutType operator()(DataInType value){ return acos(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaAtan
{
public:
	__device__ DataOutType operator()(DataInType value){ return atan(double(value)); };
};

//----device exponential function------
template<class DataInType,class DataOutType>
class CudaExp
{
public:
	__device__ DataOutType operator()(DataInType value){ return exp(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaExp2
{
public:
	__device__ DataOutType operator()(DataInType value){ return exp2(double(value)); };
};

//----device logarithmic function------

template<class DataInType,class DataOutType>
class CudaLog
{
public:
	__device__ DataOutType operator()(DataInType value){ return log(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaLog2
{
public:
	__device__ DataOutType operator()(DataInType value){ return log2(double(value)); };
};

template<class DataInType,class DataOutType>
class CudaLog10
{
public:
	__device__ DataOutType operator()(DataInType value){ return log10(double(value)); };
};

//-----device reclass function-------


template<class DataType1,class DataType2>
class ReclassValueUpdate
{
public:
	__device__ DataType2 operator()(DataType1 value, DataType1* oldValueSet, DataType2* newValueSet, int length)
	{
		for (int idx = 0; idx < length; idx++)
		{
			if (abs(oldValueSet[idx] - value) < 1e-6)
			{
				return newValueSet[idx];
			}
		}
		return value;
	}
};


template<class DataType1,class DataType2>
class ReclassRangeUpdate
{
public:
	__device__ DataType2 operator()(DataType1 value, DataType1* oldRangeSet, DataType2* newValueSet, int length)
	{
		for (int idx = 0; idx < length-1; idx++)
		{
			if (oldRangeSet[idx] < value&&value <= oldRangeSet[idx + 1])
			{
				return newValueSet[idx];
			}
		}
		return value;
	}
};



/*
user define class test code


*/
//template<class DataInType,class DataOutType>
class MultiRasterMean
{
public:
	__device__ float operator()(float* curIdx, int rasterSize, int numInRaster)
	{
		int layernum = numInRaster;
		float outvalue = 0;
		for (int i = 0; i < layernum; i++)
		{
			outvalue += *curIdx;
			curIdx += rasterSize;
		}
		outvalue /= numInRaster;

		return outvalue;
	}
};

 

//----global function template------------

template<class DataType, class Oper>
__global__ void G_SingleMathOper(DataType *input, DataType *output, int width, int height, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	output[y_idx*width + x_idx] = op(input[y_idx*width + x_idx]);
}


template<class DataType,class Oper,class ParamType>
__global__ void G_SingleMathOper(DataType *input, DataType *output, ParamType param, int width, int height, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	output[y_idx*width + x_idx] = op(input[y_idx*width + x_idx], param);
}





template<class DataType1,class DataType2,class Oper>
__global__ void G_Reclass(DataType1 *input, DataType2 *output, int width, int height, DataType1 *oldValueSet, DataType2 *newValueSet, int length,Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	output[y_idx*width + x_idx] = op(input[y_idx*width + x_idx], oldValueSet, newValueSet, length);

}



template<class DataInType,class DataOutType,class Oper>
__global__ void G_MultiStatic(DataInType *input, DataOutType* output, int width, int height, DataInType noDataValue, int numInRaster, double *paramInfo, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	int rasterSize = width*height;
	DataInType* cell = input + y_idx*width + x_idx;
	output[y_idx*width + x_idx] = op(cell, rasterSize, noDataValue, numInRaster);

}

template<class DataInType, class DataOutType, class Oper>
__global__ void G_MultiStatic(DataInType *input, DataOutType* output, int width, int height, DataInType noDataValue,int numInRaster, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	int rasterSize = width*height;
	output[y_idx*width + x_idx] = op(input + y_idx*width + x_idx, rasterSize, noDataValue,numInRaster);

}


/*----------------------------------------------------------------------------------------
//function class
template<class DataInType,class DataOutType>
class UserOperator
{
public:
	__device__ DataOutType operator()(DataInType value){};
};
----------------------------------------------------------------------------------------*/
template<class DataInType, class DataOutType, class Oper>
__global__ void G_LocalOper(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut,Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	if (isNoData(input[x_idx + y_idx*width], nodataIn))
		output[x_idx + y_idx*width] = nodataOut;
	else
		output[x_idx + y_idx*width] = op(input[x_idx + y_idx*width]);
}

/*----------------------------------------------------------------------------------------
//function class
template<class DataInType,class DataOutType>
class UserOperator
{
public:
__device__ DataOutType operator()(DataInType value,DataInType noDataIn,DataOutType noDataOut){};
};
----------------------------------------------------------------------------------------*/
template<class DataInType, class DataOutType, class Oper>
__global__ void G_LocalOperWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut,Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	output[x_idx + y_idx*width] = op(input[x_idx + y_idx*width], nodataIn, nodataOut);
}


/*----------------------------------------------------------------------------------------
//function class
template<class DataInType,class DataOutType,class ParamType>
class UserOperator
{
public:
__device__ DataOutType operator()(DataInType value,ParamType){};
};
----------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class ParamType,class Oper>
__global__ void G_LocalOperParam(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType param, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	if (isNoData(input[x_idx + y_idx*width], nodataIn))
		output[x_idx + y_idx*width] = nodataOut;
	else
		output[x_idx + y_idx*width] = op(input[x_idx + y_idx*width],param);
}

/*----------------------------------------------------------------------------------------
//function class
template<class DataInType,class DataOutType,class ParamType>
class UserOperator
{
public:
__device__ DataOutType operator()(DataInType value,DataInType noDataIn,DataOutType noDataOut,ParamType param){};
};
----------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class ParamType,class Oper>
__global__ void G_LocalOperParamWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType param, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	output[x_idx + y_idx*width] = op(input[x_idx + y_idx*width], nodataIn, nodataOut, param);
}


/*----------------------------------------------------------------------------------------------------------------------------
template<class DataInType,class DataOutType,class ParamType>
class UserOperator
{
public:
	__device__ DataOutType operator()(DataInType value,ParamType* params,int paramsnum){};
};
-----------------------------------------------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class ParamType,class Oper>
__global__ void G_LocalOperParams(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType* params, int paramsnum, Oper op)
{

	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	if (isNoData(input[x_idx + y_idx*width], nodataIn))
		output[x_idx + y_idx*width] = nodataOut;
	else
		output[x_idx + y_idx*width] = op(input[x_idx + y_idx*width], params, paramsnum);

}

/*------------------------------------------------------------------------------------------------------
template<class DataInType,class DataOutType,class ParamType>
class UserOperator
{
public:
	__device__ DataOutType operator()(DataInType value,DataInType noDataIn,DataOutType noDataOut,ParamType* params,int paramsnum){};
};
------------------------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class ParamType,class Oper>
__global__ void G_LocalOperParamsWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType* params,int paramsnum,Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;

	output[x_idx + y_idx*width] = op(input[x_idx + y_idx*width], nodataIn, nodataOut, params, paramsnum);
}

/*--------------------------------------------------------------------------------------
template<class DataInType,class DataOutType,class ParamType>
class UserOperator
{
public:
	__device__ DataOutType operator()(DataInType* value,int rasterSize,int numInRaster){};
};
---------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class Oper>
__global__ void G_LocalOperMultiLayers(DataInType* input, DataOutType* output, int width, int height, DataInType* nodataIn, DataOutType nodataOut, int numInRaster, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	int rasterSize = width*height;

	int idxLayer=0;
	for (idxLayer = 0; idxLayer < numInRaster; idxLayer++)
	{
		if (isNoData(input[y_idx*width + x_idx], nodataIn[idxLayer]))
		{
			output[y_idx*width + x_idx] = nodataOut;
			return;
		}
	}
	output[y_idx*width + x_idx] = op(input + y_idx*width + x_idx, rasterSize, numInRaster);
}


/*--------------------------------------------------------------------------------------
template<class DataInType,class DataOutType,class ParamType>
class UserOperator
{
public:
__device__ DataOutType operator()(DataInType* value,int rasterSize,DataInType noDataIn,DataOutType noDataOut,int numInRaster){};
};
---------------------------------------------------------------------------------------*/
template<class DataInType,class DataOutType,class Oper>
__global__ void G_LocalOperMultiLayersWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType* nodataIn, DataOutType *nodataOut, int numInRaster,int numOutRaster, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	int rasterSize = width*height;

	 output[y_idx*width + x_idx] = op(input + y_idx*width + x_idx, rasterSize, numInRaster, nodataIn, nodataOut);
}
template<class DataInType,class DataOutType,class Oper>
__global__ void G_LocalOperMultiLayersPaWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType* nodataIn, DataOutType *nodataOut, int numInRaster,int numOutRaster,double* params,int paramsnum, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	int rasterSize = width*height;

	op(input + y_idx*width + x_idx,output+ y_idx*width + x_idx, rasterSize, numInRaster, nodataIn, nodataOut, params, paramsnum);
}
//class cucopy
//{
//public:
//	__device__ void operator()(short *inValue,int *outValue,int rastersize,int numInRaster,short *nodataIn,int *nodataOut,double *params,int paramsum)
//	{
//		*outValue=*inValue;
//	}
//};

#endif