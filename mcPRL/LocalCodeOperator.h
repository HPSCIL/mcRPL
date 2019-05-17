#ifndef LOCALCODEOPERATOR_H_H
#define LOCALCODEOPERATOR_H_H


#include "CuPRL.h"

namespace pRPL
{


	template<class DataType, class OperType>
	void SingleMathOper(DataType* input, DataType* output, int width, int height)
	{
		DataType* d_input;
		DataType* d_output;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataType)*width*height, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_SingleMathOper<DataType,OperType><<<grid,block>>>(d_input, d_output, width, height, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));

	}


	template<class DataType, class OperType, class ParamType>
	void SingleMathOper(DataType* input,DataType* output,ParamType param,int width,int height)
	{
		DataType* d_input;
		DataType* d_output;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataType)*width*height, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_SingleMathOper<DataType, OperType, ParamType> << <grid, block >> >(d_input, d_output, param, width, height, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
	}

	template<class DataType1,class DataType2,class OperType>
	void ReClass(DataType1* input, DataType2* output, int width, int height, DataType1* oldValueSet, DataType2* newValueSet, int length)
	{
		DataType1* d_input;
		DataType2* d_output;

		DataType1* d_oldValueSet;
		DataType2* d_newValueSet;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataType1)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataType2)*width*height));

		checkCudaErrors(cudaMalloc(&d_oldValueSet, sizeof(DataType1)*length));
		checkCudaErrors(cudaMalloc(&d_newValueSet, sizeof(DataType2)*length));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataType1)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_oldValueSet, oldValueSet, sizeof(DataType1)*length, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_newValueSet, newValueSet, sizeof(DataType2)*length, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_Reclass<DataType1, DataType2, OperType> << <grid, block >> >(d_input, d_output, width, height, d_oldValueSet, d_newValueSet, length, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataType2)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_oldValueSet));
		checkCudaErrors(cudaFree(d_newValueSet));

	}

	template<class DataInType,class DataOutType,class OperType>
	void cuLocalOperatorFn(DataInType** input, DataOutType* output, int width, int height, int numInRaster,DataInType noDataValue)
	{

		DataInType* d_input;
		DataOutType* d_output;
		


		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height*numInRaster));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
			


		for (int idxNumLayer = 0; idxNumLayer < numInRaster; idxNumLayer++)
		{
			checkCudaErrors(cudaMemcpy(d_input + width*height*idxNumLayer, input[idxNumLayer], sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		}
		

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);
		
		
		
		G_MultiStatic<DataInType, DataOutType, OperType><<<grid, block >>>(d_input, d_output, width, height, noDataValue, numInRaster, OperType());

		
		

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));


	}

	template<class DataInType, class DataOutType, class OperType>
	void cuLocalOperatorFn(DataInType** input, DataOutType* output, int width, int height, int numInRaster, DataInType noDataValue, double* paramInfo, double length)
	{

		DataInType* d_input;
		DataOutType* d_output;
		double* d_paramInfo;


		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height*numInRaster));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));

		if (length != 0)
		{
			checkCudaErrors(cudaMalloc(&d_paramInfo, sizeof(double)*length));
			checkCudaErrors(cudaMemcpy(d_paramInfo, paramInfo, sizeof(double)*length, cudaMemcpyHostToDevice));
		}
		else
		{
			printError("paramInfo's length is zero")
			return;
		}



		for (int idxNumLayer = 0; idxNumLayer < numInRaster; idxNumLayer++)
		{
			checkCudaErrors(cudaMemcpy(d_input + width*height*idxNumLayer, input[idxNumLayer], sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		}


		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);


		
		G_MultiStatic<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, noDataValue, numInRaster, d_paramInfo, OperType());


		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));

		checkCudaErrors(cudaFree(d_paramInfo));


	}

	//--------------------------------------------------------------------
	template<class DataInType,class DataOutType,class OperType>
	void LocalOper(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut)
	{

		DataInType* d_input;
		DataOutType* d_output;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOper<DataInType, DataOutType,OperType> << <grid, block >> >(d_input, d_output, width, height, nodataIn, nodataOut, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
	}

	template<class DataInType,class DataOutType,class OperType>
	void LocalOperWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut)
	{

		DataInType* d_input;
		DataOutType* d_output;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperWithNoData<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, nodataIn, nodataOut, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
	}

	template<class DataInType,class DataOutType,class ParamType,class OperType>
	void LocalOperParam(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType param)
	{
		DataInType* d_input;
		DataOutType* d_output;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperParam<DataInType, DataOutType, ParamType, OperType> << <grid, block >> >(d_input, d_output, width, height, nodataIn, nodataOut, param, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
	}

	template<class DataInType, class DataOutType, class ParamType, class OperType>
	void LocalOperParamWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType param)
	{
		DataInType* d_input;
		DataOutType* d_output;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperParamWithNoData<DataInType, DataOutType, ParamType, OperType> << <grid, block >> >(d_input, d_output, width, height, nodataIn, nodataOut, param, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
	}

	template<class DataInType, class DataOutType, class ParamType, class OperType>
	void LocalOperParams(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType params,int paramsnum)
	{
		DataInType* d_input;
		DataOutType* d_output;
		ParamType* d_params;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_params, sizeof(ParamType)*paramsnum));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_params, params, sizeof(ParamType)*paramsnum, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperParams<DataInType, DataOutType, ParamType, OperType> << <grid, block >> >(d_input, d_output, width, height, nodataIn, nodataOut, params, paramsnum, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_params));
	}

	template<class DataInType, class DataOutType, class ParamType, class OperType>
	void LocalOperParamsWithNoData(DataInType* input, DataOutType* output, int width, int height, DataInType nodataIn, DataOutType nodataOut, ParamType params, int paramsnum)
	{
		DataInType* d_input;
		DataOutType* d_output;
		ParamType* d_params;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_params, sizeof(ParamType)*paramsnum));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_params, params, sizeof(ParamType)*paramsnum, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperParamsWithNoData<DataInType, DataOutType, ParamType, OperType> << <grid, block >> >(d_input, d_output, width, height, nodataIn, nodataOut, params, paramsnum, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_params));
	}

	template<class DataInType, class DataOutType, class OperType>
	void LocalOperMultiLayers(DataInType** input, DataOutType* output, int width, int height, DataInType* pNodataIn, DataOutType nodataOut, int numInRaster)
	{

		DataInType* d_input;
		DataOutType* d_output;
		DataInType* d_nodataIn;
		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height*numInRaster));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nodataIn, sizeof(DataInType)*numInRaster));

		for (int idxNumLayer = 0; idxNumLayer < numInRaster; idxNumLayer++)
		{
			checkCudaErrors(cudaMemcpy(d_input + width*height*idxNumLayer, input[idxNumLayer], sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		}

		checkCudaErrors(cudaMemcpy(d_nodataIn, pNodataIn, sizeof(DataInType)*numInRaster, cudaMemcpyHostToDevice));


		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperMultiLayers<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nodataIn, nodataOut, numInRaster, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nodataIn));
	}

	template<class DataInType, class DataOutType, class OperType>
	void LocalOperMultiLayersWithNoData(DataInType** input, DataOutType* output, int width, int height, DataInType* pNodataIn, DataOutType nodataOut, int numInRaster)
	{

		DataInType* d_input;
		DataOutType* d_output;
		DataInType* d_nodataIn;
		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height*numInRaster));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nodataIn, sizeof(DataInType)*numInRaster));

		for (int idxNumLayer = 0; idxNumLayer < numInRaster; idxNumLayer++)
		{
			checkCudaErrors(cudaMemcpy(d_input + width*height*idxNumLayer, input[idxNumLayer], sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		}

		checkCudaErrors(cudaMemcpy(d_nodataIn, pNodataIn, sizeof(DataInType)*numInRaster, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_LocalOperMultiLayersWithNoData<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nodataIn, nodataOut, numInRaster, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nodataIn));
	}



}








#endif