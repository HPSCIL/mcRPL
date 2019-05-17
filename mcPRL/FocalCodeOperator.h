#ifndef FOCALCODEOPERATOR_H_H
#define FOCALCODEOPERATOR_H_H

//#include "CuPRL.h"
#include "FocalOperatorDevice.h"


namespace pRPL
{
	template<class DataInType, class DataOutType, class WeightType, class OperType>
	void FocalStatistics(DataInType* input, DataOutType* output, int width, int height, int *nbrcoord, WeightType *weights, int nbrsize, int boundhandle)
	{
		DataInType* d_input;
		DataOutType* d_output;

		int* d_nbrcoord;
		WeightType* d_weights;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nbrcoord, sizeof(int)*nbrsize * 2));
		checkCudaErrors(cudaMalloc(&d_weights, sizeof(WeightType)*nbrsize));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nbrcoord, nbrcoord, sizeof(int)*nbrsize * 2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_weights, weights, sizeof(WeightType)*nbrsize, cudaMemcpyHostToDevice));

//		dim3 block = cupCuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_FocalStatistics<DataInType, DataOutType, WeightType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nbrcoord, d_weights, nbrsize, OperType(), boundhandle);

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nbrcoord));
		checkCudaErrors(cudaFree(d_weights));


	}
	template<class DataType, class OperType>
	void FocalStatistics(DataType* input, DataType* output, int width, int height, int *nbrcoord, int nbrsize)
	{
		DataType* d_input;
		DataType* d_output;
		int* d_nbrcoord;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nbrcoord, sizeof(int)*nbrsize * 2));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nbrcoord, nbrcoord, sizeof(int)*nbrsize * 2, cudaMemcpyHostToDevice));
		

		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_FocalStatistics<DataType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nbrcoord, nbrsize, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nbrcoord));
	}


	template<class DataInType,class DataOutType,class WeightType, class OperType>
	void cuFocalOperatorFn(DataInType* input, DataOutType* output, int width, int height, int *nbrcoord, WeightType* weights, int nbrsize, double cellWidth, double cellHeight, DataInType noDataValue)
	{
		DataInType* d_input;
		DataOutType* d_output;
		int* d_nbrcoord;
		int* d_weights;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nbrcoord, sizeof(int)*nbrsize * 2));
		checkCudaErrors(cudaMalloc(&d_weights, sizeof(WeightType)*nbrsize));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nbrcoord, nbrcoord, sizeof(int)*nbrsize * 2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_weights, weights, sizeof(WeightType)*nbrsize, cudaMemcpyHostToDevice));


		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_FocalOperator<DataInType, DataOutType, WeightType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nbrcoord, d_weights, nbrsize, noDataValue, cellWidth, cellHeight, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nbrcoord));
	}

//-------------------------------------------------------------------------------------------------------------

	
	template<class DataInType, class DataOutType, class OperType>
	void FocalOper(DataInType* input, DataOutType* output, int width, int height, int *nbrcoord, int nbrsize, DataInType nodataIn, DataOutType nodataOut, int boundhandle,int nodataHandle)
	{
		DataInType* d_input;
		DataOutType* d_output;

		int* d_nbrcoord;
	

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nbrcoord, sizeof(int)*nbrsize * 2));
	

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nbrcoord, nbrcoord, sizeof(int)*nbrsize * 2, cudaMemcpyHostToDevice));


		dim3 block = CuEnvControl::getBlock2D();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_FocalOper<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nbrcoord, nbrsize, nodataIn, nodataOut, boundhandle, nodataHandle, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nbrcoord));
	}

	template<class DataInType, class DataOutType, class WeightType, class OperType>
	void FocalOperWeight(DataInType* input, DataOutType* output, int width, int height, int *nbrcoord, WeightType *weights, int nbrsize,DataInType nodataIn,DataOutType nodataOut, int boundhandle)
	{
		DataInType* d_input;
		DataOutType* d_output;

		int* d_nbrcoord;
		WeightType* d_weights;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nbrcoord, sizeof(int)*nbrsize * 2));
		checkCudaErrors(cudaMalloc(&d_weights, sizeof(WeightType)*nbrsize));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nbrcoord, nbrcoord, sizeof(int)*nbrsize * 2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_weights, weights, sizeof(WeightType)*nbrsize, cudaMemcpyHostToDevice));

		dim3 block = CuEnvControl::getBlock2D();
	    dim3 grid = CuEnvControl::getGrid(width, height);

		G_FocalOperWeight<DataInType, DataOutType, WeightType, OperType> << <grid, block >> >(d_input, d_output, width, height, d_nbrcoord, d_weights, nbrsize, nodataIn, nodataOut, boundhandle, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nbrcoord));
		checkCudaErrors(cudaFree(d_weights));
	}

	template<class DataInType, class DataOutType, class WeightType, class OperType>
	void FocalOperWeight(DataInType* input, DataOutType* output, int width, int height, double cellWidth, double cellHeight, int *nbrcoord, WeightType *weights, int nbrsize, DataInType nodataIn, DataOutType nodataOut, int boundhandle)
	{
		DataInType* d_input;
		DataOutType* d_output;

		int* d_nbrcoord;
		WeightType* d_weights;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_nbrcoord, sizeof(int)*nbrsize * 2));
		checkCudaErrors(cudaMalloc(&d_weights, sizeof(WeightType)*nbrsize));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nbrcoord, nbrcoord, sizeof(int)*nbrsize * 2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_weights, weights, sizeof(WeightType)*nbrsize, cudaMemcpyHostToDevice));
		dim3 block = CuEnvControl::getBlock2D();
	dim3 grid = CuEnvControl::getGrid(width, height);

		G_FocalOperWeight<DataInType, DataOutType, WeightType, OperType> << <grid, block >> >(d_input, d_output, width, height, cellWidth, cellHeight, d_nbrcoord, d_weights, nbrsize, nodataIn, nodataOut, boundhandle, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_nbrcoord));
		checkCudaErrors(cudaFree(d_weights));
	}

}





#endif