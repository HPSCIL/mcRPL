#ifndef GLOBALCODEOPERATOR_H_H
#define GLOBALCODEOPERATOR_H_H

#include "CuPRL.h"
#include "GlobalOperatorDevice.h"

namespace pRPL
{





	template<class DataInType,class DataOutType,class OperType>
	void cuGlobalOperatorFn(DataInType* input, DataOutType* output, int width, int height,double cellWidth,double cellHeight,DataInType noDataType)
	{
		DataInType* d_input;
		DataOutType* d_output;
		

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		


		dim3 block = CuEnvControl::getBlock();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_GlobalEucCal<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, cellWidth, cellHeight, noDataType, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
	}

	template<class DataInType, class DataOutType, class OperType>
	void cuGlobalOperatorFn(DataInType* input, DataOutType* output, int width, int height, double cellWidth, double cellHeight, DataInType noDataType,rasterCell* pRasterCell,int cellnum)
	{
		DataInType* d_input;
		DataOutType* d_output;
		rasterCell* d_rasterCell;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(DataInType)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*width*height));
		checkCudaErrors(cudaMalloc(&d_rasterCell, sizeof(rasterCell)*cellnum));
		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataInType)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_rasterCell, pRasterCell, sizeof(rasterCell)*cellnum, cudaMemcpyHostToDevice));


		dim3 block = CuEnvControl::getBlock();
		dim3 grid = CuEnvControl::getGrid(width, height);

		G_GlobalEucCal<DataInType, DataOutType, OperType> << <grid, block >> >(d_input, d_output, width, height, cellWidth, cellHeight, noDataType, d_rasterCell, cellnum, OperType());

		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*width*height, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_rasterCell));
	}


}



#endif