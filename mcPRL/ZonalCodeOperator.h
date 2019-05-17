#ifndef ZONALCODEOPERATOR_H_H
#define ZONALCODEOPERATOR_H_H

#include "CuPRL.h"
#include "ZonalOperatorDevice.h"

namespace pRPL
{
	template<class DataOutType, class Oper>
	void cuZonelStatisticSum(int* input, DataOutType* output, int* zonelvalues, int zonelnum, int width, int height, double cellWidth, double cellHeight, int noDataValue)
	{
		int* d_input;
		DataOutType* d_output;
		int* d_zonelvalues;

		checkCudaErrors(cudaMalloc(&d_input, sizeof(int)*width*height));
		checkCudaErrors(cudaMalloc(&d_output, sizeof(DataOutType)*zonelnum*height));
		checkCudaErrors(cudaMalloc(&d_zonelvalues, sizeof(int)*zonelnum));

		checkCudaErrors(cudaMemcpy(d_input, input, sizeof(int)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_zonelvalues, zonelvalues, sizeof(int)*zonelnum, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemset(d_output, 0, sizeof(DataOutType)*zonelnum*height));

		//dim3 block = CuEnvControl::getBlock();
		//dim3 grid = CuEnvControl::getGrid(width, height);

		dim3 block(256);
		dim3 grid(height % block.x == 0 ? height / block.x : height / block.x+1);

		G_ZonelStatisticSumRow<DataOutType, Oper> << <grid, block >> >(d_input, d_output, d_zonelvalues, zonelnum, width, height, cellWidth, cellHeight, noDataValue, Oper());

		grid.x = zonelnum % block.x == 0 ? zonelnum / block.x : height / block.x + 1;
		G_ZonelStatisticSumCol << <grid, block >> >(d_output, zonelnum, height);


		checkCudaErrors(cudaMemcpy(output, d_output, sizeof(DataOutType)*zonelnum, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_input));
		checkCudaErrors(cudaFree(d_output));
		checkCudaErrors(cudaFree(d_zonelvalues));
	}

}





#endif