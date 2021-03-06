#ifndef CUPRL_H_H
#define CUPRL_H_H

//#include "CuEnvControl.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <gdal_priv.h>
#include <string>
//#include"CuEnvControl.h"
#include "mcrpl-errorhelper.h"
#include "mcrpl-DeviceParamType.h"
//#include "CuEnvControl.h"
namespace mcRPL
{

	



	enum BoundHanleType
	{
		FOCALUSE = 0,
		NOUSE
	};

	enum NoDataHandle
	{
		IGNORE,
		NOCAL
	};

	enum LocalOperType
	{
		LOCALOPER = 0,
		LOCALOPERWITHNODATA,
		LOCALOPERPARAM,
		LOCALOPERPARAMWITHNODATA,
		LOCALOPERPARAMS,
		LOCALOPERPARAMSWITHNODATA,
		LOCALOPERMULTILAYERS,
		LOCALOPERMULTILAYERSWITHNODATA
	};



}






#endif