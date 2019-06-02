#ifndef DEVICEPARAMTYPE_H_H
#define DEVICEPARAMTYPE_H_H
#include "mcrpl-CuPRL.h"
struct rasterInfo
{
	int width;
	int height;
	double cellWidth;
	double cellHeight;

};
template<class WeightType>
struct nbrInfo
{
	int* nbrcood;
	int nbrsize;
	WeightType* weights;
};
struct rasterCell
{
	int x;
	int y;
	int value;
};
namespace mcRPL
{
	//#define EPSINON 1e-06

	template<class DataType>
	__device__ bool isNoData(DataType value, DataType nodata)
	{
		if ((value - nodata) > -1e-06 && (value - nodata)<1e-06)
			return true;
		else
			return false;
	};
	class paramInfo
	{
	public:
	private:

		void* m_params;
		int* m_marks;
		int paramsnum;

	};
}
#endif