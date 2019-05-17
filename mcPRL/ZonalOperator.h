#ifndef ZONALOPERATOR_H_H
#define ZONALOPERATOR_H_H

#include "ZonalCodeOperator.h"
#include "CuLayer.h"

namespace pRPL
{
	template<class DataOutType, class Oper>
	vector<DataOutType> cuZonelStatisticSum(CuLayer<int>&culayer, vector<int>&zonelvalues)
	{
		vector<DataOutType>vOutput;
		vOutput.resize(zonelvalues.size());

		int width = culayer.getWidth();
		int height = culayer.getHeight();

		int* input = culayer.getData();

		int noDataValue = culayer.getNoDataValue();
		double cellwidth = culayer.getCellWidth();
		double cellheight = culayer.getCellHeight();

		if (std::abs(cellwidth) < 1e-6)
			cellwidth = 1;
		if (std::abs(cellheight) < 1e-6)
			cellheight = cellwidth;

		cuZonelStatisticSum<DataOutType,Oper>(input, &vOutput[0], &zonelvalues[0], zonelvalues.size(), width, height, cellwidth, cellheight, noDataValue);


		return vOutput;
	}
}








#endif