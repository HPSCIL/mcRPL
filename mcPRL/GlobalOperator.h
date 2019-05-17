#ifndef GLOBALOPERATOR_H_H
#define GLOBALOPERATOR_H_H

#include "CuPRL.h"
#include "GlobalCodeOperator.h"



namespace pRPL
{

	vector<rasterCell> getGlobalPoints(CuLayer<int>&culayer)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		int nodata = culayer.getNoDataValue();
		vector<rasterCell>vRasterCell;



		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (culayer[i*width + j] == nodata)
					continue;

				vector<rasterCell>::iterator iter = vRasterCell.begin();
				while (iter != vRasterCell.end())
				{
					if (iter->value == culayer[i*width + j])
					{
						break;
					}
					iter++;
				}
				if (iter == vRasterCell.end())
				{
					rasterCell cell;
					cell.x = j;
					cell.y = i;
					cell.value = culayer[i*width + j];
					vRasterCell.push_back(cell);
				}
			}
		}

		return vRasterCell;
	}
	



	template<class DataInType,class DataOutType,class OperType>
	CuLayer<DataOutType> cuGlobalOperatorFn(CuLayer<DataInType>&culayer)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		



		CuLayer<DataOutType>outlayer(width, height);
		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		int noDataValue = culayer.getNoDataValue();
		double cellwidth = culayer.getCellWidth();
		double cellheight = culayer.getCellHeight();

		if (std::abs(cellwidth) < 1e-6)
			cellwidth = 1;
		if (std::abs(cellheight) < 1e-6)
			cellheight = cellwidth;


		cuGlobalOperatorFn<DataInType, DataOutType, OperType>(input, output, width, height, cellwidth, cellheight, noDataValue);

		return outlayer;

	}

	template<class DataInType, class DataOutType, class OperType>
	CuLayer<DataOutType> cuGlobalOperatorFn(CuLayer<DataInType>&culayer,vector<rasterCell>&vRasterCell)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();




		CuLayer<DataOutType>outlayer(width, height);
		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		int noDataValue = culayer.getNoDataValue();
		double cellwidth = culayer.getCellWidth();
		double cellheight = culayer.getCellHeight();

		if (std::abs(cellwidth) < 1e-6)
			cellwidth = 1;
		if (std::abs(cellheight) < 1e-6)
			cellheight = cellwidth;

		int cellnum = vRasterCell.size();
		rasterCell* pRasterCell = &vRasterCell[0];
		cuGlobalOperatorFn<DataInType, DataOutType, OperType>(input, output, width, height, cellwidth, cellheight, noDataValue, pRasterCell, cellnum);

		return outlayer;

	}



}




#endif