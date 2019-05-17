#ifndef FOCALOPERATOR_H_H
#define FOCALOPERATOR_H_H

#include "CuLayer.h"
#include "FocalCodeOperator.h"
#include "FocalOperatorDevice.h"
#include "NeighborhoodRect.h"
namespace pRPL
{
	template<class DataInType,class DataOutType,class WeightType>
	CuLayer<DataOutType> focalStatisticsSum(CuLayer<DataInType>culayer, NeighborhoodBasic<WeightType>*pNbrObj, BoundHanleType boundhandle,NoDataHandle nodatahandle)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayer.getProjection());
		double adfGeoTransform[6];
		culayer.getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		DataInType nodataIn = culayer.getNoDataValue();
		DataOutType nodataOut = outlayer.getNoDataValue();

		CuNbr<WeightType>cuNbr = pNbrObj->GetInnerNbr();

		int *coords = &cuNbr.coords[0];
		int nbrsize = cuNbr.weights.size();

		FocalOper<DataInType, DataOutType, FocalStatisticsSum<DataInType, DataOutType> >(input, output, width, height, coords, nbrsize, nodataIn, nodataOut, int(boundhandle), nodatahandle);

		return outlayer;
	}

	template<class DataInType, class DataOutType, class WeightType>
	CuLayer<DataOutType> focalStatisticsMean(CuLayer<DataInType>culayer, NeighborhoodBasic<WeightType>*pNbrObj, BoundHanleType boundhandle, NoDataHandle nodatahandle)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayer.getProjection());
		double adfGeoTransform[6];
		culayer.getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		DataInType nodataIn = culayer.getNoDataValue();
		DataOutType nodataOut = outlayer.getNoDataValue();

		CuNbr<WeightType>cuNbr = pNbrObj->GetInnerNbr();

		int *coords = &cuNbr.coords[0];
		int nbrsize = cuNbr.weights.size();

		FocalOper<DataInType, DataOutType, FocalStatisticsMean<DataInType, DataOutType> >(input, output, width, height, coords, nbrsize, nodataIn, nodataOut, int(boundhandle),int(nodatahandle));

		return outlayer;
	}

	template<class DataInType,class DataOutType>
	CuLayer<DataOutType> focalStatisticsMaximum(CuLayer<DataInType>culayer, NeighborhoodBasic<int>*pNbrObj)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayer.getProjection());
		double adfGeoTransform[6];
		culayer.getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		DataInType nodataIn = culayer.getNoDataValue();
		DataOutType nodataOut = outlayer.getNoDataValue();

		CuNbr<int>cuNbr = pNbrObj->GetInnerNbr();

		int *coords = &cuNbr.coords[0];
		int nbrsize = cuNbr.weights.size();

		FocalOper<DataInType, DataOutType, FocalStatisticsMaximum<DataInType, DataOutType> >(input, output, width, height, coords, nbrsize, nodataIn, nodataOut, 0);

		return outlayer;
	}

	template<class DataInType,class DataOutType>
	CuLayer<DataOutType> focalStatisticsMinimum(CuLayer<DataInType>culayer, NeighborhoodBasic<int>*pNbrObj)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayer.getProjection());
		double adfGeoTransform[6];
		culayer.getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		DataInType nodataIn = culayer.getNoDataValue();
		DataOutType nodataOut = outlayer.getNoDataValue();

		CuNbr<int>cuNbr = pNbrObj->GetInnerNbr();

		int *coords = &cuNbr.coords[0];
		int nbrsize = cuNbr.weights.size();

		FocalOper<DataInType, DataOutType, FocalStatisticsMinimum<DataInType, DataOutType> >(input, output, width, height, coords, nbrsize, nodataIn, nodataOut, 0);

		return outlayer;
	}


	template<class DataInType,class DataOutType>
	CuLayer<DataOutType> focalStatisticsRange(CuLayer<DataOutType>culayer, NeighborhoodBasic<int>*pNbrObj)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayer.getProjection());
		double adfGeoTransform[6];
		culayer.getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();

		DataInType nodataIn = culayer.getNoDataValue();
		DataOutType nodataOut = outlayer.getNoDataValue();

		CuNbr<int>cuNbr = pNbrObj->GetInnerNbr();

		int *coords = &cuNbr.coords[0];
		int nbrsize = cuNbr.weights.size();

		FocalOper<DataInType, DataOutType, FocalStatisticsRange<DataInType, DataOutType> >(input, output, width, height, coords, nbrsize, nodataIn, nodataOut, 0);

		return outlayer;
	}


	template<class DataInType,class DataOutType,class WeightType,class Oper>
	CuLayer<DataOutType> cuFocalOperatorFn(CuLayer<DataInType>&culayer, NeighborhoodBasic<WeightType>*pNbrObj)
	{
		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		DataInType* input = culayer.getData();
		DataOutType* output = outlayer.getData();
		outlayer.setProjection(culayer.getProjection());
		double adfGeoTransform[6];
		culayer.getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);
		CuNbr<WeightType>cuNbr = pNbrObj->GetInnerNbr();

		int *coords = &cuNbr.coords[0];
		int nbrsize = cuNbr.weights.size();
		WeightType* weights = &cuNbr.weights[0];

		int noDataValue = culayer.getNoDataValue();
		double cellwidth = culayer.getCellWidth();
		double cellheight = culayer.getCellHeight();

		if (std::abs(cellwidth) < 1e-6)
			cellwidth = 1;
		if (std::abs(cellheight) < 1e-6)
			cellheight = cellwidth;


		cuFocalOperatorFn<DataInType, DataOutType, WeightType, Oper>(input, output, width, height, coords, weights, nbrsize, cellwidth, cellheight, noDataValue);

		return outlayer;
	}


//------------------------------------------------------------------------------------------------------





}






#endif