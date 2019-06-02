#ifndef LOCALOPERATOR_H_H
#define LOCALOPERATOR_H_H


#include "mcrpl-CuLayer.h"
#include "mcrpl-LocalOperatorDevice.h"
#include "mcrpl-LocalCodeOperator.h"

namespace mcRPL
{
	template<class DataInType,class DataOutType>
	CuLayer<DataOutType> abs(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaAbs<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}



	template<class DataInType,class DataOutType>
	CuLayer<DataOutType> sin(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaSin<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> cos(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaCos<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> tan(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaTan<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> asin(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaAsin<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> acos(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaAcos<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> atan(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaAtan<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}




	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> sqr(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaSqr<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> sqrt(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaSqrt<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}


	template<class DataInType,class DataOutType,class ParamType>
	CuLayer<DataOutType> pow(CuLayer<DataInType>& culayer, ParamType param)
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

		LocalOperParam<DataInType, DataOutType, ParamType, CudaPow<DataInType, DataOutType, ParamType>>(input, output, width, height, nodataIn, nodataOut, param);

		return outlayer;
	}

	template<class DataInType,class DataOutType>
	CuLayer<DataOutType> exp(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaExp<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> exp2(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaExp2<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> log(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaLog<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> log2(CuLayer<DataInType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaLog2<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType>
	CuLayer<DataOutType> log10(CuLayer<DataOutType>& culayer)
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

		LocalOper<DataInType, DataOutType, CudaLog10<DataInType, DataOutType>>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}


	template<class DataType1,class DataType2>
	CuLayer<DataType2> reclassValueUpdate(CuLayer<DataType1>& culayer,vector<DataType1>&oldValueSet,vector<DataType2>&newValueSet)
	{
		if (oldValueSet.size() != newValueSet.size())
		{
			printError("Reclass Dataset size is not the same.");
			exit(EXIT_FAILURE);
		}


		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataType2>outlayer(width, height);

		DataType1* input = culayer.getData();
		DataType2* output = culayer.getData();

		DataType1* pOldValueSet = &(oldValueSet[0]);
		DataType2* pNewValueSet = &(newValueSet[0]);

		int length = oldValueSet.size();

		ReClass<DataType1, DataType2, ReclassValueUpdate<DataType1, DataType2>>(input, output, width, height, pOldValueSet, pNewValueSet, length);

		return outlayer;

	}

	template<class DataType1, class DataType2>
	CuLayer<DataType2> reclassRangeUpdate(CuLayer<DataType1>& culayer, vector<DataType1>&oldRangeSet, vector<DataType2>&newValueSet)
	{
		if (oldRangeSet.size() != newValueSet.size())
		{
			printError("Reclass Dataset size is not the same.");
			exit(EXIT_FAILURE);
		}


		int width = culayer.getWidth();
		int height = culayer.getHeight();
		CuLayer<DataType2>outlayer(width, height);

		DataType1* input = culayer.getData();
		DataType2* output = culayer.getData();

		DataType1* pOldValueSet = &(oldRangeSet[0]);
		vector<DataType2>newValueSetExt = newValueSet;
		newValueSet.push_back(0);
		DataType2* pNewValueSet = &(newValueSetExt[0]);

		int length = oldRangeSet.size();

		ReClass<DataType1, DataType2, ReclassValueUpdate<DataType1, DataType2>>(input, output, width, height, pOldValueSet, pNewValueSet, length);

		return outlayer;

	}

	

	template<class DataInType,class DataOutType,class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn(vector<CuLayer<DataInType> >culayers, vector<double>paramInfo)
	{
		int width = culayers.getWidth();
		int height = culayers.getHeight();
		CuLayer<DataOutType>outlayer(width, height);

		vector<DataInType*>input;


		for (int idxNumLayer = 0; idxNumLayer < culayers.size(); idxNumLayer++)
		{
			input.push_back(culayers[idxNumLayer].getData());
		}

		DataOutType* output = outlayer.getData();

		DataInType noDataValue = culayers[0].getNoDataValue();
		
		cuLocalOperatorFn<DataInType, DataOutType, OperType>(input, output, width, height, culayers.size(), noDataValue, &paramInfo[0], paramInfo.size());
		
		return outlayer;
	}

	


	//----------------------------------------------------------------------------------------

	template<class DataInType,class DataOutType,class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn1(CuLayer<DataInType>culayer)
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

		LocalOper<DataInType, DataOutType, OperType>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType, class DataOutType, class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn2(CuLayer<DataInType>culayer)
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

		LocalOperWithNoData<DataInType, DataOutType, OperType>(input, output, width, height, nodataIn, nodataOut);
		return outlayer;
	}

	template<class DataInType,class DataOutType,class ParamType,class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn1(CuLayer<DataInType>culayer, ParamType param)
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

		LocalOperParam<DataInType, DataOutType, OperType>(input, output, width, height, nodataIn, nodataOut, param);
		return outlayer;
	}

	template<class DataInType, class DataOutType, class ParamType, class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn2(CuLayer<DataInType>culayer, ParamType param)
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

		LocalOperParamWithNoData<DataInType, DataOutType, OperType>(input, output, width, height, nodataIn, nodataOut, param);
		return outlayer;
	}

	template<class DataInType, class DataOutType, class ParamType, class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn1(CuLayer<DataInType>culayer, ParamType* params,int paramsnum)
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

		
		LocalOperParams<DataInType, DataOutType, OperType>(input, output, width, height, nodataIn, nodataOut, params, paramsnum);
		
		return outlayer;
	}

	template<class DataInType, class DataOutType, class ParamType, class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn2(CuLayer<DataInType>culayer, ParamType* params, int paramsnum)
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

		
		LocalOperParamsWithNoData<DataInType, DataOutType, OperType>(input, output, width, height, nodataIn, nodataOut, params, paramsnum);
		
		return outlayer;
	}


	template<class DataInType, class DataOutType, class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn1(vector<CuLayer<DataInType> >culayers)
	{
		int width = culayers[0].getWidth();
		int height = culayers[0].getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayers[0].getProjection());
		double adfGeoTransform[6];
		culayers[0].getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		vector<DataInType*>input;
		vector<DataInType>nodataIn;
		for (int idxNumLayer = 0; idxNumLayer < culayers.size(); idxNumLayer++)
		{
			input.push_back(culayers[idxNumLayer].getData());
			nodataIn.push_back(culayers[idxNumLayer].getNoDataValue());
		}

		DataOutType* output = outlayer.getData();

		DataOutType nodataOut = outlayer.getNoDataValue();

		LocalOperMultiLayers<DataInType, DataOutType, OperType>(&input[0], output, width, height, &nodataIn[0], nodataOut, culayers.size());
		
		return outlayer;
	}

	template<class DataInType, class DataOutType, class OperType>
	CuLayer<DataOutType> cuLocalOperatorFn2(vector<CuLayer<DataInType> >culayers)
	{
		int width = culayers[0].getWidth();
		int height = culayers[0].getHeight();
		CuLayer<DataOutType>outlayer(width, height);
		outlayer.setProjection(culayers[0].getProjection());
		double adfGeoTransform[6];
		culayers[0].getGeoTransform(adfGeoTransform);
		outlayer.setGeoTransform(adfGeoTransform);

		vector<DataInType*>input;
		vector<DataInType>nodataIn;
		for (int idxNumLayer = 0; idxNumLayer < culayers.size(); idxNumLayer++)
		{
			input.push_back(culayers[idxNumLayer].getData());
			nodataIn.push_back(culayers[idxNumLayer].getNoDataValue());
		}

		DataOutType* output = outlayer.getData();

		DataOutType nodataOut = outlayer.getNoDataValue();

		LocalOperMultiLayersWithNoData<DataInType, DataOutType, OperType>(&input[0], output, width, height, &nodataIn[0], nodataOut, culayers.size());

		return outlayer;
	}


}


#endif