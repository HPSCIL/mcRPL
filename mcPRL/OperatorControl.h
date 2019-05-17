#ifndef OPERATORCONTROL_H_H
#define OPERATORCONTROL_H_H

#include "CuLayer.h"
#include <queue>

namespace pRPL
{
	template<class DataType> class CuLayer;

	template<class DataType>
	class OperControl
	{
	public:
		OperControl(CuLayer<DataType>& culayer){};
		OperControl(const OperControl<DataType>&operctrl);

		CuLayer<DataType>& OperExecute();

	private:
		queue<CuLayer<DataType>&>layerQueue;

		

	};


	template<class DataType>
	OperControl<DataType>::OperControl(const OperControl<DataType>&operctrl)
	{

	}

	template<class DataType>
	CuLayer<DataType>& OperControl<DataType>::OperExecute()
	{
		return NULL;
	}


}





#endif