#ifndef NEIGHBORHOODRECT_H_H
#define NEIGHBORHOODRECT_H_H

#include "NeighborhoodBasic.h"


namespace pRPL
{
	template<class WeightType>
	class NeighborhoodRect : public NeighborhoodBasic<WeightType>
	{
	public:
		NeighborhoodRect();
		NeighborhoodRect(int width, int height);
		~NeighborhoodRect(){};


		CuNbr<WeightType> GetInnerNbr();

	private:
		int m_width;
		int m_height;
		CuNbr<WeightType>m_cunbr;
	};

	template<class WeightType>
	NeighborhoodRect<WeightType>::NeighborhoodRect()
	{

	}

	template<class WeightType>
	NeighborhoodRect<WeightType>::NeighborhoodRect(int width, int height)
	{
		this->m_height = height;
		this->m_width = width;
	}




	template<class WeightType>
	CuNbr<WeightType> NeighborhoodRect<WeightType>::GetInnerNbr()
	{
		CuNbr<WeightType>cuNbr;

		for (int idxrow = -(m_height - 1) / 2; idxrow < (m_height / 2 + 1); idxrow++)
		{
			for (int idxcol = -(m_width - 1) / 2; idxcol < (m_width / 2 + 1); idxcol++)
			{
				cuNbr.coords.push_back(idxcol);
				cuNbr.coords.push_back(idxrow);
				cuNbr.weights.push_back(1);
			}
		}

		return cuNbr;

	}


};














#endif