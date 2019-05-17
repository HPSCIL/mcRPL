#ifndef NEIGHBORHOODSLOPE_H_H
#define NEIGHBORHOODSLOPE_H_H

#include "NeighborhoodBasic.h"


namespace pRPL
{
	class NeighborhoodSlope : public NeighborhoodBasic<int>
	{
	public:
		NeighborhoodSlope(){};
		~NeighborhoodSlope(){};


		CuNbr<int> GetInnerNbr();

	private:
		int m_width;
		int m_height;
		CuNbr<int>m_cunbr;
	};
	CuNbr<int> NeighborhoodSlope::GetInnerNbr()
	{
		CuNbr<int>cuNbr;
		int nbrweight1[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		int nbrweight2[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				if (i == 0 && j == 0)
					continue;

				cuNbr.coords.push_back(j);
				cuNbr.coords.push_back(i);
				cuNbr.weights.push_back(nbrweight1[(i + 1) * 3 + j + 1]);
			}
		}

		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				if (i == 0 && j == 0)
					continue;

				cuNbr.coords.push_back(j);
				cuNbr.coords.push_back(i);
				cuNbr.weights.push_back(nbrweight2[(i + 1) * 3 + j + 1]);
			}
		}
		return cuNbr;

	}
};

#endif