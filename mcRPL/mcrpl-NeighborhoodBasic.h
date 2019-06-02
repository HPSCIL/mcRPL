#ifndef NEIGHBORHOODBASIC_H_H
#define NEIGHBORHOODBASIC_H_H


#include <vector>
using namespace std;


namespace mcRPL
{
	template<class WeightType>
	struct CuNbr
	{
		vector<int>coords;
		vector<WeightType>weights;
	};

	template<class WeightType>
	class NeighborhoodBasic
	{
	public:
		virtual CuNbr<WeightType> GetInnerNbr() = 0;
	};

};








#endif