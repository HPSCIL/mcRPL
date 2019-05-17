#ifndef PRPL_RASTERCELL_H
#define PRPL_RASTERCELL_H
namespace pRPL
{
	struct rasterCell
	{
		int x;
		int y;
		int value;
	};
	class rasterCellSpace
	{
	private:
		vector<pRPL::rasterCell>_vRcs;
	};
}
#endif