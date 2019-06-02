#ifndef MCRPL_RASTERCELL_H
#define MCRPL_RASTERCELL_H
namespace mcRPL
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
		vector<mcRPL::rasterCell>_vRcs;
	};
}
#endif