#ifndef mcPRL_RASTERCELL_H
#define mcPRL_RASTERCELL_H
namespace mcPRL
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
		vector<mcPRL::rasterCell>_vRcs;
	};
}
#endif