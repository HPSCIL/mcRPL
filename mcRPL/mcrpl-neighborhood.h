#ifndef MCRPL_NEIGHBORHOOD_H
#define MCRPL_NEIGHBORHOOD_H


#include "mcrpl-basicTypes.h"

namespace mcRPL {
  static const int MooreNbrLocs[16] = {-1, 0,
                                       -1, 1,
                                       0, 1,
                                       1, 1,
                                       1, 0,
                                       1, -1,
                                       0, -1,
                                       -1, -1};

  class Neighborhood {
    public:
      /* Constructor and destructor */
      Neighborhood(const char *aName = NULL,
                   mcRPL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                   double virtualEdgeVal = 0.0);
      Neighborhood(const mcRPL::Neighborhood &rhs);
      ~Neighborhood();
      
      /* Initiate */
      bool init(const vector<mcRPL::CellCoord> &vNbrCoords,
                double weight = 1.0,
                mcRPL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                double virtualEdgeVal = 0.0);
      bool init(const vector<mcRPL::WeightedCellCoord> &vNbrCoords,
                mcRPL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                double virtualEdgeVal = 0.0);
      bool initSingleCell(double weight = 1.0);
      bool initVonNeumann(double weight = 1.0,
                          mcRPL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                          double virtualEdgeVal = 0.0);
      bool initMoore(long nEdgeLength = 3,
                     double weight = 1.0,
                     mcRPL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                     double virtualEdgeVal = 0.0);
      void clear();
      
      /* Operators */
      mcRPL::WeightedCellCoord& operator[](int iNbr);
      const mcRPL::WeightedCellCoord& operator[](int iNbr) const;
      mcRPL::Neighborhood& operator=(const mcRPL::Neighborhood &rhs);

      /* Information */
      const char* name() const;
      void name(const char *aName);
      bool isEmpty() const;
      int size() const;
      bool isEquallyWeighted(double &weight) const;
	  vector<mcRPL::WeightedCellCoord>getInnerNbr()
	  {
		  return _vNbrs;
	  }
      long minIRow() const;
      long minICol() const;
      long maxIRow() const;
      long maxICol() const;
      long nRows() const;
      long nCols() const;
      const mcRPL::CoordBR* getMBR() const;
      mcRPL::EdgeOption edgeOption() const;
      const double* virtualEdgeVal() const;
      bool virtualEdgeVal(const double veVal);
      
      /* Neighbor accessing */
      bool hasNbrs(mcRPL::MeshDir dir) const;
      const mcRPL::IntVect* nbrIDs(mcRPL::MeshDir dir) const;
      
      /* Cellspace-related methods */
      bool calcWorkBR(mcRPL::CoordBR &workBR,
                      const mcRPL::SpaceDims &dims) const;

      /* Compression */
      void add2Buf(mcRPL::CharVect &vBuf) const;
      bool fromBuf(const mcRPL::CharVect &vBuf,
                   int &iChar);
	  void addnbrForGpu();
      vector<double>cuWeights();
	   vector<int>cuCoords();
    private:
      bool _checkNbrCoords(const vector<mcRPL::CellCoord> &vNbrCoords) const;
      bool _validNumDirects() const;

    private:
	  vector<double>cuNbrWeights;
	  vector<int>cuNbrCoords;
      string _name;
      vector<mcRPL::WeightedCellCoord> _vNbrs;
      mcRPL::CoordBR _MBR;
      vector<mcRPL::IntVect> _mNbrIDMap;
      mcRPL::EdgeOption _edgeOption;
      double *_pVirtualEdgeVal;
  };

  ostream& operator<<(ostream &os,
                      const mcRPL::Neighborhood &nbrhd);
  
  istream& operator>>(istream &is,
                      mcRPL::Neighborhood &nbrhd);
};

#endif
