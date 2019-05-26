#ifndef mcPRL_NEIGHBORHOOD_H
#define mcPRL_NEIGHBORHOOD_H


#include "mcPRL-basicTypes.h"

namespace mcPRL {
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
                   mcPRL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                   double virtualEdgeVal = 0.0);
      Neighborhood(const mcPRL::Neighborhood &rhs);
      ~Neighborhood();
      
      /* Initiate */
      bool init(const vector<mcPRL::CellCoord> &vNbrCoords,
                double weight = 1.0,
                mcPRL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                double virtualEdgeVal = 0.0);
      bool init(const vector<mcPRL::WeightedCellCoord> &vNbrCoords,
                mcPRL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                double virtualEdgeVal = 0.0);
      bool initSingleCell(double weight = 1.0);
      bool initVonNeumann(double weight = 1.0,
                          mcPRL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                          double virtualEdgeVal = 0.0);
      bool initMoore(long nEdgeLength = 3,
                     double weight = 1.0,
                     mcPRL::EdgeOption edgeOption = FORBID_VIRTUAL_EDGES,
                     double virtualEdgeVal = 0.0);
      void clear();
      
      /* Operators */
      mcPRL::WeightedCellCoord& operator[](int iNbr);
      const mcPRL::WeightedCellCoord& operator[](int iNbr) const;
      mcPRL::Neighborhood& operator=(const mcPRL::Neighborhood &rhs);

      /* Information */
      const char* name() const;
      void name(const char *aName);
      bool isEmpty() const;
      int size() const;
      bool isEquallyWeighted(double &weight) const;
	  vector<mcPRL::WeightedCellCoord>getInnerNbr()
	  {
		  return _vNbrs;
	  }
      long minIRow() const;
      long minICol() const;
      long maxIRow() const;
      long maxICol() const;
      long nRows() const;
      long nCols() const;
      const mcPRL::CoordBR* getMBR() const;
      mcPRL::EdgeOption edgeOption() const;
      const double* virtualEdgeVal() const;
      bool virtualEdgeVal(const double veVal);
      
      /* Neighbor accessing */
      bool hasNbrs(mcPRL::MeshDir dir) const;
      const mcPRL::IntVect* nbrIDs(mcPRL::MeshDir dir) const;
      
      /* Cellspace-related methods */
      bool calcWorkBR(mcPRL::CoordBR &workBR,
                      const mcPRL::SpaceDims &dims) const;

      /* Compression */
      void add2Buf(mcPRL::CharVect &vBuf) const;
      bool fromBuf(const mcPRL::CharVect &vBuf,
                   int &iChar);
	  void addnbrForGpu();
      vector<double>cuWeights();
	   vector<int>cuCoords();
    private:
      bool _checkNbrCoords(const vector<mcPRL::CellCoord> &vNbrCoords) const;
      bool _validNumDirects() const;

    private:
	  vector<double>cuNbrWeights;
	  vector<int>cuNbrCoords;
      string _name;
      vector<mcPRL::WeightedCellCoord> _vNbrs;
      mcPRL::CoordBR _MBR;
      vector<mcPRL::IntVect> _mNbrIDMap;
      mcPRL::EdgeOption _edgeOption;
      double *_pVirtualEdgeVal;
  };

  ostream& operator<<(ostream &os,
                      const mcPRL::Neighborhood &nbrhd);
  
  istream& operator>>(istream &is,
                      mcPRL::Neighborhood &nbrhd);
};

#endif
