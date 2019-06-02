#ifndef MCRPL_SUBCELLSPACEINFO_H
#define MCRPL_SUBCELLSPACEINFO_H


#include "mcrpl-basicTypes.h"
#include "mcrpl-neighborhood.h"

namespace mcRPL {
  class SubCellspaceInfo {
    public:
      SubCellspaceInfo();
      SubCellspaceInfo(int id,
                       mcRPL::DomDcmpType domDcmpType,
                       const mcRPL::SpaceDims &glbDims,
                       const mcRPL::CoordBR &MBR,
                       const mcRPL::CoordBR &workBR,
                       const vector<mcRPL::IntVect> &mNbrSpcIDs);
      SubCellspaceInfo(const mcRPL::SubCellspaceInfo &rhs);
      SubCellspaceInfo(const mcRPL::IntVect &vInfoPack,
                       mcRPL::IntVctItr &iVal);

      ~SubCellspaceInfo() {}

      mcRPL::SubCellspaceInfo& operator=(const mcRPL::SubCellspaceInfo &rhs);
      bool operator==(const mcRPL::SubCellspaceInfo &rhs) const;
      bool operator!=(const mcRPL::SubCellspaceInfo &rhs) const;

      int id() const;
      mcRPL::DomDcmpType domDcmpType() const;
      long nGlbRows() const;
      long nGlbCols() const;
      const mcRPL::SpaceDims& glbDims() const;
      long nRows() const;
      long nCols() const;
      const mcRPL::SpaceDims& dims() const;
      long iRowBegin() const;
      long iColBegin() const;
      long iRowEnd() const;
      long iColEnd() const;
      const mcRPL::CoordBR& MBR() const;
      long iRowWorkBegin() const;
      long iColWorkBegin() const;
      long iRowWorkEnd() const;
      long iColWorkEnd() const;
      const mcRPL::CoordBR& workBR() const;
      double sizeRatio() const;
      
      bool validGlbIdx(long glbIdx,
                       bool warning = true) const;
      bool validGlbCoord(const mcRPL::CellCoord &glbCoord,
                         bool warning = true) const;
      bool validGlbCoord(long iRowGlb, long iColGlb,
                         bool warning = true) const;
      const mcRPL::CellCoord glbIdx2glbCoord(long glbIdx) const;
      long glbCoord2glbIdx(const mcRPL::CellCoord &glbCoord) const;
      long glbCoord2glbIdx(long iRowGlb, long iColGlb) const;

      bool validIdx(long idx,
                    bool warning = true) const;
      bool validCoord(const mcRPL::CellCoord &coord,
                      bool warning = true) const;
      bool validCoord(long iRow, long iCol,
                      bool warning = true) const;
      long coord2idx(const mcRPL::CellCoord &coord) const;
      long coord2idx(long iRow, long iCol) const;
      const mcRPL::CellCoord idx2coord(long idx) const;

      const mcRPL::CellCoord lclCoord2glbCoord(const mcRPL::CellCoord& lclCoord) const;
      const mcRPL::CellCoord lclCoord2glbCoord(long iRowLcl, long iColLcl) const;
      const mcRPL::CellCoord glbCoord2lclCoord(const mcRPL::CellCoord& glbCoord) const;
      const mcRPL::CellCoord glbCoord2lclCoord(long iRowGlb, long iColGlb) const;
      long lclCoord2glbIdx(const mcRPL::CellCoord &lclCoord) const;
      long lclCoord2glbIdx(long iRowLcl, long iColLcl) const;
      const mcRPL::CellCoord glbIdx2lclCoord(long glbIdx) const;
      long glbIdx2lclIdx(long glbIdx) const;
      long lclIdx2glbIdx(long lclIdx) const;

      int nNbrDirs() const;
      int nEdges() const;
      int nTotNbrs() const;
      bool validSpcDir(int iDir) const;
      bool hasNbrs(int iDir) const;
      int nNbrs(int iDir) const;
      const mcRPL::IntVect& nbrSubSpcIDs(int iDir) const;
      int nbrDir(int nbrID) const;
      mcRPL::MeshDir spcDir2MeshDir(int iDir) const;
      int oppositeDir(int iDir) const;

      bool calcBRs(bool onlyUpdtCtrCell = true,
                   const mcRPL::Neighborhood *pNbrhd = NULL,
                   const list<mcRPL::SubCellspaceInfo> *plSubspcInfos = NULL);
      const mcRPL::CoordBR* interiorBR() const;
      const mcRPL::CoordBR* edgeBR(int iDir) const;
      const mcRPL::CoordBR* sendBR(int iDir,
                                  int iNbr = 0) const;

      /*
      void add2IntVect(mcRPL::IntVect &vInfoPack) const;
      bool fromIntVect(const mcRPL::IntVect &vInfoPack,
                       mcRPL::IntVctItr &iVal);
      */

    protected:
      void _initBRs();
      bool _rowwiseBRs(bool onlyUpdtCtrCell = true,
                       const mcRPL::Neighborhood *pNbrhd = NULL);
      bool _colwiseBRs(bool onlyUpdtCtrCell = true,
                       const mcRPL::Neighborhood *pNbrhd = NULL);
      bool _blockwiseBRs(bool onlyUpdtCtrCell = true,
                         const mcRPL::Neighborhood *pNbrhd = NULL,
                         const list<mcRPL::SubCellspaceInfo> *plSubspcInfos = NULL);
      const mcRPL::SubCellspaceInfo* _findSubspcInfo(const list<mcRPL::SubCellspaceInfo> *plSubspcInfos,
                                                    int subspcGlbID) const;

    protected:
      int _id;
      mcRPL::DomDcmpType _domDcmpType;
      mcRPL::SpaceDims _glbDims;
      mcRPL::CoordBR _MBR; /* MBR is in global coordinates */
      mcRPL::SpaceDims _dims;
      mcRPL::CoordBR _workBR; /* workBR is in local coordinates */
      vector<mcRPL::IntVect> _mNbrSpcIDs;

      vector<vector<CoordBR> > _mSendBRs;
      vector<CoordBR> _vEdgeBRs;
      CoordBR _interiorBR;
  };
};

#endif
