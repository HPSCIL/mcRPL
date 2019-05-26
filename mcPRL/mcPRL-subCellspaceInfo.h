#ifndef mcPRL_SUBCELLSPACEINFO_H
#define mcPRL_SUBCELLSPACEINFO_H

/***************************************************************************
* mcPRL-subCellspaceInfo.h
*
* Project: mcPRL, v 2.2
* Purpose: Header file for class mcPRL::SubCellspaceInfo
* Author:  Qingfeng (Gene) Guan
* E-mail:  guanqf {at} gmail.com
****************************************************************************
* Copyright (c) 2008, Qingfeng Guan
* NOTE: this library can ONLY be used for EDUCATIONAL and SCIENTIFIC 
* purposes, NO COMMERCIAL usages are allowed unless the author is 
* contacted and a permission is granted
* 
****************************************************************************/

#include "mcPRL-basicTypes.h"
#include "mcPRL-neighborhood.h"

namespace mcPRL {
  class SubCellspaceInfo {
    public:
      SubCellspaceInfo();
      SubCellspaceInfo(int id,
                       mcPRL::DomDcmpType domDcmpType,
                       const mcPRL::SpaceDims &glbDims,
                       const mcPRL::CoordBR &MBR,
                       const mcPRL::CoordBR &workBR,
                       const vector<mcPRL::IntVect> &mNbrSpcIDs);
      SubCellspaceInfo(const mcPRL::SubCellspaceInfo &rhs);
      SubCellspaceInfo(const mcPRL::IntVect &vInfoPack,
                       mcPRL::IntVctItr &iVal);

      ~SubCellspaceInfo() {}

      mcPRL::SubCellspaceInfo& operator=(const mcPRL::SubCellspaceInfo &rhs);
      bool operator==(const mcPRL::SubCellspaceInfo &rhs) const;
      bool operator!=(const mcPRL::SubCellspaceInfo &rhs) const;

      int id() const;
      mcPRL::DomDcmpType domDcmpType() const;
      long nGlbRows() const;
      long nGlbCols() const;
      const mcPRL::SpaceDims& glbDims() const;
      long nRows() const;
      long nCols() const;
      const mcPRL::SpaceDims& dims() const;
      long iRowBegin() const;
      long iColBegin() const;
      long iRowEnd() const;
      long iColEnd() const;
      const mcPRL::CoordBR& MBR() const;
      long iRowWorkBegin() const;
      long iColWorkBegin() const;
      long iRowWorkEnd() const;
      long iColWorkEnd() const;
      const mcPRL::CoordBR& workBR() const;
      double sizeRatio() const;
      
      bool validGlbIdx(long glbIdx,
                       bool warning = true) const;
      bool validGlbCoord(const mcPRL::CellCoord &glbCoord,
                         bool warning = true) const;
      bool validGlbCoord(long iRowGlb, long iColGlb,
                         bool warning = true) const;
      const mcPRL::CellCoord glbIdx2glbCoord(long glbIdx) const;
      long glbCoord2glbIdx(const mcPRL::CellCoord &glbCoord) const;
      long glbCoord2glbIdx(long iRowGlb, long iColGlb) const;

      bool validIdx(long idx,
                    bool warning = true) const;
      bool validCoord(const mcPRL::CellCoord &coord,
                      bool warning = true) const;
      bool validCoord(long iRow, long iCol,
                      bool warning = true) const;
      long coord2idx(const mcPRL::CellCoord &coord) const;
      long coord2idx(long iRow, long iCol) const;
      const mcPRL::CellCoord idx2coord(long idx) const;

      const mcPRL::CellCoord lclCoord2glbCoord(const mcPRL::CellCoord& lclCoord) const;
      const mcPRL::CellCoord lclCoord2glbCoord(long iRowLcl, long iColLcl) const;
      const mcPRL::CellCoord glbCoord2lclCoord(const mcPRL::CellCoord& glbCoord) const;
      const mcPRL::CellCoord glbCoord2lclCoord(long iRowGlb, long iColGlb) const;
      long lclCoord2glbIdx(const mcPRL::CellCoord &lclCoord) const;
      long lclCoord2glbIdx(long iRowLcl, long iColLcl) const;
      const mcPRL::CellCoord glbIdx2lclCoord(long glbIdx) const;
      long glbIdx2lclIdx(long glbIdx) const;
      long lclIdx2glbIdx(long lclIdx) const;

      int nNbrDirs() const;
      int nEdges() const;
      int nTotNbrs() const;
      bool validSpcDir(int iDir) const;
      bool hasNbrs(int iDir) const;
      int nNbrs(int iDir) const;
      const mcPRL::IntVect& nbrSubSpcIDs(int iDir) const;
      int nbrDir(int nbrID) const;
      mcPRL::MeshDir spcDir2MeshDir(int iDir) const;
      int oppositeDir(int iDir) const;

      bool calcBRs(bool onlyUpdtCtrCell = true,
                   const mcPRL::Neighborhood *pNbrhd = NULL,
                   const list<mcPRL::SubCellspaceInfo> *plSubspcInfos = NULL);
      const mcPRL::CoordBR* interiorBR() const;
      const mcPRL::CoordBR* edgeBR(int iDir) const;
      const mcPRL::CoordBR* sendBR(int iDir,
                                  int iNbr = 0) const;

      /*
      void add2IntVect(mcPRL::IntVect &vInfoPack) const;
      bool fromIntVect(const mcPRL::IntVect &vInfoPack,
                       mcPRL::IntVctItr &iVal);
      */

    protected:
      void _initBRs();
      bool _rowwiseBRs(bool onlyUpdtCtrCell = true,
                       const mcPRL::Neighborhood *pNbrhd = NULL);
      bool _colwiseBRs(bool onlyUpdtCtrCell = true,
                       const mcPRL::Neighborhood *pNbrhd = NULL);
      bool _blockwiseBRs(bool onlyUpdtCtrCell = true,
                         const mcPRL::Neighborhood *pNbrhd = NULL,
                         const list<mcPRL::SubCellspaceInfo> *plSubspcInfos = NULL);
      const mcPRL::SubCellspaceInfo* _findSubspcInfo(const list<mcPRL::SubCellspaceInfo> *plSubspcInfos,
                                                    int subspcGlbID) const;

    protected:
      int _id;
      mcPRL::DomDcmpType _domDcmpType;
      mcPRL::SpaceDims _glbDims;
      mcPRL::CoordBR _MBR; /* MBR is in global coordinates */
      mcPRL::SpaceDims _dims;
      mcPRL::CoordBR _workBR; /* workBR is in local coordinates */
      vector<mcPRL::IntVect> _mNbrSpcIDs;

      vector<vector<CoordBR> > _mSendBRs;
      vector<CoordBR> _vEdgeBRs;
      CoordBR _interiorBR;
  };
};

#endif
