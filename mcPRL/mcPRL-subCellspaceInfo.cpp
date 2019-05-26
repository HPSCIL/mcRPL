#include "mcPRL-subCellspaceInfo.h"

/***************************************************************************
* mcPRL-subCellspaceInfo.cpp
*
* Project: pRPL, v 2.0
* Purpose: Implementation for class pRPL::SubCellspaceInfo
* Author:  Qingfeng (Gene) Guan
* E-mail:  guanqf {at} gmail.com
****************************************************************************
* Copyright (c) 2008, Qingfeng Guan
* NOTE: this library can ONLY be used for EDUCATIONAL and SCIENTIFIC 
* purposes, NO COMMERCIAL usages are allowed unless the author is 
* contacted and a permission is granted
* 
****************************************************************************/

void mcPRL::SubCellspaceInfo::
_initBRs() {
  _mSendBRs.clear();
  _mSendBRs.resize(nNbrDirs());
  _vEdgeBRs.clear();
  _vEdgeBRs.resize(nEdges());
  _interiorBR = CoordBR();
}

bool mcPRL::SubCellspaceInfo::
_rowwiseBRs(bool onlyUpdtCtrCell,
            const mcPRL::Neighborhood *pNbrhd) {
  long maxICol = (pNbrhd == NULL)?0:pNbrhd->maxICol();
  long minICol = (pNbrhd == NULL)?0:pNbrhd->minICol();
  long maxIRow = (pNbrhd == NULL)?0:pNbrhd->maxIRow();
  long minIRow = (pNbrhd == NULL)?0:pNbrhd->minIRow();

  long iRowWorkBegin = mcPRL::SubCellspaceInfo::iRowWorkBegin();
  long iColWorkBegin = mcPRL::SubCellspaceInfo::iColWorkBegin();
  long iRowWorkEnd = mcPRL::SubCellspaceInfo::iRowWorkEnd();
  long iColWorkEnd = mcPRL::SubCellspaceInfo::iColWorkEnd();

  if(hasNbrs(UPPER_DIR)) {
    _mSendBRs[UPPER_DIR].push_back(CoordBR());
  }
  if(hasNbrs(UPPER_DIR) && maxIRow > 0) {
    if(onlyUpdtCtrCell) {
      _mSendBRs[UPPER_DIR][0].nwCorner(iRowWorkBegin, iColWorkBegin);
      _mSendBRs[UPPER_DIR][0].seCorner(iRowWorkBegin - 1 + maxIRow,
                                       iColWorkEnd);
      _vEdgeBRs[UPPER_DIR].nwCorner(iRowWorkBegin, iColWorkBegin);
      _vEdgeBRs[UPPER_DIR].seCorner(iRowWorkBegin - 1 + maxIRow,
                                    iColWorkEnd);
      _interiorBR.nwCorner(iRowWorkBegin + maxIRow,
                           iColWorkBegin);
    }
    else {
      _mSendBRs[UPPER_DIR][0].nwCorner(0, 0);
      _mSendBRs[UPPER_DIR][0].seCorner(iRowWorkBegin - 1 + maxIRow,
                                       _dims.nCols() - 1);
      _vEdgeBRs[UPPER_DIR].nwCorner(iRowWorkBegin, iColWorkBegin);
      _vEdgeBRs[UPPER_DIR].seCorner(iRowWorkBegin - 1 + maxIRow - minIRow,
                                    iColWorkEnd);
      _interiorBR.nwCorner(iRowWorkBegin + maxIRow - minIRow,
                           iColWorkBegin);
    }
  }
  else {
    _interiorBR.nwCorner(iRowWorkBegin, iColWorkBegin);
  }

  if(hasNbrs(LOWER_DIR)) {
    _mSendBRs[LOWER_DIR].push_back(CoordBR());
  }
  if(hasNbrs(LOWER_DIR) && minIRow < 0) {
    if(onlyUpdtCtrCell) {
      _mSendBRs[LOWER_DIR][0].nwCorner(iRowWorkEnd + 1 + minIRow,
                                       iColWorkBegin);
      _mSendBRs[LOWER_DIR][0].seCorner(iRowWorkEnd, iColWorkEnd);
      _vEdgeBRs[LOWER_DIR].nwCorner(iRowWorkEnd + 1 + minIRow,
                                    iColWorkBegin);
      _vEdgeBRs[LOWER_DIR].seCorner(iRowWorkEnd, iColWorkEnd);
      _interiorBR.seCorner(iRowWorkEnd + minIRow,
                           iColWorkEnd);
    }
    else {
      _mSendBRs[LOWER_DIR][0].nwCorner(iRowWorkEnd + 1 + minIRow,
                                       0);
      _mSendBRs[LOWER_DIR][0].seCorner(_dims.nRows() - 1,
                                       _dims.nCols() - 1);
      _vEdgeBRs[LOWER_DIR].nwCorner(iRowWorkEnd + 1 + minIRow - maxIRow,
                                    iColWorkBegin);
      _vEdgeBRs[LOWER_DIR].seCorner(iRowWorkEnd, iColWorkEnd);
      _interiorBR.seCorner(iRowWorkEnd + minIRow - maxIRow,
                           iColWorkEnd);
    }
  }
  else {
    _interiorBR.seCorner(iRowWorkEnd, iColWorkEnd);
  }

  return true;
}

bool mcPRL::SubCellspaceInfo::
_colwiseBRs(bool onlyUpdtCtrCell,
            const mcPRL::Neighborhood *pNbrhd) {
  long maxICol = (pNbrhd == NULL)?0:pNbrhd->maxICol();
  long minICol = (pNbrhd == NULL)?0:pNbrhd->minICol();
  long maxIRow = (pNbrhd == NULL)?0:pNbrhd->maxIRow();
  long minIRow = (pNbrhd == NULL)?0:pNbrhd->minIRow();

  long iRowWorkBegin = mcPRL::SubCellspaceInfo::iRowWorkBegin();
  long iColWorkBegin = mcPRL::SubCellspaceInfo::iColWorkBegin();
  long iRowWorkEnd = mcPRL::SubCellspaceInfo::iRowWorkEnd();
  long iColWorkEnd = mcPRL::SubCellspaceInfo::iColWorkEnd();

  if(hasNbrs(LEFT_DIR)) {
    _mSendBRs[LEFT_DIR].push_back(CoordBR());
  }
  if(hasNbrs(LEFT_DIR) && maxICol > 0) {
    if(onlyUpdtCtrCell) {
      _mSendBRs[LEFT_DIR][0].nwCorner(iRowWorkBegin, iColWorkBegin);
      _mSendBRs[LEFT_DIR][0].seCorner(iRowWorkEnd,
                                      iColWorkBegin - 1 + maxICol);
      _vEdgeBRs[LEFT_DIR].nwCorner(iRowWorkBegin, iColWorkBegin);
      _vEdgeBRs[LEFT_DIR].seCorner(iRowWorkEnd,
                                   iColWorkBegin - 1 + maxICol);
      _interiorBR.nwCorner(iRowWorkBegin,
                           iColWorkBegin + maxICol);
    }
    else {
      _mSendBRs[LEFT_DIR][0].nwCorner(0, 0);
      _mSendBRs[LEFT_DIR][0].seCorner(_dims.nRows() - 1,
                                      iColWorkBegin - 1 + maxICol);
      _vEdgeBRs[LEFT_DIR].nwCorner(iRowWorkBegin, iColWorkBegin);
      _vEdgeBRs[LEFT_DIR].seCorner(iRowWorkEnd,
                                   iColWorkBegin - 1 + maxICol - minICol);
      _interiorBR.nwCorner(iRowWorkBegin,
                           iColWorkBegin + maxICol - minICol);
    }
  }
  else {
    _interiorBR.nwCorner(iRowWorkBegin, iColWorkBegin);
  }

  if(hasNbrs(RIGHT_DIR)) {
    _mSendBRs[RIGHT_DIR].push_back(CoordBR());
  }
  if(hasNbrs(RIGHT_DIR) && minICol < 0) {
    if(onlyUpdtCtrCell) {
      _mSendBRs[RIGHT_DIR][0].nwCorner(iRowWorkBegin,
                                       iColWorkEnd + 1 + minICol);
      _mSendBRs[RIGHT_DIR][0].seCorner(iRowWorkEnd, iColWorkEnd);
      _vEdgeBRs[RIGHT_DIR].nwCorner(iRowWorkBegin,
                                    iColWorkEnd + 1 + minICol);
      _vEdgeBRs[RIGHT_DIR].seCorner(iRowWorkEnd, iColWorkEnd);
      _interiorBR.seCorner(iRowWorkEnd,
                           iColWorkEnd + minICol);
    }
    else {
      _mSendBRs[RIGHT_DIR][0].nwCorner(0,
                                       iColWorkEnd + 1 + minICol);
      _mSendBRs[RIGHT_DIR][0].seCorner(_dims.nRows() - 1,
                                       _dims.nCols() - 1);
      _vEdgeBRs[RIGHT_DIR].nwCorner(iRowWorkBegin,
                                    iColWorkEnd + 1 + minICol - maxICol);
      _vEdgeBRs[RIGHT_DIR].seCorner(iRowWorkEnd, iColWorkEnd);
      _interiorBR.seCorner(iRowWorkEnd,
                           iColWorkEnd + minICol - maxICol);
    }
  }
  else {
    _interiorBR.seCorner(iRowWorkEnd, iColWorkEnd);
  }

  return true;
}

bool mcPRL::SubCellspaceInfo::
_blockwiseBRs(bool onlyUpdtCtrCell,
              const mcPRL::Neighborhood *pNbrhd,
              const list<mcPRL::SubCellspaceInfo> *plSubspcInfos) {
  long maxICol = (pNbrhd == NULL)?0:pNbrhd->maxICol();
  long minICol = (pNbrhd == NULL)?0:pNbrhd->minICol();
  long maxIRow = (pNbrhd == NULL)?0:pNbrhd->maxIRow();
  long minIRow = (pNbrhd == NULL)?0:pNbrhd->minIRow();

  long iRowWorkBegin = mcPRL::SubCellspaceInfo::iRowWorkBegin();
  long iColWorkBegin = mcPRL::SubCellspaceInfo::iColWorkBegin();
  long iRowWorkEnd = mcPRL::SubCellspaceInfo::iRowWorkEnd();
  long iColWorkEnd = mcPRL::SubCellspaceInfo::iColWorkEnd();

  long iRowBegin = mcPRL::SubCellspaceInfo::iRowBegin();
  long iColBegin = mcPRL::SubCellspaceInfo::iColBegin();
  long iRowEnd = mcPRL::SubCellspaceInfo::iRowEnd();
  long iColEnd = mcPRL::SubCellspaceInfo::iColEnd();

  long nRows = mcPRL::SubCellspaceInfo::nRows();
  long nCols = mcPRL::SubCellspaceInfo::nCols();

  for(int iDir = 0; iDir < 8; iDir++) {
    _mSendBRs[iDir].resize(nNbrs(iDir));
  }

  int iRowNIntr, iRowSIntr, iColWIntr, iColEIntr;
  if(hasNbrs(NORTH_DIR) && maxIRow > 0) {
    const IntVect &vNbrIDs = nbrSubSpcIDs(NORTH_DIR);
    for(int iNbr = 0; iNbr < vNbrIDs.size(); iNbr++) {
      int nbrSpcID = vNbrIDs[iNbr];
      const mcPRL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
      if(pNbrInfo == NULL) {
        cerr << __FILE__ << " " << __FUNCTION__ \
            << " Error: unable to find neighboring SubSpace [" \
            << nbrSpcID << "]" << endl;
        return false;
      }
      int iNbrColBegin = pNbrInfo->iColBegin();
      iNbrColBegin = (iNbrColBegin <= iColBegin) ? 0 : (iNbrColBegin - iColBegin);
      _mSendBRs[NORTH_DIR][iNbr].nwCorner(0, iNbrColBegin);
      int iNbrColEnd = pNbrInfo->iColEnd();
      iNbrColEnd = (iNbrColEnd >= iColEnd) ? (nCols - 1) : (iNbrColEnd - iColBegin);
      _mSendBRs[NORTH_DIR][iNbr].seCorner(iRowWorkBegin - 1 + maxIRow,
                                          iNbrColEnd);
    }
    if(onlyUpdtCtrCell) {
      _vEdgeBRs[NORTH_PDIR].nwCorner(iRowWorkBegin, iColWorkBegin);
      _vEdgeBRs[NORTH_PDIR].seCorner(iRowWorkBegin - 1 + maxIRow,
                                     iColWorkEnd);
      iRowNIntr = iRowWorkBegin + maxIRow;
    }
    else {
      _vEdgeBRs[NORTH_PDIR].nwCorner(iRowWorkBegin, iColWorkBegin);
      _vEdgeBRs[NORTH_PDIR].seCorner(iRowWorkBegin - 1 + maxIRow - minIRow,
                                     iColWorkEnd);
      iRowNIntr = iRowWorkBegin + maxIRow - minIRow;
    }
  }
  else {
    iRowNIntr = iRowWorkBegin;
  }

  if(hasNbrs(SOUTH_DIR) && minIRow < 0) {
    const IntVect &vNbrIDs = nbrSubSpcIDs(SOUTH_DIR);
    for(int iNbr = 0; iNbr < vNbrIDs.size(); iNbr++) {
      int nbrSpcID = vNbrIDs[iNbr];
      const mcPRL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
      if(pNbrInfo == NULL) {
        cerr << __FILE__ << " " << __FUNCTION__ \
            << " Error: unable to find neighboring SubSpace [" \
            << nbrSpcID << "]" << endl;
        return false;
      }
      int iNbrColBegin = pNbrInfo->iColBegin();
      iNbrColBegin = (iNbrColBegin <= iColBegin) ? 0 : (iNbrColBegin - iColBegin);
      _mSendBRs[SOUTH_DIR][iNbr].nwCorner(iRowWorkEnd + 1 + minIRow,
                                          iNbrColBegin);
      int iNbrColEnd = pNbrInfo->iColEnd();
      iNbrColEnd = (iNbrColEnd >= iColEnd) ? (nCols - 1) : (iNbrColEnd - iColBegin);
      _mSendBRs[SOUTH_DIR][iNbr].seCorner(nRows - 1, iNbrColEnd);
    }
    if(onlyUpdtCtrCell) {
      _vEdgeBRs[SOUTH_PDIR].nwCorner(iRowWorkEnd + 1 + minIRow,
                                     iColWorkBegin);
      _vEdgeBRs[SOUTH_PDIR].seCorner(iRowWorkEnd, iColWorkEnd);
      iRowSIntr = iRowWorkEnd + minIRow;
    }
    else {
      _vEdgeBRs[SOUTH_PDIR].nwCorner(iRowWorkEnd + 1 + minIRow - maxIRow,
                                     iColWorkBegin);
      _vEdgeBRs[SOUTH_PDIR].seCorner(iRowWorkEnd, iColWorkEnd);
      iRowSIntr = iRowWorkEnd + minIRow - maxIRow;
    }
  }
  else {
    iRowSIntr = iRowWorkEnd;
  }

  if(hasNbrs(WEST_DIR) && maxICol > 0) {
    const IntVect &vNbrIDs = nbrSubSpcIDs(WEST_DIR);
    for(int iNbr = 0; iNbr < vNbrIDs.size(); iNbr++) {
      int nbrSpcID = vNbrIDs[iNbr];
      const mcPRL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
      if(pNbrInfo == NULL) {
        cerr << __FILE__ << " " << __FUNCTION__ \
            << " Error: unable to find neighboring SubSpace [" \
            << nbrSpcID << "]" << endl;
        return false;
      }
      int iNbrRowBegin = pNbrInfo->iRowBegin();
      iNbrRowBegin = (iNbrRowBegin <= iRowBegin) ? 0 : (iNbrRowBegin - iRowBegin);
      _mSendBRs[WEST_DIR][iNbr].nwCorner(iNbrRowBegin, 0);
      int iNbrRowEnd = pNbrInfo->iRowEnd();
      iNbrRowEnd = (iNbrRowEnd >= iRowEnd) ? (nRows - 1) : (iNbrRowEnd - iRowBegin);
      _mSendBRs[WEST_DIR][iNbr].seCorner(iNbrRowEnd,
                                         iColWorkBegin - 1 + maxICol);
    }
    if(onlyUpdtCtrCell) {
      _vEdgeBRs[WEST_PDIR].nwCorner(iRowNIntr, iColWorkBegin);
      _vEdgeBRs[WEST_PDIR].seCorner(iRowSIntr,
                                    iColWorkBegin - 1 + maxICol);
      iColWIntr = iColWorkBegin + maxICol;
    }
    else {
      _vEdgeBRs[WEST_PDIR].nwCorner(iRowNIntr, iColWorkBegin);
      _vEdgeBRs[WEST_PDIR].seCorner(iRowSIntr,
                                    iColWorkBegin - 1 + maxICol - minICol);
      iColWIntr = iColWorkBegin + maxICol - minICol;
    }
  }
  else {
    iColWIntr = iColWorkBegin;
  }

  if(hasNbrs(EAST_DIR) && minICol < 0) {
    const IntVect &vNbrIDs = nbrSubSpcIDs(EAST_DIR);
    for(int iNbr = 0; iNbr < vNbrIDs.size(); iNbr++) {
      int nbrSpcID = vNbrIDs[iNbr];
      const mcPRL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
      if(pNbrInfo == NULL) {
        cerr << __FILE__ << " " << __FUNCTION__ \
            << " Error: unable to find neighboring SubSpace [" \
            << nbrSpcID << "]" << endl;
        return false;
      }
      int iNbrRowBegin = pNbrInfo->iRowBegin();
      iNbrRowBegin = (iNbrRowBegin <= iRowBegin) ? 0 : (iNbrRowBegin - iRowBegin);
      _mSendBRs[EAST_DIR][iNbr].nwCorner(iNbrRowBegin,
                                         iColWorkEnd + 1 + minICol);
      int iNbrRowEnd = pNbrInfo->iRowEnd();
      iNbrRowEnd = (iNbrRowEnd >= iRowEnd) ? (nRows - 1) : (iNbrRowEnd - iRowBegin);
      _mSendBRs[EAST_DIR][iNbr].seCorner(iNbrRowEnd,
                                         nCols - 1);
    }
    if(onlyUpdtCtrCell) {
      _vEdgeBRs[EAST_PDIR].nwCorner(iRowNIntr,
                                    iColWorkEnd + 1 + minICol);
      _vEdgeBRs[EAST_PDIR].seCorner(iRowSIntr, iColWorkEnd);
      iColEIntr = iColWorkEnd + minICol;
    }
    else {
      _vEdgeBRs[EAST_PDIR].nwCorner(iRowNIntr,
                                    iColWorkEnd + 1 + minICol - maxICol);
      _vEdgeBRs[EAST_PDIR].seCorner(iRowSIntr, iColWorkEnd);
      iColEIntr = iColWorkEnd + minICol - maxICol;
    }
  }
  else {
    iColEIntr = iColWorkEnd;
  }

  _interiorBR.nwCorner(iRowNIntr, iColWIntr);
  _interiorBR.seCorner(iRowSIntr, iColEIntr);

  if(hasNbrs(NORTHEAST_DIR) &&
     pNbrhd != NULL &&
     pNbrhd->hasNbrs(SOUTHWEST_DIR)) {
    _mSendBRs[NORTHEAST_DIR][0].nwCorner(0,
                                         iColWorkEnd + minICol + 1);
    _mSendBRs[NORTHEAST_DIR][0].seCorner(iRowWorkBegin + maxIRow -1,
                                         nCols - 1);
  }
  if(hasNbrs(SOUTHEAST_DIR) &&
     pNbrhd != NULL &&
     pNbrhd->hasNbrs(NORTHWEST_DIR)) {
    _mSendBRs[SOUTHEAST_DIR][0].nwCorner(iRowWorkEnd + minIRow + 1,
                                         iColWorkEnd + minICol + 1);
    _mSendBRs[SOUTHEAST_DIR][0].seCorner(nRows - 1, nCols - 1);
  }
  if(hasNbrs(SOUTHWEST_DIR) &&
     pNbrhd != NULL &&
     pNbrhd->hasNbrs(NORTHEAST_DIR)) {
    _mSendBRs[SOUTHWEST_DIR][0].nwCorner(iRowWorkEnd + minIRow + 1,
                                         0);
    _mSendBRs[SOUTHWEST_DIR][0].seCorner(nRows - 1,
                                         iColWorkBegin + maxICol - 1);
  }
  if(hasNbrs(NORTHWEST_DIR) &&
     pNbrhd != NULL &&
     pNbrhd->hasNbrs(SOUTHEAST_DIR)) {
    _mSendBRs[NORTHWEST_DIR][0].nwCorner(0, 0);
    _mSendBRs[NORTHWEST_DIR][0].seCorner(iRowWorkBegin + maxIRow - 1,
                                         iColWorkBegin + maxICol - 1);
  }

  return true;
}

const mcPRL::SubCellspaceInfo* mcPRL::SubCellspaceInfo::
_findSubspcInfo(const list<mcPRL::SubCellspaceInfo> *plSubspcInfos,
                int subspcGlbID) const{
  const mcPRL::SubCellspaceInfo* pSubInfo = NULL;
  list<mcPRL::SubCellspaceInfo>::const_iterator itrSubInfo = plSubspcInfos->begin();
  while(itrSubInfo != plSubspcInfos->end()) {
    if(subspcGlbID == itrSubInfo->id()) {
      pSubInfo = &(*itrSubInfo);
      break;
    }
    itrSubInfo++;
  }
  return pSubInfo;
}

mcPRL::SubCellspaceInfo::
SubCellspaceInfo()
  :_id(mcPRL::ERROR_ID),
   _domDcmpType(mcPRL::NON_DCMP) {}

mcPRL::SubCellspaceInfo::
SubCellspaceInfo(int id,
                 mcPRL::DomDcmpType domDcmpType,
                 const mcPRL::SpaceDims &glbDims,
                 const mcPRL::CoordBR &MBR,
                 const mcPRL::CoordBR &workBR,
                 const vector<mcPRL::IntVect> &mNbrSpcIDs)
 :_domDcmpType(domDcmpType) {
  if(id < 0) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid SubSpace ID (" \
         << id << ")" << endl;
    exit(-1);
  }
  _id = id;

  if(!glbDims.valid()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspaceInfo[" << _id \
         << "] invalid glbDims (" << glbDims
         << ")" << endl;
    exit(-1);
  }
  _glbDims = glbDims;

  if(!MBR.valid(glbDims)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspaceInfo[" << _id \
         << "] invalid MBR (" \
         << MBR << ") with the global dimensions (" \
         << glbDims << ")" << endl;
    exit(-1);
  }
  _MBR = MBR;
  _dims = SpaceDims(MBR.nRows(), MBR.nCols());

  if(!workBR.valid(_dims)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspaceInfo[" << _id \
         << "] invalid working range (" \
         << workBR << ") within the local dimensions (" \
         << _dims << ")" << endl;
    exit(-1);
  }
  _workBR = workBR;

  int numNbrDirs = nNbrDirs();
  if(numNbrDirs != mNbrSpcIDs.size()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspaceInfo[" << _id \
         << "] invalid number of neighbor ID directions (" \
         << mNbrSpcIDs.size() << ")" << endl;
    exit(-1);
  }
  _mNbrSpcIDs = mNbrSpcIDs;
}

mcPRL::SubCellspaceInfo::
SubCellspaceInfo(const mcPRL::SubCellspaceInfo &rhs)
  :_id(rhs._id),
   _domDcmpType(rhs._domDcmpType),
   _glbDims(rhs._glbDims),
   _dims(rhs._dims),
   _MBR(rhs._MBR),
   _workBR(rhs._workBR),
   _mNbrSpcIDs(rhs._mNbrSpcIDs) {}

/*
mcPRL::SubCellspaceInfo::
SubCellspaceInfo(const mcPRL::IntVect &vInfoPack,
                 mcPRL::IntVctItr &iVal) {
  if(!fromIntVect(vInfoPack, iVal)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: unable to construct a SubCellspaceInfo" << endl;
    exit(-1);
  }

  cout << "spc[" << _id << "]" << endl \
       << "domDcmpType " << _domDcmpType << endl \
       << "glbDims " << _glbDims << endl \
       << "dims " << _dims << endl \
       << "MBR " << _MBR << endl \
       << "workBR " << _workBR << endl;
  for(int iDir = 0; iDir < nNbrDirs(); iDir++) {
    cout << "dir[" << iDir << "]" << endl;
    for(int iNbr = 0; iNbr < nNbrs(iDir); iNbr++) {
      cout << "nbr id " << _mNbrSpcIDs[iDir][iNbr] << endl;
    }
  }

}
*/

int mcPRL::SubCellspaceInfo::
id() const {
  return _id;
}

mcPRL::DomDcmpType mcPRL::SubCellspaceInfo::
domDcmpType() const {
  return _domDcmpType;
}

long mcPRL::SubCellspaceInfo::
nGlbRows() const {
  return _glbDims.nRows();
}

long mcPRL::SubCellspaceInfo::
nGlbCols() const {
  return _glbDims.nCols();
}

const mcPRL::SpaceDims& mcPRL::SubCellspaceInfo::
glbDims() const {
  return _glbDims;
}

long mcPRL::SubCellspaceInfo::
nRows() const {
  return _dims.nRows();
}

long mcPRL::SubCellspaceInfo::
nCols() const {
  return _dims.nCols();
}

const mcPRL::SpaceDims& mcPRL::SubCellspaceInfo::
dims() const {
  return _dims;
}

long mcPRL::SubCellspaceInfo::
iRowBegin() const {
  return _MBR.minIRow();
}

long mcPRL::SubCellspaceInfo::
iColBegin() const {
  return _MBR.minICol();
}

long mcPRL::SubCellspaceInfo::
iRowEnd() const {
  return _MBR.maxIRow();
}

long mcPRL::SubCellspaceInfo::
iColEnd() const {
  return _MBR.maxICol();
}

const mcPRL::CoordBR& mcPRL::SubCellspaceInfo::
MBR() const {
  return _MBR;
}

long mcPRL::SubCellspaceInfo::
iRowWorkBegin() const {
  return _workBR.minIRow();
}

long mcPRL::SubCellspaceInfo::
iColWorkBegin() const {
  return _workBR.minICol();
}

long mcPRL::SubCellspaceInfo::
iRowWorkEnd() const {
  return _workBR.maxIRow();
}

long mcPRL::SubCellspaceInfo::
iColWorkEnd() const {
  return _workBR.maxICol();
}

const mcPRL::CoordBR& mcPRL::SubCellspaceInfo::
workBR() const {
  return _workBR;
}

mcPRL::SubCellspaceInfo& mcPRL::SubCellspaceInfo::
operator=(const mcPRL::SubCellspaceInfo &rhs) {
  if(this != &rhs) {
    _id = rhs._id;
    _domDcmpType = rhs._domDcmpType;
    _glbDims = rhs._glbDims;
    _dims = rhs._dims;
    _MBR = rhs._MBR;
    _workBR = rhs._workBR;
    _mNbrSpcIDs = rhs._mNbrSpcIDs;
  }
  return *this;
}

bool mcPRL::SubCellspaceInfo::
operator==(const mcPRL::SubCellspaceInfo &rhs) const {
  return (_id == rhs._id &&
          _domDcmpType == rhs._domDcmpType &&
          _glbDims == rhs._glbDims &&
          _dims == rhs._dims &&
          _MBR == rhs._MBR &&
          _workBR == rhs._workBR &&
          _mNbrSpcIDs == rhs._mNbrSpcIDs);
}

bool mcPRL::SubCellspaceInfo::
operator!=(const mcPRL::SubCellspaceInfo &rhs) const {
  return !(operator==(rhs));
}

double mcPRL::SubCellspaceInfo::
sizeRatio() const {
  return (double)(_dims.size()) / (double)(_glbDims.size());
}

bool mcPRL::SubCellspaceInfo::
validGlbIdx(long glbIdx,
            bool warning) const {
  bool valid = true;
  if(!_glbDims.validIdx(glbIdx)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " " << __FUNCTION__ \
           << " Error: in SubCellspace[" << _id \
           << "], glbIdx [" << glbIdx \
           << "] out of global boundary [" \
           << _glbDims << "]" << endl;
    }
  }
  return valid;
}

bool mcPRL::SubCellspaceInfo::
validGlbCoord(const mcPRL::CellCoord &glbCoord,
              bool warning) const {
  bool valid = true;
  if(!glbCoord.valid(_glbDims)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " " << __FUNCTION__ \
           << " Error: in SubCellspace[" << _id \
           << "], glbCoord [" \
           << glbCoord << "] out of global boundary [" \
           << _glbDims << "]" << endl;
    }
  }
  return valid; 
}

bool mcPRL::SubCellspaceInfo::
validGlbCoord(long iRowGlb, long iColGlb,
              bool warning) const {
  return validGlbCoord(mcPRL::CellCoord(iRowGlb, iColGlb), warning);
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
glbIdx2glbCoord(long glbIdx) const {
  return mcPRL::CellCoord(glbIdx, _glbDims);
}

long mcPRL::SubCellspaceInfo::
glbCoord2glbIdx(const mcPRL::CellCoord &glbCoord) const {
  return glbCoord.toIdx(_glbDims);
}

long mcPRL::SubCellspaceInfo::
glbCoord2glbIdx(long iRowGlb, long iColGlb) const {
  return glbCoord2glbIdx(mcPRL::CellCoord(iRowGlb, iColGlb));
}

bool mcPRL::SubCellspaceInfo::
validIdx(long idx,
         bool warning) const {
  bool valid = true;
  if(!_dims.validIdx(idx)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " " << __FUNCTION__ \
           << " Error: in SubCellspace[" << _id \
           << "], index [" << idx \
           << "] out of boundary [" \
           << _dims << "]" << endl;
    }
  }
  return valid;
}

bool mcPRL::SubCellspaceInfo::
validCoord(const mcPRL::CellCoord &coord,
           bool warning) const {
  bool valid = true;
  if(!coord.valid(_dims)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " " << __FUNCTION__ \
           << " Error: in SubCellspace[" << _id \
           << "], coordinate [" \
           << coord << "] out of boundary [" \
           << _dims << "]" << endl;
    }
  }
  return valid;
}

bool mcPRL::SubCellspaceInfo::
validCoord(long iRow, long iCol,
           bool warning) const {
  return validCoord(mcPRL::CellCoord(iRow, iCol), warning);
}

long mcPRL::SubCellspaceInfo::
coord2idx(const mcPRL::CellCoord &coord) const {
  return coord.toIdx(_dims);
}

long mcPRL::SubCellspaceInfo::
coord2idx(long iRow, long iCol) const {
  return coord2idx(mcPRL::CellCoord(iRow, iCol));
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
idx2coord(long idx) const {
  return mcPRL::CellCoord(idx, _dims);
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
lclCoord2glbCoord(const mcPRL::CellCoord& lclCoord) const {
  return (lclCoord + _MBR.nwCorner()); 
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
lclCoord2glbCoord(long iRowLcl, long iColLcl) const {
  return lclCoord2glbCoord(mcPRL::CellCoord(iRowLcl, iColLcl));
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
glbCoord2lclCoord(const mcPRL::CellCoord& glbCoord) const {
  return (glbCoord - _MBR.nwCorner());
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
glbCoord2lclCoord(long iRowGlb, long iColGlb) const {
  return glbCoord2lclCoord(mcPRL::CellCoord(iRowGlb, iColGlb));
}

long mcPRL::SubCellspaceInfo::
lclCoord2glbIdx(const mcPRL::CellCoord &lclCoord) const {
  return glbCoord2glbIdx(lclCoord2glbCoord(lclCoord));
}

long mcPRL::SubCellspaceInfo::
lclCoord2glbIdx(long iRowLcl, long iColLcl) const {
  return lclCoord2glbIdx(mcPRL::CellCoord(iRowLcl, iColLcl));
}

const mcPRL::CellCoord mcPRL::SubCellspaceInfo::
glbIdx2lclCoord(long glbIdx) const {
  return glbCoord2lclCoord(mcPRL::CellCoord(glbIdx, _glbDims));
}

long mcPRL::SubCellspaceInfo::
glbIdx2lclIdx(long glbIdx) const {
  return coord2idx(glbIdx2lclCoord(glbIdx));
}

long mcPRL::SubCellspaceInfo::
lclIdx2glbIdx(long lclIdx) const {
  return lclCoord2glbIdx(idx2coord(lclIdx));
}

int mcPRL::SubCellspaceInfo::
nNbrDirs() const {
  int nNbrDirects;
  switch(_domDcmpType) {
    case mcPRL::NON_DCMP:
      nNbrDirects = 0;
      break;
    case mcPRL::ROW_DCMP:
    case mcPRL::COL_DCMP:
      nNbrDirects = 2;
      break;
    case mcPRL::BLK_DCMP:
      nNbrDirects = 8;
      break;
    default:
      cerr << __FILE__ << __FUNCTION__ \
           << " Error: invalid decomposition type (" \
           << _domDcmpType << ")" << endl;
      nNbrDirects = -1;
      break;
  }
  return nNbrDirects;
}

int mcPRL::SubCellspaceInfo::
nEdges() const {
  int numEdges;
  switch(_domDcmpType) {
    case mcPRL::NON_DCMP:
      numEdges = 0;
      break;
    case mcPRL::ROW_DCMP:
    case mcPRL::COL_DCMP:
      numEdges = 2;
      break;
    case mcPRL::BLK_DCMP:
      numEdges = 4;
      break;
    default:
      cerr << __FILE__ << __FUNCTION__ \
           << " Error: invalid decomposition type (" \
           << _domDcmpType << ")" << endl;
      numEdges = -1;
      break;
  }
  return numEdges;
}

int mcPRL::SubCellspaceInfo::
nTotNbrs() const {
  int numNbrs = 0;
  vector<mcPRL::IntVect>::const_iterator iNbrMap = _mNbrSpcIDs.begin();
  while(iNbrMap != _mNbrSpcIDs.end()) {
    numNbrs += (*iNbrMap).size();
    iNbrMap++;
  }
  return numNbrs;
}

bool mcPRL::SubCellspaceInfo::
validSpcDir(int iDir) const {
  bool valid = true;
  if(iDir < 0 ||
     iDir >= nNbrDirs()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invliad direction (" << iDir \
         << ") of neighboring SubSpaces on SubCellspace[" \
         << _id << "]" << endl;
    valid = false;
  }
  return valid;
}

bool mcPRL::SubCellspaceInfo::
hasNbrs(int iDir) const {
  bool hasIt = true;
  if(!validSpcDir(iDir) ||
     _mNbrSpcIDs[iDir].size() == 0) {
    hasIt = false;
  }
  return hasIt;
}

int mcPRL::SubCellspaceInfo::
nNbrs(int iDir) const {
  int numNbrs = 0;
  if(validSpcDir(iDir)) {
    numNbrs = _mNbrSpcIDs[iDir].size();
  }
  return numNbrs;
}

const mcPRL::IntVect& mcPRL::SubCellspaceInfo::
nbrSubSpcIDs(int iDir) const {
  return _mNbrSpcIDs.at(iDir);
}

int mcPRL::SubCellspaceInfo::
nbrDir(int nbrID) const {
  int dir = -1;
  for(int iDir = 0; iDir < nNbrDirs(); iDir++) {
    const IntVect &vNbrs = nbrSubSpcIDs(iDir);
    if(std::find(vNbrs.begin(), vNbrs.end(), nbrID) != vNbrs.end()) {
      dir = iDir;
      break;
    }
  }
  return dir;
}

mcPRL::MeshDir mcPRL::SubCellspaceInfo::
spcDir2MeshDir(int iDir) const {
  MeshDir opDir = mcPRL::NONE_DIR;
  if(validSpcDir(iDir)) {
    switch(_domDcmpType) {
      case mcPRL::NON_DCMP:
        break;
      case mcPRL::ROW_DCMP:
        opDir = (iDir == mcPRL::UPPER_DIR) ? mcPRL::NORTH_DIR : mcPRL::SOUTH_DIR;
        break;
      case mcPRL::COL_DCMP:
        opDir = (iDir == mcPRL::LEFT_DIR) ? mcPRL::WEST_DIR : mcPRL::EAST_DIR;
        break;
      case mcPRL::BLK_DCMP:
        opDir = static_cast<mcPRL::MeshDir>(iDir);
        break;
      default:
        break;
    }
  }
  return opDir;
}

int mcPRL::SubCellspaceInfo::
oppositeDir(int iDir) const {
  int opDir = -1;
  if(validSpcDir(iDir)) {
    switch(_domDcmpType) {
      case mcPRL::NON_DCMP:
        break;
      case mcPRL::ROW_DCMP:
      case mcPRL::COL_DCMP:
        opDir = (iDir == 0) ? 1 : 0;
        break;
      case mcPRL::BLK_DCMP:
        opDir = static_cast<int>(mcPRL::oppositeDir(static_cast<mcPRL::MeshDir>(iDir)));
        break;
      default:
        break;
    }
  }
  return opDir;
}

bool mcPRL::SubCellspaceInfo::
calcBRs(bool onlyUpdtCtrCell,
        const mcPRL::Neighborhood *pNbrhd,
        const list<mcPRL::SubCellspaceInfo> *plSubspcInfos) {
  bool done = true;

  _initBRs();
  switch(_domDcmpType) {
    case mcPRL::NON_DCMP:
      break;
    case mcPRL::ROW_DCMP:
      done = _rowwiseBRs(onlyUpdtCtrCell, pNbrhd);
      break;
    case mcPRL::COL_DCMP:
      done = _colwiseBRs(onlyUpdtCtrCell, pNbrhd);
      break;
    case mcPRL::BLK_DCMP:
      if(plSubspcInfos == NULL) {
        cerr << __FILE__ << " " << __FUNCTION__ \
            << " Error: NULL pointer to the vector of SubCellspaceInfo" \
            << endl;
        done = false;
      }
      else {
        done = _blockwiseBRs(onlyUpdtCtrCell, pNbrhd, plSubspcInfos);
      }
      break;
    default:
      break;
  }

  return done;
}

const mcPRL::CoordBR* mcPRL::SubCellspaceInfo::
interiorBR() const {
  const mcPRL::CoordBR *pIntrBR = NULL;
  if(_interiorBR.valid(_dims)) {
    pIntrBR = &_interiorBR;
  }
  return pIntrBR;
}

const mcPRL::CoordBR* mcPRL::SubCellspaceInfo::
edgeBR(int iDir) const {
  const mcPRL::CoordBR *pEdgeBR = NULL;
  if(_vEdgeBRs.size() == nEdges() &&
     iDir >= 0 && iDir < nEdges()) {
    pEdgeBR = _vEdgeBRs[iDir].valid(_dims)?&(_vEdgeBRs[iDir]):NULL;
  }
  return pEdgeBR;
}

const mcPRL::CoordBR* mcPRL::SubCellspaceInfo::
sendBR(int iDir, int iNbr) const {
  const mcPRL::CoordBR *pSendBR = NULL;
  if(_mSendBRs.size() == nNbrDirs() &&
     validSpcDir(iDir) &&
     iNbr >= 0 && iNbr < nNbrs(iDir)) {
    pSendBR = _mSendBRs[iDir][iNbr].valid(_dims)?&(_mSendBRs[iDir][iNbr]):NULL;
  }
  /*
  else {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid Cellspace's neighboring direction (" \
         << iDir << ") or invalid neighborhood index (" \
         << iNbr << ")" << endl;
  }
  */
  return pSendBR;
}

/*
void mcPRL::SubCellspaceInfo::
add2IntVect(mcPRL::IntVect &vInfoPack) const {
  vInfoPack.push_back(_id);
  vInfoPack.push_back(_domDcmpType);
  vInfoPack.push_back(_glbDims.nRows());
  vInfoPack.push_back(_glbDims.nCols());
  vInfoPack.push_back(_MBR.nwCorner().toIdx(_glbDims));
  vInfoPack.push_back(_MBR.seCorner().toIdx(_glbDims));
  vInfoPack.push_back(_workBR.nwCorner().toIdx(_dims));
  vInfoPack.push_back(_workBR.seCorner().toIdx(_dims));

  for(int iDir = 0; iDir < nNbrDirs(); iDir++) {
    const mcPRL::IntVect &vNbrIDs = _mNbrSpcIDs[iDir];
    int nNbrs = vNbrIDs.size();
    vInfoPack.push_back(nNbrs);
    for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
      vInfoPack.push_back(vNbrIDs[iNbr]);
    }
  }
}

bool mcPRL::SubCellspaceInfo::
fromIntVect(const mcPRL::IntVect &vInfoPack,
            mcPRL::IntVctItr &iVal) {
  if(vInfoPack.end() - iVal < 8 ||
     iVal < vInfoPack.begin()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid iterator of the integer vector" \
         << " to construct a SubCellspaceInfo" << endl;
    return false;
  }

  _id = *iVal;
  iVal++;
  if(_id < 0) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid SubCellspace ID (" \
         << _id << ")" << endl;
    return false;
  }
  
  _domDcmpType = static_cast<mcPRL::DomDcmpType>(*iVal);
  iVal++;

  _glbDims.nRows(*iVal);
  iVal++;
  _glbDims.nCols(*iVal);
  iVal++;
  if(!_glbDims.valid()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspace[" << _id \
         << "] invalid global SpaceDims (" \
         << _glbDims << ")" << endl;
    return false;
  }

  _MBR.nwCorner(mcPRL::CellCoord(*iVal, _glbDims));
  iVal++;
  _MBR.seCorner(mcPRL::CellCoord(*iVal, _glbDims));
  iVal++;
  if(!_MBR.valid(_glbDims)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspace[" << _id \
         << "] invalid MBR (" << _MBR \
         << ")" << endl;
    return false;
  }
  _dims = mcPRL::SpaceDims(_MBR.nRows(), _MBR.nCols());

  _workBR.nwCorner(mcPRL::CellCoord(*iVal, _dims));
  iVal++;
  _workBR.seCorner(mcPRL::CellCoord(*iVal, _dims));
  iVal++;
  if(!_workBR.valid(_dims)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspace[" << _id \
         << "] invalid workBR (" \
         << _workBR << ")" << endl;
    return false;
  }
  
  int numNbrDirs = nNbrDirs();
  _mNbrSpcIDs.resize(numNbrDirs);
  for(int iDir = 0; iDir < numNbrDirs; iDir++) {
    int nNbrs = *iVal;
    iVal++;
    
    for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
      _mNbrSpcIDs[iDir].push_back(*iVal);
      iVal++;
    }
  }

  return true;
}
*/
