#include "mcrpl-subCellspaceInfo.h"



void mcRPL::SubCellspaceInfo::
_initBRs() {
  _mSendBRs.clear();
  _mSendBRs.resize(nNbrDirs());
  _vEdgeBRs.clear();
  _vEdgeBRs.resize(nEdges());
  _interiorBR = CoordBR();
}

bool mcRPL::SubCellspaceInfo::
_rowwiseBRs(bool onlyUpdtCtrCell,
            const mcRPL::Neighborhood *pNbrhd) {
  long maxICol = (pNbrhd == NULL)?0:pNbrhd->maxICol();
  long minICol = (pNbrhd == NULL)?0:pNbrhd->minICol();
  long maxIRow = (pNbrhd == NULL)?0:pNbrhd->maxIRow();
  long minIRow = (pNbrhd == NULL)?0:pNbrhd->minIRow();

  long iRowWorkBegin = mcRPL::SubCellspaceInfo::iRowWorkBegin();
  long iColWorkBegin = mcRPL::SubCellspaceInfo::iColWorkBegin();
  long iRowWorkEnd = mcRPL::SubCellspaceInfo::iRowWorkEnd();
  long iColWorkEnd = mcRPL::SubCellspaceInfo::iColWorkEnd();

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

bool mcRPL::SubCellspaceInfo::
_colwiseBRs(bool onlyUpdtCtrCell,
            const mcRPL::Neighborhood *pNbrhd) {
  long maxICol = (pNbrhd == NULL)?0:pNbrhd->maxICol();
  long minICol = (pNbrhd == NULL)?0:pNbrhd->minICol();
  long maxIRow = (pNbrhd == NULL)?0:pNbrhd->maxIRow();
  long minIRow = (pNbrhd == NULL)?0:pNbrhd->minIRow();

  long iRowWorkBegin = mcRPL::SubCellspaceInfo::iRowWorkBegin();
  long iColWorkBegin = mcRPL::SubCellspaceInfo::iColWorkBegin();
  long iRowWorkEnd = mcRPL::SubCellspaceInfo::iRowWorkEnd();
  long iColWorkEnd = mcRPL::SubCellspaceInfo::iColWorkEnd();

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

bool mcRPL::SubCellspaceInfo::
_blockwiseBRs(bool onlyUpdtCtrCell,
              const mcRPL::Neighborhood *pNbrhd,
              const list<mcRPL::SubCellspaceInfo> *plSubspcInfos) {
  long maxICol = (pNbrhd == NULL)?0:pNbrhd->maxICol();
  long minICol = (pNbrhd == NULL)?0:pNbrhd->minICol();
  long maxIRow = (pNbrhd == NULL)?0:pNbrhd->maxIRow();
  long minIRow = (pNbrhd == NULL)?0:pNbrhd->minIRow();

  long iRowWorkBegin = mcRPL::SubCellspaceInfo::iRowWorkBegin();
  long iColWorkBegin = mcRPL::SubCellspaceInfo::iColWorkBegin();
  long iRowWorkEnd = mcRPL::SubCellspaceInfo::iRowWorkEnd();
  long iColWorkEnd = mcRPL::SubCellspaceInfo::iColWorkEnd();

  long iRowBegin = mcRPL::SubCellspaceInfo::iRowBegin();
  long iColBegin = mcRPL::SubCellspaceInfo::iColBegin();
  long iRowEnd = mcRPL::SubCellspaceInfo::iRowEnd();
  long iColEnd = mcRPL::SubCellspaceInfo::iColEnd();

  long nRows = mcRPL::SubCellspaceInfo::nRows();
  long nCols = mcRPL::SubCellspaceInfo::nCols();

  for(int iDir = 0; iDir < 8; iDir++) {
    _mSendBRs[iDir].resize(nNbrs(iDir));
  }

  int iRowNIntr, iRowSIntr, iColWIntr, iColEIntr;
  if(hasNbrs(NORTH_DIR) && maxIRow > 0) {
    const IntVect &vNbrIDs = nbrSubSpcIDs(NORTH_DIR);
    for(int iNbr = 0; iNbr < vNbrIDs.size(); iNbr++) {
      int nbrSpcID = vNbrIDs[iNbr];
      const mcRPL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
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
      const mcRPL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
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
      const mcRPL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
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
      const mcRPL::SubCellspaceInfo *pNbrInfo = _findSubspcInfo(plSubspcInfos, nbrSpcID);
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

const mcRPL::SubCellspaceInfo* mcRPL::SubCellspaceInfo::
_findSubspcInfo(const list<mcRPL::SubCellspaceInfo> *plSubspcInfos,
                int subspcGlbID) const{
  const mcRPL::SubCellspaceInfo* pSubInfo = NULL;
  list<mcRPL::SubCellspaceInfo>::const_iterator itrSubInfo = plSubspcInfos->begin();
  while(itrSubInfo != plSubspcInfos->end()) {
    if(subspcGlbID == itrSubInfo->id()) {
      pSubInfo = &(*itrSubInfo);
      break;
    }
    itrSubInfo++;
  }
  return pSubInfo;
}

mcRPL::SubCellspaceInfo::
SubCellspaceInfo()
  :_id(mcRPL::ERROR_ID),
   _domDcmpType(mcRPL::NON_DCMP) {}

mcRPL::SubCellspaceInfo::
SubCellspaceInfo(int id,
                 mcRPL::DomDcmpType domDcmpType,
                 const mcRPL::SpaceDims &glbDims,
                 const mcRPL::CoordBR &MBR,
                 const mcRPL::CoordBR &workBR,
                 const vector<mcRPL::IntVect> &mNbrSpcIDs)
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

mcRPL::SubCellspaceInfo::
SubCellspaceInfo(const mcRPL::SubCellspaceInfo &rhs)
  :_id(rhs._id),
   _domDcmpType(rhs._domDcmpType),
   _glbDims(rhs._glbDims),
   _dims(rhs._dims),
   _MBR(rhs._MBR),
   _workBR(rhs._workBR),
   _mNbrSpcIDs(rhs._mNbrSpcIDs) {}

/*
mcRPL::SubCellspaceInfo::
SubCellspaceInfo(const mcRPL::IntVect &vInfoPack,
                 mcRPL::IntVctItr &iVal) {
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

int mcRPL::SubCellspaceInfo::
id() const {
  return _id;
}

mcRPL::DomDcmpType mcRPL::SubCellspaceInfo::
domDcmpType() const {
  return _domDcmpType;
}

long mcRPL::SubCellspaceInfo::
nGlbRows() const {
  return _glbDims.nRows();
}

long mcRPL::SubCellspaceInfo::
nGlbCols() const {
  return _glbDims.nCols();
}

const mcRPL::SpaceDims& mcRPL::SubCellspaceInfo::
glbDims() const {
  return _glbDims;
}

long mcRPL::SubCellspaceInfo::
nRows() const {
  return _dims.nRows();
}

long mcRPL::SubCellspaceInfo::
nCols() const {
  return _dims.nCols();
}

const mcRPL::SpaceDims& mcRPL::SubCellspaceInfo::
dims() const {
  return _dims;
}

long mcRPL::SubCellspaceInfo::
iRowBegin() const {
  return _MBR.minIRow();
}

long mcRPL::SubCellspaceInfo::
iColBegin() const {
  return _MBR.minICol();
}

long mcRPL::SubCellspaceInfo::
iRowEnd() const {
  return _MBR.maxIRow();
}

long mcRPL::SubCellspaceInfo::
iColEnd() const {
  return _MBR.maxICol();
}

const mcRPL::CoordBR& mcRPL::SubCellspaceInfo::
MBR() const {
  return _MBR;
}

long mcRPL::SubCellspaceInfo::
iRowWorkBegin() const {
  return _workBR.minIRow();
}

long mcRPL::SubCellspaceInfo::
iColWorkBegin() const {
  return _workBR.minICol();
}

long mcRPL::SubCellspaceInfo::
iRowWorkEnd() const {
  return _workBR.maxIRow();
}

long mcRPL::SubCellspaceInfo::
iColWorkEnd() const {
  return _workBR.maxICol();
}

const mcRPL::CoordBR& mcRPL::SubCellspaceInfo::
workBR() const {
  return _workBR;
}

mcRPL::SubCellspaceInfo& mcRPL::SubCellspaceInfo::
operator=(const mcRPL::SubCellspaceInfo &rhs) {
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

bool mcRPL::SubCellspaceInfo::
operator==(const mcRPL::SubCellspaceInfo &rhs) const {
  return (_id == rhs._id &&
          _domDcmpType == rhs._domDcmpType &&
          _glbDims == rhs._glbDims &&
          _dims == rhs._dims &&
          _MBR == rhs._MBR &&
          _workBR == rhs._workBR &&
          _mNbrSpcIDs == rhs._mNbrSpcIDs);
}

bool mcRPL::SubCellspaceInfo::
operator!=(const mcRPL::SubCellspaceInfo &rhs) const {
  return !(operator==(rhs));
}

double mcRPL::SubCellspaceInfo::
sizeRatio() const {
  return (double)(_dims.size()) / (double)(_glbDims.size());
}

bool mcRPL::SubCellspaceInfo::
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

bool mcRPL::SubCellspaceInfo::
validGlbCoord(const mcRPL::CellCoord &glbCoord,
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

bool mcRPL::SubCellspaceInfo::
validGlbCoord(long iRowGlb, long iColGlb,
              bool warning) const {
  return validGlbCoord(mcRPL::CellCoord(iRowGlb, iColGlb), warning);
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
glbIdx2glbCoord(long glbIdx) const {
  return mcRPL::CellCoord(glbIdx, _glbDims);
}

long mcRPL::SubCellspaceInfo::
glbCoord2glbIdx(const mcRPL::CellCoord &glbCoord) const {
  return glbCoord.toIdx(_glbDims);
}

long mcRPL::SubCellspaceInfo::
glbCoord2glbIdx(long iRowGlb, long iColGlb) const {
  return glbCoord2glbIdx(mcRPL::CellCoord(iRowGlb, iColGlb));
}

bool mcRPL::SubCellspaceInfo::
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

bool mcRPL::SubCellspaceInfo::
validCoord(const mcRPL::CellCoord &coord,
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

bool mcRPL::SubCellspaceInfo::
validCoord(long iRow, long iCol,
           bool warning) const {
  return validCoord(mcRPL::CellCoord(iRow, iCol), warning);
}

long mcRPL::SubCellspaceInfo::
coord2idx(const mcRPL::CellCoord &coord) const {
  return coord.toIdx(_dims);
}

long mcRPL::SubCellspaceInfo::
coord2idx(long iRow, long iCol) const {
  return coord2idx(mcRPL::CellCoord(iRow, iCol));
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
idx2coord(long idx) const {
  return mcRPL::CellCoord(idx, _dims);
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
lclCoord2glbCoord(const mcRPL::CellCoord& lclCoord) const {
  return (lclCoord + _MBR.nwCorner()); 
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
lclCoord2glbCoord(long iRowLcl, long iColLcl) const {
  return lclCoord2glbCoord(mcRPL::CellCoord(iRowLcl, iColLcl));
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
glbCoord2lclCoord(const mcRPL::CellCoord& glbCoord) const {
  return (glbCoord - _MBR.nwCorner());
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
glbCoord2lclCoord(long iRowGlb, long iColGlb) const {
  return glbCoord2lclCoord(mcRPL::CellCoord(iRowGlb, iColGlb));
}

long mcRPL::SubCellspaceInfo::
lclCoord2glbIdx(const mcRPL::CellCoord &lclCoord) const {
  return glbCoord2glbIdx(lclCoord2glbCoord(lclCoord));
}

long mcRPL::SubCellspaceInfo::
lclCoord2glbIdx(long iRowLcl, long iColLcl) const {
  return lclCoord2glbIdx(mcRPL::CellCoord(iRowLcl, iColLcl));
}

const mcRPL::CellCoord mcRPL::SubCellspaceInfo::
glbIdx2lclCoord(long glbIdx) const {
  return glbCoord2lclCoord(mcRPL::CellCoord(glbIdx, _glbDims));
}

long mcRPL::SubCellspaceInfo::
glbIdx2lclIdx(long glbIdx) const {
  return coord2idx(glbIdx2lclCoord(glbIdx));
}

long mcRPL::SubCellspaceInfo::
lclIdx2glbIdx(long lclIdx) const {
  return lclCoord2glbIdx(idx2coord(lclIdx));
}

int mcRPL::SubCellspaceInfo::
nNbrDirs() const {
  int nNbrDirects;
  switch(_domDcmpType) {
    case mcRPL::NON_DCMP:
      nNbrDirects = 0;
      break;
    case mcRPL::ROW_DCMP:
    case mcRPL::COL_DCMP:
      nNbrDirects = 2;
      break;
    case mcRPL::BLK_DCMP:
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

int mcRPL::SubCellspaceInfo::
nEdges() const {
  int numEdges;
  switch(_domDcmpType) {
    case mcRPL::NON_DCMP:
      numEdges = 0;
      break;
    case mcRPL::ROW_DCMP:
    case mcRPL::COL_DCMP:
      numEdges = 2;
      break;
    case mcRPL::BLK_DCMP:
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

int mcRPL::SubCellspaceInfo::
nTotNbrs() const {
  int numNbrs = 0;
  vector<mcRPL::IntVect>::const_iterator iNbrMap = _mNbrSpcIDs.begin();
  while(iNbrMap != _mNbrSpcIDs.end()) {
    numNbrs += (*iNbrMap).size();
    iNbrMap++;
  }
  return numNbrs;
}

bool mcRPL::SubCellspaceInfo::
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

bool mcRPL::SubCellspaceInfo::
hasNbrs(int iDir) const {
  bool hasIt = true;
  if(!validSpcDir(iDir) ||
     _mNbrSpcIDs[iDir].size() == 0) {
    hasIt = false;
  }
  return hasIt;
}

int mcRPL::SubCellspaceInfo::
nNbrs(int iDir) const {
  int numNbrs = 0;
  if(validSpcDir(iDir)) {
    numNbrs = _mNbrSpcIDs[iDir].size();
  }
  return numNbrs;
}

const mcRPL::IntVect& mcRPL::SubCellspaceInfo::
nbrSubSpcIDs(int iDir) const {
  return _mNbrSpcIDs.at(iDir);
}

int mcRPL::SubCellspaceInfo::
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

mcRPL::MeshDir mcRPL::SubCellspaceInfo::
spcDir2MeshDir(int iDir) const {
  MeshDir opDir = mcRPL::NONE_DIR;
  if(validSpcDir(iDir)) {
    switch(_domDcmpType) {
      case mcRPL::NON_DCMP:
        break;
      case mcRPL::ROW_DCMP:
        opDir = (iDir == mcRPL::UPPER_DIR) ? mcRPL::NORTH_DIR : mcRPL::SOUTH_DIR;
        break;
      case mcRPL::COL_DCMP:
        opDir = (iDir == mcRPL::LEFT_DIR) ? mcRPL::WEST_DIR : mcRPL::EAST_DIR;
        break;
      case mcRPL::BLK_DCMP:
        opDir = static_cast<mcRPL::MeshDir>(iDir);
        break;
      default:
        break;
    }
  }
  return opDir;
}

int mcRPL::SubCellspaceInfo::
oppositeDir(int iDir) const {
  int opDir = -1;
  if(validSpcDir(iDir)) {
    switch(_domDcmpType) {
      case mcRPL::NON_DCMP:
        break;
      case mcRPL::ROW_DCMP:
      case mcRPL::COL_DCMP:
        opDir = (iDir == 0) ? 1 : 0;
        break;
      case mcRPL::BLK_DCMP:
        opDir = static_cast<int>(mcRPL::oppositeDir(static_cast<mcRPL::MeshDir>(iDir)));
        break;
      default:
        break;
    }
  }
  return opDir;
}

bool mcRPL::SubCellspaceInfo::
calcBRs(bool onlyUpdtCtrCell,
        const mcRPL::Neighborhood *pNbrhd,
        const list<mcRPL::SubCellspaceInfo> *plSubspcInfos) {
  bool done = true;

  _initBRs();
  switch(_domDcmpType) {
    case mcRPL::NON_DCMP:
      break;
    case mcRPL::ROW_DCMP:
      done = _rowwiseBRs(onlyUpdtCtrCell, pNbrhd);
      break;
    case mcRPL::COL_DCMP:
      done = _colwiseBRs(onlyUpdtCtrCell, pNbrhd);
      break;
    case mcRPL::BLK_DCMP:
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

const mcRPL::CoordBR* mcRPL::SubCellspaceInfo::
interiorBR() const {
  const mcRPL::CoordBR *pIntrBR = NULL;
  if(_interiorBR.valid(_dims)) {
    pIntrBR = &_interiorBR;
  }
  return pIntrBR;
}

const mcRPL::CoordBR* mcRPL::SubCellspaceInfo::
edgeBR(int iDir) const {
  const mcRPL::CoordBR *pEdgeBR = NULL;
  if(_vEdgeBRs.size() == nEdges() &&
     iDir >= 0 && iDir < nEdges()) {
    pEdgeBR = _vEdgeBRs[iDir].valid(_dims)?&(_vEdgeBRs[iDir]):NULL;
  }
  return pEdgeBR;
}

const mcRPL::CoordBR* mcRPL::SubCellspaceInfo::
sendBR(int iDir, int iNbr) const {
  const mcRPL::CoordBR *pSendBR = NULL;
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
void mcRPL::SubCellspaceInfo::
add2IntVect(mcRPL::IntVect &vInfoPack) const {
  vInfoPack.push_back(_id);
  vInfoPack.push_back(_domDcmpType);
  vInfoPack.push_back(_glbDims.nRows());
  vInfoPack.push_back(_glbDims.nCols());
  vInfoPack.push_back(_MBR.nwCorner().toIdx(_glbDims));
  vInfoPack.push_back(_MBR.seCorner().toIdx(_glbDims));
  vInfoPack.push_back(_workBR.nwCorner().toIdx(_dims));
  vInfoPack.push_back(_workBR.seCorner().toIdx(_dims));

  for(int iDir = 0; iDir < nNbrDirs(); iDir++) {
    const mcRPL::IntVect &vNbrIDs = _mNbrSpcIDs[iDir];
    int nNbrs = vNbrIDs.size();
    vInfoPack.push_back(nNbrs);
    for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
      vInfoPack.push_back(vNbrIDs[iNbr]);
    }
  }
}

bool mcRPL::SubCellspaceInfo::
fromIntVect(const mcRPL::IntVect &vInfoPack,
            mcRPL::IntVctItr &iVal) {
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
  
  _domDcmpType = static_cast<mcRPL::DomDcmpType>(*iVal);
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

  _MBR.nwCorner(mcRPL::CellCoord(*iVal, _glbDims));
  iVal++;
  _MBR.seCorner(mcRPL::CellCoord(*iVal, _glbDims));
  iVal++;
  if(!_MBR.valid(_glbDims)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: SubCellspace[" << _id \
         << "] invalid MBR (" << _MBR \
         << ")" << endl;
    return false;
  }
  _dims = mcRPL::SpaceDims(_MBR.nRows(), _MBR.nCols());

  _workBR.nwCorner(mcRPL::CellCoord(*iVal, _dims));
  iVal++;
  _workBR.seCorner(mcRPL::CellCoord(*iVal, _dims));
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
