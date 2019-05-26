#include "mcPRL-subCellspace.h"

mcPRL::SubCellspace::
SubCellspace()
  :_pSubInfo(NULL),
   mcPRL::Cellspace() {}

mcPRL::SubCellspace::
SubCellspace(mcPRL::CellspaceInfo *pInfo,
             mcPRL::SubCellspaceInfo *pSubInfo)
  :_pSubInfo(pSubInfo),
   mcPRL::Cellspace(pInfo) {}

mcPRL::SubCellspace::
SubCellspace(const mcPRL::SubCellspace &rhs)
  :_pSubInfo(rhs._pSubInfo),
   mcPRL::Cellspace(rhs) {}

mcPRL::SubCellspace::
~SubCellspace() {
  _pSubInfo = NULL;
}

mcPRL::SubCellspace& mcPRL::SubCellspace::
operator=(const mcPRL::SubCellspace &rhs) {
  if(this != &rhs) {
    _pSubInfo = rhs._pSubInfo;
    Cellspace::operator=(rhs);
  }
  return *this;
}

mcPRL::SubCellspaceInfo* mcPRL::SubCellspace::
subInfo() {
  return _pSubInfo;
}

const mcPRL::SubCellspaceInfo* mcPRL::SubCellspace::
subInfo() const {
  return _pSubInfo;
}

/*
bool mcPRL::SubCellspace::
add2UpdtStream(mcPRL::CellStream &updtStream,
               int iDir,
               int iNbr) const {
  if(iDir >= 0 && iDir < _pSubInfo->nNbrDirs() &&
     iNbr >= 0 && iNbr < _pSubInfo->nNbrs(iDir)) {
    if(updtStream.valSize() <= 0) {
      updtStream.init(_pInfo->dataSize());
    }
    else if(updtStream.valSize() != _pInfo->dataSize()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: Cellspace's data size (" << _pInfo->dataSize() \
           << ") is NOT the same as the CellStream's data size (" \
           << updtStream.valSize() << ")" << endl;
      return false;
    }

    for(int iIdx = 0; iIdx < _vUpdtIdxs.size(); iIdx++) {
      long lclIdx = _vUpdtIdxs[iIdx];
      const mcPRL::CoordBR *pSendBR = _pSubInfo->sendBR(iDir, iNbr);
      if(pSendBR != NULL && pSendBR->ifContain(_pInfo->idx2coord(lclIdx))) {
        long glbIdx = _pSubInfo->lclIdx2glbIdx(lclIdx);
        void *aVal = at(lclIdx);
        if(!updtStream.addCell(glbIdx, aVal)) {
          return false;
        }
      }
    } // end -- for(iIdx)
  }
  return true;
}

bool mcPRL::SubCellspace::
loadUpdtStream(const mcPRL::CellStream &updtStream) {
  if(updtStream.valSize() != _pInfo->dataSize()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: Cellspace's data size (" << _pInfo->dataSize() \
         << ") is NOT the same as the CellStream's data size (" \
         << updtStream.valSize() << ")" << endl;
    return false;
  }

  long glbIdx, lclIdx;
  for(long iCell = 0; iCell < updtStream.nCells(); iCell++) {
    const void *aCell = updtStream.at(iCell);
    memcpy(&glbIdx, aCell, sizeof(long));
    if(_pSubInfo->MBR().ifContain(_pSubInfo->glbIdx2glbCoord(glbIdx))) {
      lclIdx = _pSubInfo->glbIdx2lclIdx(glbIdx);
      memcpy(at(lclIdx), aCell+sizeof(long), updtStream.valSize());
    }
  }

  return true;
}
*/
