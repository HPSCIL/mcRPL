#include "mcPRL-transferInfo.h"

mcPRL::TransferInfo::
TransferInfo()
  :_fromPrcID(mcPRL::ERROR_ID),
   _toPrcID(mcPRL::ERROR_ID),
   _subspcGlbID(mcPRL::ERROR_ID),
   _tDataType(mcPRL::SPACE_TRANSFDATA),
   _cmplt(false) {}

mcPRL::TransferInfo::
TransferInfo(int fromPrcID,
             int toPrcID,
             const string &lyrName,
             int subspcGlbID,
             mcPRL::TransferDataType transfType)
  :_fromPrcID(fromPrcID),
   _toPrcID(toPrcID),
   _lyrName(lyrName),
   _subspcGlbID(subspcGlbID),
   _tDataType(transfType),
   _cmplt(false) {}

int mcPRL::TransferInfo::
fromPrcID() const {
  return _fromPrcID;
}

int mcPRL::TransferInfo::
toPrcID() const {
  return _toPrcID;
}

const string& mcPRL::TransferInfo::
lyrName() const {
  return _lyrName;
}

int mcPRL::TransferInfo::
subspcGlbID() const {
  return _subspcGlbID;
}

mcPRL::TransferDataType mcPRL::TransferInfo::
transfDataType() const {
  return _tDataType;
}

bool mcPRL::TransferInfo::
completed() const {
  return _cmplt;
}

void mcPRL::TransferInfo::
complete() {
  _cmplt = true;
}

void mcPRL::TransferInfo::
restart() {
  _cmplt = false;
}

const mcPRL::TransferInfo* mcPRL::TransferInfoVect::
findInfo(const string &lyrName,
         int subspcGlbID) const {
  const TransferInfo *pInfo = NULL;
  TransferInfoVect::const_iterator itrInfo = begin();
  while(itrInfo != end()) {
    if(itrInfo->lyrName() == lyrName &&
       itrInfo->subspcGlbID() == subspcGlbID) {
      pInfo = &(*itrInfo);
      break;
    }
    itrInfo++;
  }
  return pInfo;
}

bool mcPRL::TransferInfoVect::
checkBcastCellspace(int prcID,
                    const string &lyrName) const {
  bool cmplt = true;
  bool found = false;
  for(int iInfo = 0; iInfo < size(); iInfo++) {
    if(at(iInfo).fromPrcID() == prcID &&
       at(iInfo).lyrName() == lyrName &&
       at(iInfo).transfDataType() == mcPRL::SPACE_TRANSFDATA &&
       at(iInfo).subspcGlbID() == mcPRL::ERROR_ID) {
      found = true;
      if(at(iInfo).completed() == false) {
        cmplt = false;
        break;
      }
    }
  }
  if(found == false) {
    cmplt = false;
  }

  return cmplt;
}
