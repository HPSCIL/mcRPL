#include "mcrpl-transferInfo.h"

mcRPL::TransferInfo::
TransferInfo()
  :_fromPrcID(mcRPL::ERROR_ID),
   _toPrcID(mcRPL::ERROR_ID),
   _subspcGlbID(mcRPL::ERROR_ID),
   _tDataType(mcRPL::SPACE_TRANSFDATA),
   _cmplt(false) {}

mcRPL::TransferInfo::
TransferInfo(int fromPrcID,
             int toPrcID,
             const string &lyrName,
             int subspcGlbID,
             mcRPL::TransferDataType transfType)
  :_fromPrcID(fromPrcID),
   _toPrcID(toPrcID),
   _lyrName(lyrName),
   _subspcGlbID(subspcGlbID),
   _tDataType(transfType),
   _cmplt(false) {}

int mcRPL::TransferInfo::
fromPrcID() const {
  return _fromPrcID;
}

int mcRPL::TransferInfo::
toPrcID() const {
  return _toPrcID;
}

const string& mcRPL::TransferInfo::
lyrName() const {
  return _lyrName;
}

int mcRPL::TransferInfo::
subspcGlbID() const {
  return _subspcGlbID;
}

mcRPL::TransferDataType mcRPL::TransferInfo::
transfDataType() const {
  return _tDataType;
}

bool mcRPL::TransferInfo::
completed() const {
  return _cmplt;
}

void mcRPL::TransferInfo::
complete() {
  _cmplt = true;
}

void mcRPL::TransferInfo::
restart() {
  _cmplt = false;
}

const mcRPL::TransferInfo* mcRPL::TransferInfoVect::
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

bool mcRPL::TransferInfoVect::
checkBcastCellspace(int prcID,
                    const string &lyrName) const {
  bool cmplt = true;
  bool found = false;
  for(int iInfo = 0; iInfo < size(); iInfo++) {
    if(at(iInfo).fromPrcID() == prcID &&
       at(iInfo).lyrName() == lyrName &&
       at(iInfo).transfDataType() == mcRPL::SPACE_TRANSFDATA &&
       at(iInfo).subspcGlbID() == mcRPL::ERROR_ID) {
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
