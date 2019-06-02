#include "mcrpl-ownershipMap.h"

mcRPL::OwnershipMap::
OwnershipMap() {
  _itrSubspc2Map = _vSubspcIDs.begin();
}

void mcRPL::OwnershipMap::
init(const mcRPL::IntVect &vSubspcIDs,
     const mcRPL::IntVect &vPrcIDs) {
  _vSubspcIDs = vSubspcIDs;
  _itrSubspc2Map = _vSubspcIDs.begin();

  _mPrcSubspcs.clear();
  for(int iPrc = 0; iPrc < vPrcIDs.size(); iPrc++) {
    _mPrcSubspcs.insert(make_pair(vPrcIDs[iPrc], IntVect()));
  }
}

void mcRPL::OwnershipMap::
clearMapping() {
  map<int, mcRPL::IntVect>::iterator itrMap = _mPrcSubspcs.begin();
  while(itrMap != _mPrcSubspcs.end()) {
    itrMap->second.clear();
    itrMap++;
  }
  _itrSubspc2Map = _vSubspcIDs.begin();
}

void mcRPL::OwnershipMap::
mapping(mcRPL::MappingMethod mapMethod,
        int nSubspcs2Map) {
  clearMapping();

  int nActualSubspcs2Map = (nSubspcs2Map > 0)?nSubspcs2Map:_vSubspcIDs.size();
  map<int, mcRPL::IntVect>::iterator itrMap = _mPrcSubspcs.begin();

  if(mapMethod == mcRPL::CYLC_MAP) {
    for(int iSubspc = 0; iSubspc < nActualSubspcs2Map; iSubspc++) {
      if(_itrSubspc2Map != _vSubspcIDs.end()) {
        itrMap->second.push_back(*_itrSubspc2Map);
          _itrSubspc2Map++;

        itrMap++;
        if(itrMap == _mPrcSubspcs.end()) {
          itrMap = _mPrcSubspcs.begin();
        }
      }
      else {
        break;
      }
    } // end of for(iSubspc) loop
  }
  else if(mapMethod == mcRPL::BKFR_MAP) {
    bool goForward = true;
    for(int iSubspc = 0; iSubspc < nActualSubspcs2Map; iSubspc++) {
      if(_itrSubspc2Map != _vSubspcIDs.end()) {
        itrMap->second.push_back(*_itrSubspc2Map);
        _itrSubspc2Map++;

        if(goForward) {
          itrMap++;
          if(itrMap == _mPrcSubspcs.end()) {
            itrMap--;
            goForward = false;
          }
        }
        else {
          itrMap--;
          if(itrMap == _mPrcSubspcs.begin()) {
            goForward = true;
          }
        }
      }
      else {
        break;
      }
    } // end of for(iSubspc) loop
  }
}

int mcRPL::OwnershipMap::
mappingNextTo(int prcID) {
  int mappedSubspcID = mcRPL::ERROR_ID;

  map<int, mcRPL::IntVect>::iterator itrMap = _mPrcSubspcs.find(prcID);
  if(itrMap == _mPrcSubspcs.end()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid process ID (" \
         << prcID << ")" << endl;
  }
  else if(_itrSubspc2Map != _vSubspcIDs.end()) {
    mappedSubspcID = *_itrSubspc2Map;
    itrMap->second.push_back(mappedSubspcID);
    _itrSubspc2Map++;
  }
  else {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: all SubCellspaces have been mapped" \
         << endl;
  }

  return mappedSubspcID;
}

bool mcRPL::OwnershipMap::
mappingTo(int subspcID,
          int prcID) {
  if(std::find(_vSubspcIDs.begin(), _vSubspcIDs.end(), subspcID) == _vSubspcIDs.end()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid SubCellspace ID (" << subspcID \
         << ")" << endl;
    return false;
  }
  map<int, mcRPL::IntVect>::iterator itrMap = _mPrcSubspcs.find(prcID);
  if(itrMap == _mPrcSubspcs.end()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid process ID (" \
         << prcID << ")" << endl;
    return false;
  }
  itrMap->second.push_back(subspcID);

  return true;
}

bool mcRPL::OwnershipMap::
allMapped() const {
  return (_itrSubspc2Map == _vSubspcIDs.end());
}

const mcRPL::IntVect* mcRPL::OwnershipMap::
subspcIDsOnPrc(int prcID) const {
  const mcRPL::IntVect *pvSubspcIDs = NULL;
  map<int, mcRPL::IntVect>::const_iterator itrMap = _mPrcSubspcs.find(prcID);
  if(itrMap != _mPrcSubspcs.end()) {
    pvSubspcIDs = &(itrMap->second);
  }
  return pvSubspcIDs;
}

mcRPL::IntVect mcRPL::OwnershipMap::
mappedSubspcIDs() const {
  mcRPL::IntVect vMappedIDs;
  mcRPL::IntVect::const_iterator itrMap = _vSubspcIDs.begin();
  while(itrMap != _itrSubspc2Map) {
    vMappedIDs.push_back(*itrMap);
    itrMap++;
  }
  return vMappedIDs;
}

int mcRPL::OwnershipMap::
findPrcBySubspcID(int subspcID) const {
  int prcID = mcRPL::ERROR_ID;
  map<int, mcRPL::IntVect>::const_iterator itrPrc = _mPrcSubspcs.begin();
  while(itrPrc != _mPrcSubspcs.end()) {
    const mcRPL::IntVect &vSubspcs = itrPrc->second;
    if(std::find(vSubspcs.begin(), vSubspcs.end(), subspcID) != vSubspcs.end()) {
      prcID = itrPrc->first;
      break;
    }
    itrPrc++;
  }

  return prcID;
}

bool mcRPL::OwnershipMap::
findSubspcIDOnPrc(int subspcID,
                  int prcID) const {
  bool found = false;
  const mcRPL::IntVect *pvSubspcIDs = subspcIDsOnPrc(prcID);
  if(pvSubspcIDs != NULL) {
    if(std::find(pvSubspcIDs->begin(), pvSubspcIDs->end(), subspcID) != pvSubspcIDs->end()) {
      found = true;
    }
  }
  return found;
}

bool mcRPL::OwnershipMap::
map2buf(vector<char> &buf) const {
  buf.clear();

  int bufPtr = 0;
  map<int, mcRPL::IntVect>::const_iterator itrMap = _mPrcSubspcs.begin();
  while(itrMap != _mPrcSubspcs.end()) {
    int iPrc = itrMap->first;
    const vector<int> &vSubspcIDs = itrMap->second;
    int nSubspcs = vSubspcIDs.size();
    bufPtr = buf.size();
    buf.resize(buf.size() + (nSubspcs + 2) * sizeof(int));
    memcpy((void *)&(buf[bufPtr]), &iPrc, sizeof(int));
    bufPtr += sizeof(int);
    memcpy((void *)&(buf[bufPtr]), &nSubspcs, sizeof(int));
    bufPtr += sizeof(int);
    /*
    for(int iID = 0; iID < nSubspcs; iID++) {
      memcpy((void *)&(buf[bufPtr]), &(vSubspcIDs[iID]), sizeof(int));
      bufPtr += sizeof(int);
    }
    */
    memcpy((void *)&(buf[bufPtr]), &(vSubspcIDs[0]), nSubspcs*sizeof(int));
    bufPtr += nSubspcs * sizeof(int);
    itrMap++;
  }

  return true;
}

bool mcRPL::OwnershipMap::
buf2map(const vector<char> &buf) {
  _mPrcSubspcs.clear();

  int bufPtr = 0;
  int iPrc = mcRPL::ERROR_ID;
  int nSubspcs = 0;
  while(bufPtr < buf.size()) {
    memcpy(&iPrc, (void *)&(buf[bufPtr]), sizeof(int));
    bufPtr += sizeof(int);
    memcpy(&nSubspcs, (void *)&(buf[bufPtr]), sizeof(int));
    bufPtr += sizeof(int);
    _mPrcSubspcs[iPrc].resize(nSubspcs);
    memcpy(&(_mPrcSubspcs[iPrc][0]), (void *)&(buf[bufPtr]), nSubspcs*sizeof(int));
    bufPtr += nSubspcs * sizeof(int);
    /*
    for(int iID = 0; iID < nSubspcs; iID++) {
      memcpy(&(_mPrcSubspcs[iPrc][iID]), (void *)&(buf[bufPtr]), sizeof(int));
      bufPtr += sizeof(int);
    }
    */
  }

  return true;
}

ostream& mcRPL::
operator<<(ostream &os, const mcRPL::OwnershipMap &mOwnerships) {
  map<int, mcRPL::IntVect>::const_iterator itrMap = mOwnerships._mPrcSubspcs.begin();
  while(itrMap != mOwnerships._mPrcSubspcs.end()) {
    cout << "Process [" << itrMap->first << "] gets " \
         << itrMap->second.size() << " SubCellspaces: ";
    mcRPL::IntVect::const_iterator itrSubID = itrMap->second.begin();
    while(itrSubID != itrMap->second.end()) {
      cout << *itrSubID << " ";
      itrSubID++;
    }
    cout << endl;
    itrMap++;
  }
  return os;
}
