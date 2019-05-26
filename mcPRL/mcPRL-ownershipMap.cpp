#include "mcPRL-ownershipMap.h"

mcPRL::OwnershipMap::
OwnershipMap() {
  _itrSubspc2Map = _vSubspcIDs.begin();
}

void mcPRL::OwnershipMap::
init(const mcPRL::IntVect &vSubspcIDs,
     const mcPRL::IntVect &vPrcIDs) {
  _vSubspcIDs = vSubspcIDs;
  _itrSubspc2Map = _vSubspcIDs.begin();

  _mPrcSubspcs.clear();
  for(int iPrc = 0; iPrc < vPrcIDs.size(); iPrc++) {
    _mPrcSubspcs.insert(make_pair(vPrcIDs[iPrc], IntVect()));
  }
}

void mcPRL::OwnershipMap::
clearMapping() {
  map<int, mcPRL::IntVect>::iterator itrMap = _mPrcSubspcs.begin();
  while(itrMap != _mPrcSubspcs.end()) {
    itrMap->second.clear();
    itrMap++;
  }
  _itrSubspc2Map = _vSubspcIDs.begin();
}

void mcPRL::OwnershipMap::
mapping(mcPRL::MappingMethod mapMethod,
        int nSubspcs2Map) {
  clearMapping();

  int nActualSubspcs2Map = (nSubspcs2Map > 0)?nSubspcs2Map:_vSubspcIDs.size();
  map<int, mcPRL::IntVect>::iterator itrMap = _mPrcSubspcs.begin();

  if(mapMethod == mcPRL::CYLC_MAP) {
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
  else if(mapMethod == mcPRL::BKFR_MAP) {
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

int mcPRL::OwnershipMap::
mappingNextTo(int prcID) {
  int mappedSubspcID = mcPRL::ERROR_ID;

  map<int, mcPRL::IntVect>::iterator itrMap = _mPrcSubspcs.find(prcID);
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

bool mcPRL::OwnershipMap::
mappingTo(int subspcID,
          int prcID) {
  if(std::find(_vSubspcIDs.begin(), _vSubspcIDs.end(), subspcID) == _vSubspcIDs.end()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid SubCellspace ID (" << subspcID \
         << ")" << endl;
    return false;
  }
  map<int, mcPRL::IntVect>::iterator itrMap = _mPrcSubspcs.find(prcID);
  if(itrMap == _mPrcSubspcs.end()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid process ID (" \
         << prcID << ")" << endl;
    return false;
  }
  itrMap->second.push_back(subspcID);

  return true;
}

bool mcPRL::OwnershipMap::
allMapped() const {
  return (_itrSubspc2Map == _vSubspcIDs.end());
}

const mcPRL::IntVect* mcPRL::OwnershipMap::
subspcIDsOnPrc(int prcID) const {
  const mcPRL::IntVect *pvSubspcIDs = NULL;
  map<int, mcPRL::IntVect>::const_iterator itrMap = _mPrcSubspcs.find(prcID);
  if(itrMap != _mPrcSubspcs.end()) {
    pvSubspcIDs = &(itrMap->second);
  }
  return pvSubspcIDs;
}

mcPRL::IntVect mcPRL::OwnershipMap::
mappedSubspcIDs() const {
  mcPRL::IntVect vMappedIDs;
  mcPRL::IntVect::const_iterator itrMap = _vSubspcIDs.begin();
  while(itrMap != _itrSubspc2Map) {
    vMappedIDs.push_back(*itrMap);
    itrMap++;
  }
  return vMappedIDs;
}

int mcPRL::OwnershipMap::
findPrcBySubspcID(int subspcID) const {
  int prcID = mcPRL::ERROR_ID;
  map<int, mcPRL::IntVect>::const_iterator itrPrc = _mPrcSubspcs.begin();
  while(itrPrc != _mPrcSubspcs.end()) {
    const mcPRL::IntVect &vSubspcs = itrPrc->second;
    if(std::find(vSubspcs.begin(), vSubspcs.end(), subspcID) != vSubspcs.end()) {
      prcID = itrPrc->first;
      break;
    }
    itrPrc++;
  }

  return prcID;
}

bool mcPRL::OwnershipMap::
findSubspcIDOnPrc(int subspcID,
                  int prcID) const {
  bool found = false;
  const mcPRL::IntVect *pvSubspcIDs = subspcIDsOnPrc(prcID);
  if(pvSubspcIDs != NULL) {
    if(std::find(pvSubspcIDs->begin(), pvSubspcIDs->end(), subspcID) != pvSubspcIDs->end()) {
      found = true;
    }
  }
  return found;
}

bool mcPRL::OwnershipMap::
map2buf(vector<char> &buf) const {
  buf.clear();

  int bufPtr = 0;
  map<int, mcPRL::IntVect>::const_iterator itrMap = _mPrcSubspcs.begin();
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

bool mcPRL::OwnershipMap::
buf2map(const vector<char> &buf) {
  _mPrcSubspcs.clear();

  int bufPtr = 0;
  int iPrc = mcPRL::ERROR_ID;
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

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::OwnershipMap &mOwnerships) {
  map<int, mcPRL::IntVect>::const_iterator itrMap = mOwnerships._mPrcSubspcs.begin();
  while(itrMap != mOwnerships._mPrcSubspcs.end()) {
    cout << "Process [" << itrMap->first << "] gets " \
         << itrMap->second.size() << " SubCellspaces: ";
    mcPRL::IntVect::const_iterator itrSubID = itrMap->second.begin();
    while(itrSubID != itrMap->second.end()) {
      cout << *itrSubID << " ";
      itrSubID++;
    }
    cout << endl;
    itrMap++;
  }
  return os;
}
