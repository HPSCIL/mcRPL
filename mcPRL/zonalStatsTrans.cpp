/*
 * zonalStatsTrans.cpp
 *
 *  Created on: Mar 18, 2014
 *      Author: sonicben
 */

#include "zonalStatsTrans.h"

bool ZonalStatsTransition::
_checkZoneRange() const {
  if(_minZoneID > _maxZoneID) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: invalid minimal and maxmal zone IDs (" \
        << _minZoneID << ", " << _maxZoneID << ")" << endl;
    return false;
  }
  return true;
}

ZonalStatsTransition::
ZonalStatsTransition()
  :pRPL::Transition(),
   _pValCellspc(NULL),
   _pZoneCellspc(NULL),
   _valNoData(pRPL::DEFAULT_NODATA_USHORT),
   _zoneNoData(pRPL::DEFAULT_NODATA_USHORT),
   _minZoneID(pRPL::DEFAULT_NODATA_USHORT),
   _maxZoneID(0) {
  _needExchange = false;
  _edgesFirst = false;
}

ZonalStatsTransition::
~ZonalStatsTransition() {}

unsigned short ZonalStatsTransition::
getMinZoneID() const {
  return _minZoneID;
}

void ZonalStatsTransition::
setMinZoneID(unsigned short minZoneID) {
  _minZoneID = minZoneID;
}

unsigned short ZonalStatsTransition::
getMaxZoneID() const {
  return _maxZoneID;
}

void ZonalStatsTransition::
setMaxZoneID(unsigned short maxZoneID) {
  _maxZoneID = maxZoneID;
}

bool ZonalStatsTransition::
minValVector(vector<unsigned short> &vMinVals) const {
  vMinVals.clear();

  if(!_checkZoneRange()) {
    return false;
  }

  for(int zoneID = _minZoneID; zoneID <= _maxZoneID; zoneID++) {
    map<unsigned short, unsigned short>::const_iterator itrMin = _mMins.find(zoneID);
    if(itrMin != _mMins.end()) {
      vMinVals.push_back(itrMin->second);
    }
    else {
      vMinVals.push_back(9999);
    }
  }

  return true;
}

bool ZonalStatsTransition::
maxValVector(vector<unsigned short> &vMaxVals) const {
  vMaxVals.clear();

  if(!_checkZoneRange()) {
    return false;
  }

  for(int zoneID = _minZoneID; zoneID <= _maxZoneID; zoneID++) {
    map<unsigned short, unsigned short>::const_iterator itrMax = _mMaxs.find(zoneID);
    if(itrMax != _mMaxs.end()) {
      vMaxVals.push_back(itrMax->second);
    }
    else {
      vMaxVals.push_back(0);
    }
  }

  return true;
}

bool ZonalStatsTransition::
countVector(vector<long> &vCounts) const {
  vCounts.clear();

  if(!_checkZoneRange()) {
    return false;
  }

  for(int zoneID = _minZoneID; zoneID <= _maxZoneID; zoneID++) {
    map<unsigned short, long>::const_iterator itrCount = _mCounts.find(zoneID);
    if(itrCount != _mCounts.end()) {
      vCounts.push_back(itrCount->second);
    }
    else {
      vCounts.push_back(0);
    }
  }

  return true;
}

bool ZonalStatsTransition::
totalValVector(vector<long> &vTotVals) const {
  vTotVals.clear();

  if(!_checkZoneRange()) {
    return false;
  }

  for(int zoneID = _minZoneID; zoneID <= _maxZoneID; zoneID++) {
    map<unsigned short, long>::const_iterator itrTotal = _mTotals.find(zoneID);
    if(itrTotal != _mTotals.end()) {
      vTotVals.push_back(itrTotal->second);
    }
    else {
      vTotVals.push_back(0);
    }
  }

  return true;
}

bool ZonalStatsTransition::
calcStats(vector<unsigned short> &vMinVals,
          vector<unsigned short> &vMaxVals,
          vector<long> &vTotVals,
          vector<long> &vCounts) {
  _mMeans.clear();
  _mMins.clear();
  _mMaxs.clear();

  if(vTotVals.size() != _maxZoneID - _minZoneID + 1 ||
     vCounts.size() != _maxZoneID - _minZoneID + 1 ||
     vMinVals.size() != _maxZoneID - _minZoneID + 1 ||
     vMaxVals.size() != _maxZoneID - _minZoneID + 1) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid size (" << vMinVals.size() << ", "  << vMaxVals.size() << ", " \
         << vTotVals.size() << "," << vCounts.size() \
         << "), given the zone ID range [" << _minZoneID << ", " << _maxZoneID << "]" << endl;
    return false;
  }

  int iZone = 0;
  for(int zone = _minZoneID; zone <= _maxZoneID; zone++) {
    _mMeans[zone] = (double)vTotVals[iZone] / (double)vCounts[iZone];
    _mMins[zone] = vMinVals[iZone];
    _mMaxs[zone] = vMaxVals[iZone];
    iZone++;
  }
  return true;
}

const map<unsigned short, double>& ZonalStatsTransition::
meanMap() const {
  return _mMeans;
}

const map<unsigned short, unsigned short>& ZonalStatsTransition::
minMap() const {
  return _mMins;
}

const map<unsigned short, unsigned short>& ZonalStatsTransition::
maxMap() const {
  return _mMaxs;
}

bool ZonalStatsTransition::
check() const {
  if(_mpCellspcs.size() != 2) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: TWO Cellspaces are needed" \
        << endl;
    return false;
  }

  if(_vInLyrNames.size() != 2) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: TWO input Layers are needed" \
         << endl;
    return false;
  }

  if(_vOutLyrNames.size() != 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: NO output Layer should be specified" \
         << endl;
    return false;
  }

  const pRPL::Cellspace *pPrmCellspc = getCellspaceByLyrName(_primeLyrName);
  if(pPrmCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: cannot find the primary Cellspace" \
        << endl;
    return false;
  }
  const pRPL::SpaceDims &myDims = pPrmCellspc->info()->dims();

  map<string, pRPL::Cellspace *>::const_iterator itrCellspcMap = _mpCellspcs.begin();
  while(itrCellspcMap != _mpCellspcs.end()) {
    const pRPL::Cellspace *pCellspc = itrCellspcMap->second;
    if(pCellspc == NULL) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL pointer to Cellspace (" \
          << itrCellspcMap->first << ")" \
          << endl;
      return false;
    }
    if(pCellspc->info()->dims() != myDims) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: Cellspace(" << itrCellspcMap->first \
          << ")'s dimensions (" << pCellspc->info()->dims() \
          << ") do NOT match with the primary Cellspace's dimensions (" \
          << myDims << ")" \
          << endl;
      return false;
    }
    itrCellspcMap++;
  } // end -- while(itrCellspcMap != _mpCellspcs.end())

  return true;
}

bool ZonalStatsTransition::
afterSetCellspaces(int subCellspcGlbIdx) {
  _pValCellspc = getCellspaceByLyrName(_vInLyrNames[0]);
  if(_pValCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NULL pointer to input value Cellspace (" \
        << _vInLyrNames[0] << ")" \
        << endl;
    return false;
  }
  _valNoData = _pValCellspc->info()->getNoDataValAs<unsigned short>();


  _pZoneCellspc = getCellspaceByLyrName(_vInLyrNames[1]);
  if(_pZoneCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NULL pointer to input zone Cellspace (" \
        << _vOutLyrNames[0] << ")" \
        << endl;
    return false;
  }
  _zoneNoData = _pZoneCellspc->info()->getNoDataValAs<unsigned short>();

  return true;
}

pRPL::EvaluateReturn ZonalStatsTransition::
evaluate(const pRPL::CellCoord &coord) {
  unsigned short val = _pValCellspc->atAs<unsigned short>(coord, true);
  unsigned short zone = _pZoneCellspc->atAs<unsigned short>(coord, true);

  if(val != _valNoData &&
     zone != _zoneNoData &&
     val <= 8848) {
    if(_mCounts.empty()) {
      _minZoneID = zone;
      _maxZoneID = zone;
    }
    else {
      if(_minZoneID > zone) {
        _minZoneID = zone;
      }
      if(_maxZoneID < zone) {
        _maxZoneID = zone;
      }
    }

    if(_mMins.find(zone) == _mMins.end()) {
      _mMins[zone] = val;
    }
    else if(_mMins[zone] > val) {
      _mMins[zone] = val;
    }

    if(_mMaxs.find(zone) == _mMaxs.end()) {
      _mMaxs[zone] = val;
    }
    else if(_mMaxs[zone] < val) {
      _mMaxs[zone] = val;
    }

    if(_mCounts.find(zone) == _mCounts.end()) {
      _mCounts[zone] = 1;
    }
    else {
      _mCounts[zone]++;
    }

    if(_mTotals.find(zone) == _mTotals.end()) {
      _mTotals[zone] = (long)val;
    }
    else {
      _mTotals[zone] += (long)val;
    }
  }

  return pRPL::EVAL_SUCCEEDED;
}

ZonalOutputTransition::
ZonalOutputTransition()
  :_pZoneCellspc(NULL),
   _pMeanCellspc(NULL),
   _pMinCellspc(NULL),
   _pMaxCellspc(NULL),
   _zoneNoData(pRPL::DEFAULT_NODATA_USHORT),
   _meanNoData(pRPL::DEFAULT_NODATA_FLOAT),
   _minNoData(pRPL::DEFAULT_NODATA_USHORT),
   _maxNoData(pRPL::DEFAULT_NODATA_USHORT),
   _pmMeans(NULL),
   _pmMins(NULL),
   _pmMaxs(NULL) {
  _needExchange = false;
}

ZonalOutputTransition::
~ZonalOutputTransition() {}

void ZonalOutputTransition::
setMeanMap(const map<unsigned short, double> *pmMeans) {
  _pmMeans = pmMeans;
}

void ZonalOutputTransition::
setMinMap(const map<unsigned short, unsigned short> *pmMins) {
  _pmMins = pmMins;
}

void ZonalOutputTransition::
setMaxMap(const map<unsigned short, unsigned short> *pmMaxs) {
  _pmMaxs = pmMaxs;
}

bool ZonalOutputTransition::
check() const {
  if(_mpCellspcs.size() != 4) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: FOUR Cellspaces are needed" \
        << endl;
    return false;
  }

  if(_vInLyrNames.size() != 1) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: ONE input Layer should be specified" \
        << endl;
    return false;
  }

  if(_vOutLyrNames.size() != 3) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: THREE output Layers should be specified" \
        << endl;
    return false;
  }

  const pRPL::Cellspace *pPrmCellspc = getCellspaceByLyrName(_primeLyrName);
  if(pPrmCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: cannot find the primary Cellspace" \
        << endl;
    return false;
  }
  const pRPL::SpaceDims &myDims = pPrmCellspc->info()->dims();

  map<string, pRPL::Cellspace *>::const_iterator itrCellspcMap = _mpCellspcs.begin();
  while(itrCellspcMap != _mpCellspcs.end()) {
    const pRPL::Cellspace *pCellspc = itrCellspcMap->second;
    if(pCellspc == NULL) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL pointer to Cellspace (" \
          << itrCellspcMap->first << ")" \
          << endl;
      return false;
    }
    if(pCellspc->info()->dims() != myDims) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: Cellspace(" << itrCellspcMap->first \
          << ")'s dimensions (" << pCellspc->info()->dims() \
          << ") do NOT match with the primary Cellspace's dimensions (" \
          << myDims << ")" \
          << endl;
      return false;
    }
    itrCellspcMap++;
  } // end -- while(itrCellspcMap != _mpCellspcs.end())

  if(_pmMeans == NULL ||
     _pmMins == NULL ||
     _pmMaxs == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error：the maps of means, minimal, and maximal have NOT been specified yet" \
         << endl;
    return false;
  }

  return true;
}

bool ZonalOutputTransition::
afterSetCellspaces(int subCellspcGlbIdx) {
  _pZoneCellspc = getCellspaceByLyrName(_vInLyrNames[0]);
  if(_pZoneCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NULL pointer to input zone Cellspace (" \
        << _vInLyrNames[0] << ")" \
        << endl;
    return false;
  }
  _zoneNoData = _pZoneCellspc->info()->getNoDataValAs<unsigned short>();

  _pMeanCellspc = getCellspaceByLyrName(_vOutLyrNames[0]);
  if(_pMeanCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NULL pointer to output mean Cellspace (" \
        << _vOutLyrNames[0] << ")" \
        << endl;
    return false;
  }
  _meanNoData = _pMeanCellspc->info()->getNoDataValAs<float>();

  _pMinCellspc = getCellspaceByLyrName(_vOutLyrNames[1]);
  if(_pMinCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NULL pointer to output min Cellspace (" \
        << _vOutLyrNames[1] << ")" \
        << endl;
    return false;
  }
  _minNoData = _pMinCellspc->info()->getNoDataValAs<unsigned short>();

  _pMaxCellspc = getCellspaceByLyrName(_vOutLyrNames[2]);
  if(_pMaxCellspc == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NULL pointer to output max Cellspace (" \
        << _vOutLyrNames[2] << ")" \
        << endl;
    return false;
  }
  _maxNoData = _pMaxCellspc->info()->getNoDataValAs<unsigned short>();

  return true;
}

pRPL::EvaluateReturn ZonalOutputTransition::
evaluate(const pRPL::CellCoord &coord) {
  unsigned short zone = _pZoneCellspc->atAs<unsigned short>(coord, true);

  float meanVal = _meanNoData;
  unsigned short minVal = _minNoData;
  unsigned short maxVal = _maxNoData;

  if(zone != _zoneNoData) {
    map<unsigned short, double>::const_iterator itrMean = _pmMeans->find(zone);
    if(itrMean != _pmMeans->end()) {
      meanVal = (float)(itrMean->second);
    }
    map<unsigned short, unsigned short>::const_iterator itrMin = _pmMins->find(zone);
    if(itrMin != _pmMins->end()) {
      minVal = itrMin->second;
    }
    map<unsigned short, unsigned short>::const_iterator itrMax = _pmMaxs->find(zone);
    if(itrMax != _pmMaxs->end()) {
      maxVal = itrMax->second;
    }
  }

  if(!_pMeanCellspc->updateCellAs<float>(coord, meanVal, true) ||
     !_pMinCellspc->updateCellAs<unsigned short>(coord, minVal, true) ||
     !_pMaxCellspc->updateCellAs<unsigned short>(coord, maxVal, true)) {
    return pRPL::EVAL_FAILED;
  }

  return pRPL::EVAL_SUCCEEDED;
}
