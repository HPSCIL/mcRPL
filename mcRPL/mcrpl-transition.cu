#include "mcrpl-transition.h"
//#include "CuPRL.h"
//#include"errorhelper.h"
//#include"FocalOperator.h"
//
mcRPL::Transition::
Transition(bool onlyUpdtCtrCell,
           bool needExchange,
           bool edgesFirst)
    :_pNbrhd(NULL),
     _onlyUpdtCtrCell(onlyUpdtCtrCell),
     _needExchange(needExchange),
     _edgesFirst(edgesFirst),
     _smCount(20){}

void mcRPL::Transition::
addInputLyr(const char *aInLyrName,
            bool isPrimeLyr) {
  if(aInLyrName != NULL &&
     std::find(_vInLyrNames.begin(), _vInLyrNames.end(), aInLyrName) == _vInLyrNames.end()) {
    _vInLyrNames.push_back(aInLyrName);
    if(_mpCellspcs.find(aInLyrName) == _mpCellspcs.end()) {
      _mpCellspcs[aInLyrName] = NULL;
    }
    if(isPrimeLyr) {
      _primeLyrName = aInLyrName;
    }
  }
}
mcRPL::EvaluateReturn mcRPL::Transition::
evalBR(const mcRPL::CoordBR &br,bool isGPUCompute, mcRPL::pCuf pf) {
  mcRPL::Cellspace *pPrmSpc = getCellspaceByLyrName(getInLyrNames()[0]);
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: unable to find the primary Cellspace with name (" \
         << getPrimeLyrName() << ")" << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::EvaluateReturn done = mcRPL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
	  //vector<float>cuCoord;
   /* for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) {
      for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) {
        done = evaluate(mcRPL::CellCoord(iRow, iCol));
        if(done == mcRPL::EVAL_FAILED ||
           done == mcRPL::EVAL_TERMINATED) {
          return done;
        }
      }
    }*/
	  if(!isGPUCompute)
	  {
	  for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) {
      for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) {
        done = evaluate(mcRPL::CellCoord(iRow, iCol));
        if(done == mcRPL::EVAL_FAILED ||
           done == mcRPL::EVAL_TERMINATED) {
          return done;
        }
      }
    }
	  }
	  else
	  {
		  (this->*pf)(br);
	  }
		// cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	//  case mcRPL::cuLOCAL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuLocalOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	//  case mcRPL::cuGLOBAL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuGlobalOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	//  case mcRPL::cuZONEL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuZonelOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	 /* case 2:
		  mcRPL::cuLocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;
	  case 3:
		   mcRPL::cuZonalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;
	  case 4:
		  mcRPL::cuGlobalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;*/
	  }


  return mcRPL::EVAL_SUCCEEDED;
}
void mcRPL::Transition::
addOutputLyr(const char *aOutLyrName,
             bool isPrimeLyr) {
  if(aOutLyrName != NULL &&
     std::find(_vOutLyrNames.begin(), _vOutLyrNames.end(), aOutLyrName) == _vOutLyrNames.end()) {
    _vOutLyrNames.push_back(aOutLyrName);
    if(_mpCellspcs.find(aOutLyrName) == _mpCellspcs.end()) {
      _mpCellspcs[aOutLyrName] = NULL;
    }
    if(isPrimeLyr) {
      _primeLyrName = aOutLyrName;
    }
  }
}

bool mcRPL::Transition::
setLyrsByNames(vector<string> *pvInLyrNames,
               vector<string> *pvOutLyrNames,
               string &primeLyrName) {
  clearLyrSettings();

  vector<string>::iterator itrLyrName;
  bool foundPrm = false;
  if(pvInLyrNames != NULL) {
    itrLyrName = std::find(pvInLyrNames->begin(), pvInLyrNames->end(), primeLyrName);
    if(itrLyrName != pvInLyrNames->end()) {
      _primeLyrName = primeLyrName;
      foundPrm = true;
    }
    _vInLyrNames = *pvInLyrNames;
  }
  if(pvOutLyrNames != NULL) {
    itrLyrName = find(pvOutLyrNames->begin(), pvOutLyrNames->end(), primeLyrName);
    if(itrLyrName != pvOutLyrNames->end()) {
      _primeLyrName = primeLyrName;
      foundPrm = true;
    }
    _vOutLyrNames = *pvOutLyrNames;
  }
  if(!foundPrm) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: primary Layer name (" << primeLyrName \
         << ") is not found in either Input Layer names or Output Layer names" << endl;
    return false;
  }

  for(int iLyrName = 0; iLyrName < _vInLyrNames.size(); iLyrName++) {
    const string &lyrName = _vInLyrNames[iLyrName];
    _mpCellspcs[lyrName] = NULL;
  }
  for(int iLyrName = 0; iLyrName < _vOutLyrNames.size(); iLyrName++) {
    const string &lyrName = _vOutLyrNames[iLyrName];
    if(std::find(_vInLyrNames.begin(), _vInLyrNames.end(), lyrName) != _vInLyrNames.end()) {
      /*
      cerr << __FILE__ << " function: " << __FUNCTION__ \
           << " Error: output Layer (" << lyrName \
           << ") can NOT be also input Layer" << endl;
      return false;
      */
      continue; // ignore the output Layer that is also an input Layer
    }
    _mpCellspcs[lyrName] = NULL;
  }

  return true;
}

const vector<string>& mcRPL::Transition::
getInLyrNames() const {
  return _vInLyrNames;
}

const vector<string>& mcRPL::Transition::
getOutLyrNames() const {
  return _vOutLyrNames;
}

const string& mcRPL::Transition::
getPrimeLyrName() const {
  return _primeLyrName;
}

bool mcRPL::Transition::
isInLyr(const string &lyrName) const {
  return (std::find(_vInLyrNames.begin(), _vInLyrNames.end(), lyrName) != _vInLyrNames.end());
}

bool mcRPL::Transition::
isOutLyr(const string &lyrName) const {
  return (std::find(_vOutLyrNames.begin(), _vOutLyrNames.end(), lyrName) != _vOutLyrNames.end());
}

bool mcRPL::Transition::
isPrimeLyr(const string &lyrName) const {
  return (lyrName == _primeLyrName);
}

void mcRPL::Transition::
clearLyrSettings() {
  _vInLyrNames.clear();
  _vOutLyrNames.clear();
  _primeLyrName.clear();
  _mpCellspcs.clear();
}
void mcRPL::Transition::
setSMcount(int _sm)
{
	_smCount=_sm;
}
bool mcRPL::Transition::
setCellspace(const string &lyrName,
             mcRPL::Cellspace *pCellspc) {
  map<string, mcRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc == _mpCellspcs.end()) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: unable to find a Layer with the name (" \
         << lyrName << ")" << endl;
    return false;
  }

  if(pCellspc == NULL ||
     pCellspc->isEmpty(true)) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: NULL or empty Cellspace to be added to the Transition" \
         << endl;
    return false;
  }
  itrCellspc->second = pCellspc;

  return true;
}

void mcRPL::Transition::
clearCellspaces() {
  map<string, mcRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.begin();
  while(itrCellspc != _mpCellspcs.end()) {
    itrCellspc->second = NULL;
    itrCellspc++;
  }
}
bool mcRPL::Transition::
initGlobalCoords(const vector<int> &vGlobalCoords)
{
	if(vGlobalCoords.empty())
	{
		  cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: NULL or empty vGlobalCoords to be added to the Transition" \
         << endl;
		return false;
	}
	int nSize=vGlobalCoords.size();
	for(int i=0;i<nSize;i++)
	{
		_globalCoords.push_back(vGlobalCoords[i]);
	}
	return true;
}
bool mcRPL::Transition::
initGlobalCoords(const vector<mcRPL::CellCoord> &vGlobalCoords)
{
	if(vGlobalCoords.empty())
	{
		  cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: NULL or empty vGlobalCoords to be added to the Transition" \
         << endl;
		return false;
	}
	int nSize=vGlobalCoords.size();
	for(int i=0;i<nSize;i++)
	{
		_globalCoords.push_back(vGlobalCoords[i].iCol());
		_globalCoords.push_back(vGlobalCoords[i].iRow());
	}
	return true;
}
void mcRPL::Transition::
	clearGlobalCoords()
{
	_globalCoords.clear();
}
void mcRPL::Transition::
clearGPUMem()
{
	 map<string, mcRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.begin();
	  while(itrCellspc != _mpCellspcs.end())
	  {
		  if(itrCellspc->second->getGPUData()!=NULL)
		  {
		  itrCellspc->second->deleteGPUData();
		  }
		   itrCellspc++;
	  }
}
mcRPL::Cellspace* mcRPL::Transition::
getCellspaceByLyrName(const string &lyrName) {
  mcRPL::Cellspace *pCellspc = NULL;
  map<string, mcRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc != _mpCellspcs.end()) {
    pCellspc = itrCellspc->second;
  }
 //std::cout<<lyrName;
  return pCellspc;
}

const mcRPL::Cellspace* mcRPL::Transition::
getCellspaceByLyrName(const string &lyrName) const {
  const mcRPL::Cellspace *pCellspc = NULL;
  map<string, mcRPL::Cellspace *>::const_iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc != _mpCellspcs.end()) {
    pCellspc = itrCellspc->second;
  }
  return pCellspc;
}

void mcRPL::Transition::
setUpdateTracking(bool toTrack) {
  for(int iOutLyrName = 0; iOutLyrName < _vOutLyrNames.size(); iOutLyrName++) {
    map<string, Cellspace *>::iterator itrCellspc = _mpCellspcs.find(_vOutLyrNames[iOutLyrName]);
    if(itrCellspc != _mpCellspcs.end() &&
       itrCellspc->second != NULL) {
      itrCellspc->second->setUpdateTracking(toTrack);
    }
  }
}

void mcRPL::Transition::
clearUpdateTracks() {
  for(int iOutLyrName = 0; iOutLyrName < _vOutLyrNames.size(); iOutLyrName++) {
    map<string, Cellspace *>::iterator itrCellspc = _mpCellspcs.find(_vOutLyrNames[iOutLyrName]);
    if(itrCellspc != _mpCellspcs.end() &&
       itrCellspc->second != NULL) {
      itrCellspc->second->clearUpdatedIdxs();
    }
  }
}

void mcRPL::Transition::
setNbrhdByName(const char *aNbrhdName) {
  if(aNbrhdName != NULL) {
    _nbrhdName = aNbrhdName;
  }
}

const string& mcRPL::Transition::
getNbrhdName() const {
  return _nbrhdName;
}

void mcRPL::Transition::
clearNbrhdSetting() {
  _nbrhdName.clear();
  _pNbrhd = NULL;
}

void mcRPL::Transition::
setNbrhd(Neighborhood *pNbrhd) {
  _pNbrhd = pNbrhd;
}

mcRPL::Neighborhood* mcRPL::Transition::
getNbrhd() {
  return _pNbrhd;
}

const mcRPL::Neighborhood* mcRPL::Transition::
getNbrhd() const {
  return _pNbrhd;
}

void mcRPL::Transition::
clearDataSettings() {
  clearLyrSettings();
  clearNbrhdSetting();
}

bool mcRPL::Transition::
onlyUpdtCtrCell() const {
  return _onlyUpdtCtrCell;
}

bool mcRPL::Transition::
needExchange() const {
  return _needExchange;
}

bool mcRPL::Transition::
edgesFirst() const {
  return _edgesFirst;
}


mcRPL::EvaluateReturn mcRPL::Transition::
evalRandomly(const mcRPL::CoordBR &br) {
  mcRPL::Cellspace *pPrmSpc = getCellspaceByLyrName(getPrimeLyrName());
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace with name (" \
        << getPrimeLyrName() << ")" << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::EvaluateReturn done = mcRPL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
    while(done != mcRPL::EVAL_TERMINATED && done != mcRPL::EVAL_FAILED) {
      long iRow = rand() % br.nRows() + br.minIRow();
      long iCol = rand() % br.nCols() + br.minICol();
      mcRPL::CellCoord coord2Eval(iRow, iCol);
      if(br.ifContain(coord2Eval)) {
        done = evaluate(coord2Eval);
      }
    }
  }

  return done;
}

mcRPL::EvaluateReturn mcRPL::Transition::
evalSelected(const mcRPL::CoordBR &br,
             const mcRPL::LongVect &vlclIdxs) {
  mcRPL::Cellspace *pPrmSpc = getCellspaceByLyrName(getPrimeLyrName());
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace with name (" \
        << getPrimeLyrName() << ")" << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::EvaluateReturn done = mcRPL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
    for(int iIdx = 0; iIdx < vlclIdxs.size(); iIdx++) {
      mcRPL::CellCoord coord = pPrmSpc->info()->idx2coord(vlclIdxs[iIdx]);
      if(br.ifContain(coord)) {
        done = evaluate(coord);
        if(done == mcRPL::EVAL_FAILED ||
           done == mcRPL::EVAL_TERMINATED) {
          return done;
        }
      }
    }
  }

  return mcRPL::EVAL_SUCCEEDED;
}

bool mcRPL::Transition::
afterSetCellspaces(int subCellspcGlbIdx) {
  return true;
}

bool mcRPL::Transition::
afterSetNbrhd() {
  return true;
}

bool mcRPL::Transition::
check() const {
  return true;
}

mcRPL::EvaluateReturn mcRPL::Transition::
evaluate(const CellCoord &coord) {
  return mcRPL::EVAL_SUCCEEDED;
}

double mcRPL::Transition::
workload(const CoordBR &workBR) const {
  return 0.0;
}

void mcRPL::Transition::addparam(double dparam)
{
	_vparamInfo.push_back(dparam);
}
