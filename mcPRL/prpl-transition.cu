#include "prpl-transition.h"
//#include "CuPRL.h"
//#include"errorhelper.h"
//#include"FocalOperator.h"
//
pRPL::Transition::
Transition(bool onlyUpdtCtrCell,
           bool needExchange,
           bool edgesFirst)
    :_pNbrhd(NULL),
     _onlyUpdtCtrCell(onlyUpdtCtrCell),
     _needExchange(needExchange),
     _edgesFirst(edgesFirst),
     _smCount(20){}

void pRPL::Transition::
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
pRPL::EvaluateReturn pRPL::Transition::
evalBR(const pRPL::CoordBR &br,bool isGPUCompute, pRPL::pCuf pf) {
  pRPL::Cellspace *pPrmSpc = getCellspaceByLyrName(getInLyrNames()[0]);
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: unable to find the primary Cellspace with name (" \
         << getPrimeLyrName() << ")" << endl;
    return pRPL::EVAL_FAILED;
  }

  pRPL::EvaluateReturn done = pRPL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
	  //vector<float>cuCoord;
   /* for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) {
      for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) {
        done = evaluate(pRPL::CellCoord(iRow, iCol));
        if(done == pRPL::EVAL_FAILED ||
           done == pRPL::EVAL_TERMINATED) {
          return done;
        }
      }
    }*/
	  if(!isGPUCompute)
	  {
	  for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) {
      for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) {
        done = evaluate(pRPL::CellCoord(iRow, iCol));
        if(done == pRPL::EVAL_FAILED ||
           done == pRPL::EVAL_TERMINATED) {
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
	//  case pRPL::cuLOCAL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuLocalOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	//  case pRPL::cuGLOBAL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuGlobalOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	//  case pRPL::cuZONEL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuZonelOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	 /* case 2:
		  pRPL::cuLocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;
	  case 3:
		   pRPL::cuZonalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;
	  case 4:
		  pRPL::cuGlobalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;*/
	  }


  return pRPL::EVAL_SUCCEEDED;
}
void pRPL::Transition::
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

bool pRPL::Transition::
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

const vector<string>& pRPL::Transition::
getInLyrNames() const {
  return _vInLyrNames;
}

const vector<string>& pRPL::Transition::
getOutLyrNames() const {
  return _vOutLyrNames;
}

const string& pRPL::Transition::
getPrimeLyrName() const {
  return _primeLyrName;
}

bool pRPL::Transition::
isInLyr(const string &lyrName) const {
  return (std::find(_vInLyrNames.begin(), _vInLyrNames.end(), lyrName) != _vInLyrNames.end());
}

bool pRPL::Transition::
isOutLyr(const string &lyrName) const {
  return (std::find(_vOutLyrNames.begin(), _vOutLyrNames.end(), lyrName) != _vOutLyrNames.end());
}

bool pRPL::Transition::
isPrimeLyr(const string &lyrName) const {
  return (lyrName == _primeLyrName);
}

void pRPL::Transition::
clearLyrSettings() {
  _vInLyrNames.clear();
  _vOutLyrNames.clear();
  _primeLyrName.clear();
  _mpCellspcs.clear();
}
void pRPL::Transition::
setSMcount(int _sm)
{
	_smCount=_sm;
}
bool pRPL::Transition::
setCellspace(const string &lyrName,
             pRPL::Cellspace *pCellspc) {
  map<string, pRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.find(lyrName);
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

void pRPL::Transition::
clearCellspaces() {
  map<string, pRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.begin();
  while(itrCellspc != _mpCellspcs.end()) {
    itrCellspc->second = NULL;
    itrCellspc++;
  }
}
bool pRPL::Transition::
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
bool pRPL::Transition::
initGlobalCoords(const vector<pRPL::CellCoord> &vGlobalCoords)
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
void pRPL::Transition::
	clearGlobalCoords()
{
	_globalCoords.clear();
}
void pRPL::Transition::
clearGPUMem()
{
	 map<string, pRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.begin();
	  while(itrCellspc != _mpCellspcs.end())
	  {
		  if(itrCellspc->second->getGPUData()!=NULL)
		  {
		  itrCellspc->second->deleteGPUData();
		  }
		   itrCellspc++;
	  }
}
pRPL::Cellspace* pRPL::Transition::
getCellspaceByLyrName(const string &lyrName) {
  pRPL::Cellspace *pCellspc = NULL;
  map<string, pRPL::Cellspace *>::iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc != _mpCellspcs.end()) {
    pCellspc = itrCellspc->second;
  }
 //std::cout<<lyrName;
  return pCellspc;
}

const pRPL::Cellspace* pRPL::Transition::
getCellspaceByLyrName(const string &lyrName) const {
  const pRPL::Cellspace *pCellspc = NULL;
  map<string, pRPL::Cellspace *>::const_iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc != _mpCellspcs.end()) {
    pCellspc = itrCellspc->second;
  }
  return pCellspc;
}

void pRPL::Transition::
setUpdateTracking(bool toTrack) {
  for(int iOutLyrName = 0; iOutLyrName < _vOutLyrNames.size(); iOutLyrName++) {
    map<string, Cellspace *>::iterator itrCellspc = _mpCellspcs.find(_vOutLyrNames[iOutLyrName]);
    if(itrCellspc != _mpCellspcs.end() &&
       itrCellspc->second != NULL) {
      itrCellspc->second->setUpdateTracking(toTrack);
    }
  }
}

void pRPL::Transition::
clearUpdateTracks() {
  for(int iOutLyrName = 0; iOutLyrName < _vOutLyrNames.size(); iOutLyrName++) {
    map<string, Cellspace *>::iterator itrCellspc = _mpCellspcs.find(_vOutLyrNames[iOutLyrName]);
    if(itrCellspc != _mpCellspcs.end() &&
       itrCellspc->second != NULL) {
      itrCellspc->second->clearUpdatedIdxs();
    }
  }
}

void pRPL::Transition::
setNbrhdByName(const char *aNbrhdName) {
  if(aNbrhdName != NULL) {
    _nbrhdName = aNbrhdName;
  }
}

const string& pRPL::Transition::
getNbrhdName() const {
  return _nbrhdName;
}

void pRPL::Transition::
clearNbrhdSetting() {
  _nbrhdName.clear();
  _pNbrhd = NULL;
}

void pRPL::Transition::
setNbrhd(Neighborhood *pNbrhd) {
  _pNbrhd = pNbrhd;
}

pRPL::Neighborhood* pRPL::Transition::
getNbrhd() {
  return _pNbrhd;
}

const pRPL::Neighborhood* pRPL::Transition::
getNbrhd() const {
  return _pNbrhd;
}

void pRPL::Transition::
clearDataSettings() {
  clearLyrSettings();
  clearNbrhdSetting();
}

bool pRPL::Transition::
onlyUpdtCtrCell() const {
  return _onlyUpdtCtrCell;
}

bool pRPL::Transition::
needExchange() const {
  return _needExchange;
}

bool pRPL::Transition::
edgesFirst() const {
  return _edgesFirst;
}


pRPL::EvaluateReturn pRPL::Transition::
evalRandomly(const pRPL::CoordBR &br) {
  pRPL::Cellspace *pPrmSpc = getCellspaceByLyrName(getPrimeLyrName());
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace with name (" \
        << getPrimeLyrName() << ")" << endl;
    return pRPL::EVAL_FAILED;
  }

  pRPL::EvaluateReturn done = pRPL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
    while(done != pRPL::EVAL_TERMINATED && done != pRPL::EVAL_FAILED) {
      long iRow = rand() % br.nRows() + br.minIRow();
      long iCol = rand() % br.nCols() + br.minICol();
      pRPL::CellCoord coord2Eval(iRow, iCol);
      if(br.ifContain(coord2Eval)) {
        done = evaluate(coord2Eval);
      }
    }
  }

  return done;
}

pRPL::EvaluateReturn pRPL::Transition::
evalSelected(const pRPL::CoordBR &br,
             const pRPL::LongVect &vlclIdxs) {
  pRPL::Cellspace *pPrmSpc = getCellspaceByLyrName(getPrimeLyrName());
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace with name (" \
        << getPrimeLyrName() << ")" << endl;
    return pRPL::EVAL_FAILED;
  }

  pRPL::EvaluateReturn done = pRPL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
    for(int iIdx = 0; iIdx < vlclIdxs.size(); iIdx++) {
      pRPL::CellCoord coord = pPrmSpc->info()->idx2coord(vlclIdxs[iIdx]);
      if(br.ifContain(coord)) {
        done = evaluate(coord);
        if(done == pRPL::EVAL_FAILED ||
           done == pRPL::EVAL_TERMINATED) {
          return done;
        }
      }
    }
  }

  return pRPL::EVAL_SUCCEEDED;
}

bool pRPL::Transition::
afterSetCellspaces(int subCellspcGlbIdx) {
  return true;
}

bool pRPL::Transition::
afterSetNbrhd() {
  return true;
}

bool pRPL::Transition::
check() const {
  return true;
}

pRPL::EvaluateReturn pRPL::Transition::
evaluate(const CellCoord &coord) {
  return pRPL::EVAL_SUCCEEDED;
}

double pRPL::Transition::
workload(const CoordBR &workBR) const {
  return 0.0;
}

void pRPL::Transition::addparam(double dparam)
{
	_vparamInfo.push_back(dparam);
}
