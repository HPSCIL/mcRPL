#include "mcPRL-transition.h"
//#include "CuPRL.h"
//#include"errorhelper.h"
//#include"FocalOperator.h"
//
mcPRL::Transition::
Transition(bool onlyUpdtCtrCell,
           bool needExchange,
           bool edgesFirst)
    :_pNbrhd(NULL),
     _onlyUpdtCtrCell(onlyUpdtCtrCell),
     _needExchange(needExchange),
     _edgesFirst(edgesFirst),
     _smCount(20){}

void mcPRL::Transition::
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
mcPRL::EvaluateReturn mcPRL::Transition::
evalBR(const mcPRL::CoordBR &br,bool isGPUCompute, mcPRL::pCuf pf) {
  mcPRL::Cellspace *pPrmSpc = getCellspaceByLyrName(getInLyrNames()[0]);
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
         << " Error: unable to find the primary Cellspace with name (" \
         << getPrimeLyrName() << ")" << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::EvaluateReturn done = mcPRL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
	  //vector<float>cuCoord;
   /* for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) {
      for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) {
        done = evaluate(mcPRL::CellCoord(iRow, iCol));
        if(done == mcPRL::EVAL_FAILED ||
           done == mcPRL::EVAL_TERMINATED) {
          return done;
        }
      }
    }*/
	  if(!isGPUCompute)
	  {
	  for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) {
      for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) {
        done = evaluate(mcPRL::CellCoord(iRow, iCol));
        if(done == mcPRL::EVAL_FAILED ||
           done == mcPRL::EVAL_TERMINATED) {
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
	//  case mcPRL::cuLOCAL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuLocalOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	//  case mcPRL::cuGLOBAL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuGlobalOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	//  case mcPRL::cuZONEL:
	//	  cuFocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
	////	  cuZonelOperator<DataInType,DataOutType, WeightType,OperType>(br);
	//	  break;
	 /* case 2:
		  mcPRL::cuLocalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;
	  case 3:
		   mcPRL::cuZonalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;
	  case 4:
		  mcPRL::cuGlobalOperator<DataInType, DataOutType, WeightType,OperType>(br);
		  break;*/
	  }


  return mcPRL::EVAL_SUCCEEDED;
}
void mcPRL::Transition::
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

bool mcPRL::Transition::
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

const vector<string>& mcPRL::Transition::
getInLyrNames() const {
  return _vInLyrNames;
}

const vector<string>& mcPRL::Transition::
getOutLyrNames() const {
  return _vOutLyrNames;
}

const string& mcPRL::Transition::
getPrimeLyrName() const {
  return _primeLyrName;
}

bool mcPRL::Transition::
isInLyr(const string &lyrName) const {
  return (std::find(_vInLyrNames.begin(), _vInLyrNames.end(), lyrName) != _vInLyrNames.end());
}

bool mcPRL::Transition::
isOutLyr(const string &lyrName) const {
  return (std::find(_vOutLyrNames.begin(), _vOutLyrNames.end(), lyrName) != _vOutLyrNames.end());
}

bool mcPRL::Transition::
isPrimeLyr(const string &lyrName) const {
  return (lyrName == _primeLyrName);
}

void mcPRL::Transition::
clearLyrSettings() {
  _vInLyrNames.clear();
  _vOutLyrNames.clear();
  _primeLyrName.clear();
  _mpCellspcs.clear();
}
void mcPRL::Transition::
setSMcount(int _sm)
{
	_smCount=_sm;
}
bool mcPRL::Transition::
setCellspace(const string &lyrName,
             mcPRL::Cellspace *pCellspc) {
  map<string, mcPRL::Cellspace *>::iterator itrCellspc = _mpCellspcs.find(lyrName);
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

void mcPRL::Transition::
clearCellspaces() {
  map<string, mcPRL::Cellspace *>::iterator itrCellspc = _mpCellspcs.begin();
  while(itrCellspc != _mpCellspcs.end()) {
    itrCellspc->second = NULL;
    itrCellspc++;
  }
}
bool mcPRL::Transition::
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
bool mcPRL::Transition::
initGlobalCoords(const vector<mcPRL::CellCoord> &vGlobalCoords)
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
void mcPRL::Transition::
	clearGlobalCoords()
{
	_globalCoords.clear();
}
void mcPRL::Transition::
clearGPUMem()
{
	 map<string, mcPRL::Cellspace *>::iterator itrCellspc = _mpCellspcs.begin();
	  while(itrCellspc != _mpCellspcs.end())
	  {
		  if(itrCellspc->second->getGPUData()!=NULL)
		  {
		  itrCellspc->second->deleteGPUData();
		  }
		   itrCellspc++;
	  }
}
mcPRL::Cellspace* mcPRL::Transition::
getCellspaceByLyrName(const string &lyrName) {
  mcPRL::Cellspace *pCellspc = NULL;
  map<string, mcPRL::Cellspace *>::iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc != _mpCellspcs.end()) {
    pCellspc = itrCellspc->second;
  }
 //std::cout<<lyrName;
  return pCellspc;
}

const mcPRL::Cellspace* mcPRL::Transition::
getCellspaceByLyrName(const string &lyrName) const {
  const mcPRL::Cellspace *pCellspc = NULL;
  map<string, mcPRL::Cellspace *>::const_iterator itrCellspc = _mpCellspcs.find(lyrName);
  if(itrCellspc != _mpCellspcs.end()) {
    pCellspc = itrCellspc->second;
  }
  return pCellspc;
}

void mcPRL::Transition::
setUpdateTracking(bool toTrack) {
  for(int iOutLyrName = 0; iOutLyrName < _vOutLyrNames.size(); iOutLyrName++) {
    map<string, Cellspace *>::iterator itrCellspc = _mpCellspcs.find(_vOutLyrNames[iOutLyrName]);
    if(itrCellspc != _mpCellspcs.end() &&
       itrCellspc->second != NULL) {
      itrCellspc->second->setUpdateTracking(toTrack);
    }
  }
}

void mcPRL::Transition::
clearUpdateTracks() {
  for(int iOutLyrName = 0; iOutLyrName < _vOutLyrNames.size(); iOutLyrName++) {
    map<string, Cellspace *>::iterator itrCellspc = _mpCellspcs.find(_vOutLyrNames[iOutLyrName]);
    if(itrCellspc != _mpCellspcs.end() &&
       itrCellspc->second != NULL) {
      itrCellspc->second->clearUpdatedIdxs();
    }
  }
}

void mcPRL::Transition::
setNbrhdByName(const char *aNbrhdName) {
  if(aNbrhdName != NULL) {
    _nbrhdName = aNbrhdName;
  }
}

const string& mcPRL::Transition::
getNbrhdName() const {
  return _nbrhdName;
}

void mcPRL::Transition::
clearNbrhdSetting() {
  _nbrhdName.clear();
  _pNbrhd = NULL;
}

void mcPRL::Transition::
setNbrhd(Neighborhood *pNbrhd) {
  _pNbrhd = pNbrhd;
}

mcPRL::Neighborhood* mcPRL::Transition::
getNbrhd() {
  return _pNbrhd;
}

const mcPRL::Neighborhood* mcPRL::Transition::
getNbrhd() const {
  return _pNbrhd;
}

void mcPRL::Transition::
clearDataSettings() {
  clearLyrSettings();
  clearNbrhdSetting();
}

bool mcPRL::Transition::
onlyUpdtCtrCell() const {
  return _onlyUpdtCtrCell;
}

bool mcPRL::Transition::
needExchange() const {
  return _needExchange;
}

bool mcPRL::Transition::
edgesFirst() const {
  return _edgesFirst;
}


mcPRL::EvaluateReturn mcPRL::Transition::
evalRandomly(const mcPRL::CoordBR &br) {
  mcPRL::Cellspace *pPrmSpc = getCellspaceByLyrName(getPrimeLyrName());
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace with name (" \
        << getPrimeLyrName() << ")" << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::EvaluateReturn done = mcPRL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
    while(done != mcPRL::EVAL_TERMINATED && done != mcPRL::EVAL_FAILED) {
      long iRow = rand() % br.nRows() + br.minIRow();
      long iCol = rand() % br.nCols() + br.minICol();
      mcPRL::CellCoord coord2Eval(iRow, iCol);
      if(br.ifContain(coord2Eval)) {
        done = evaluate(coord2Eval);
      }
    }
  }

  return done;
}

mcPRL::EvaluateReturn mcPRL::Transition::
evalSelected(const mcPRL::CoordBR &br,
             const mcPRL::LongVect &vlclIdxs) {
  mcPRL::Cellspace *pPrmSpc = getCellspaceByLyrName(getPrimeLyrName());
  if(pPrmSpc == NULL) {
    cerr << __FILE__ << " function: " << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace with name (" \
        << getPrimeLyrName() << ")" << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::EvaluateReturn done = mcPRL::EVAL_SUCCEEDED;
  if(br.valid(pPrmSpc->info()->dims())) {
    for(int iIdx = 0; iIdx < vlclIdxs.size(); iIdx++) {
      mcPRL::CellCoord coord = pPrmSpc->info()->idx2coord(vlclIdxs[iIdx]);
      if(br.ifContain(coord)) {
        done = evaluate(coord);
        if(done == mcPRL::EVAL_FAILED ||
           done == mcPRL::EVAL_TERMINATED) {
          return done;
        }
      }
    }
  }

  return mcPRL::EVAL_SUCCEEDED;
}

bool mcPRL::Transition::
afterSetCellspaces(int subCellspcGlbIdx) {
  return true;
}

bool mcPRL::Transition::
afterSetNbrhd() {
  return true;
}

bool mcPRL::Transition::
check() const {
  return true;
}

mcPRL::EvaluateReturn mcPRL::Transition::
evaluate(const CellCoord &coord) {
  return mcPRL::EVAL_SUCCEEDED;
}

double mcPRL::Transition::
workload(const CoordBR &workBR) const {
  return 0.0;
}

void mcPRL::Transition::addparam(double dparam)
{
	_vparamInfo.push_back(dparam);
}
