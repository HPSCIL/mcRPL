#include "mcPRL-neighborhood.h"

//#include <sys/_types/_size_t.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

/****************************************
*             Private Methods           *
*****************************************/
bool mcPRL::Neighborhood::
_checkNbrCoords(const vector<mcPRL::CellCoord> &vNbrCoords) const {
  bool valid = true;
  if(vNbrCoords.empty()) {
    cerr << "Error: the coordinate vector is empty" << endl;
    valid = false;
  }
  else if(std::find(vNbrCoords.begin(), vNbrCoords.end(), CellCoord(0, 0)) == vNbrCoords.end()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: the coordinate vector has to have a coordinate [0, 0]" << endl;
    valid = false;
  }
  return valid;
}

bool mcPRL::Neighborhood::
_validNumDirects() const {
  bool numDirectsValid = true;
  if(_mNbrIDMap.size() != 8) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid _mNbrIDMap size" << endl;
    numDirectsValid = false;
  }
  return numDirectsValid;
}

/****************************************
*             Public Methods            *
*****************************************/
mcPRL::Neighborhood::
Neighborhood(const char *aName,
             mcPRL::EdgeOption edgeOption,
             double virtualEdgeVal) {
  if(aName != NULL) {
    _name.assign(aName);
  }
  else {
    _name.assign("Untitled_Neighborhood");
  }
  _edgeOption = edgeOption;
  _pVirtualEdgeVal = NULL;

  if(edgeOption == mcPRL::CUSTOM_VIRTUAL_EDGES) {
    _pVirtualEdgeVal = new double;
    memcpy(_pVirtualEdgeVal, &virtualEdgeVal, sizeof(double));
  }
}

mcPRL::Neighborhood::
Neighborhood(const mcPRL::Neighborhood &rhs) {
  _name = rhs._name;
  _vNbrs = rhs._vNbrs;
  _MBR = rhs._MBR;
  _mNbrIDMap = rhs._mNbrIDMap;
  _edgeOption = rhs._edgeOption;
  if(rhs._pVirtualEdgeVal != NULL) {
    _pVirtualEdgeVal = new double(*(rhs._pVirtualEdgeVal));
  }
  else {
    _pVirtualEdgeVal = NULL;
  }
}

mcPRL::Neighborhood::
~Neighborhood() {
  if(_pVirtualEdgeVal != NULL) {
    delete _pVirtualEdgeVal;
  }
}
void mcPRL::Neighborhood::addnbrForGpu()
{
	 long long  nbrsize=this->size();
	 vector<mcPRL::WeightedCellCoord>coords =this->getInnerNbr(); 
	  for(int iNbr = 0; iNbr < nbrsize; iNbr++)
	  {
		  /* if(coords[iNbr].iRow() == 0 && coords[iNbr].iCol() == 0 ) 
		  {
		  continue;
		  }*/
		  cuNbrCoords.push_back(coords[iNbr].iRow());
		  cuNbrCoords.push_back(coords[iNbr].iCol());
		  cuNbrWeights.push_back(coords[iNbr].weight());
	  }
}
bool mcPRL::Neighborhood::
init(const vector<mcPRL::CellCoord> &vNbrCoords,
     double weight,
     mcPRL::EdgeOption edgeOption,
     double virtualEdgeVal) {
  if(!_checkNbrCoords(vNbrCoords)) {
    return false;
  }

  _vNbrs.clear();
  _mNbrIDMap.clear();
  _mNbrIDMap.resize(8);

  _edgeOption = edgeOption;
  _pVirtualEdgeVal = NULL;
  if(edgeOption == mcPRL::CUSTOM_VIRTUAL_EDGES) {
    _pVirtualEdgeVal = new double;
    memcpy(_pVirtualEdgeVal, &virtualEdgeVal, sizeof(double));
  }

  CellCoord nwCorner, seCorner;
  for(int iNbr = 0; iNbr < vNbrCoords.size(); iNbr++) {
    vector<mcPRL::WeightedCellCoord>::iterator nbrItr = _vNbrs.begin();
    while(nbrItr != _vNbrs.end()) {
      mcPRL::CellCoord coord(nbrItr->iRow(), nbrItr->iCol());
      if(coord == vNbrCoords[iNbr]) {
        cerr << __FILE__ << " " << __FUNCTION__ \
             << " Error: a duplicated coordinate [" \
             << vNbrCoords[iNbr] \
             << "] has been found" << endl;
        return false;
      }
      nbrItr++;
    }

    _vNbrs.push_back(mcPRL::WeightedCellCoord(vNbrCoords[iNbr].iRow(), vNbrCoords[iNbr].iCol(), weight));

    // calc _MBR
    if(0 == iNbr) {
      nwCorner = vNbrCoords[iNbr];
      seCorner = vNbrCoords[iNbr];
    }
    else{
      if(nwCorner.iRow() > vNbrCoords[iNbr].iRow()) {
         nwCorner.iRow(vNbrCoords[iNbr].iRow());
      }
      if(nwCorner.iCol() > vNbrCoords[iNbr].iCol()) {
         nwCorner.iCol(vNbrCoords[iNbr].iCol());
      }
      if(seCorner.iRow() < vNbrCoords[iNbr].iRow()) {
         seCorner.iRow(vNbrCoords[iNbr].iRow());
      }
      if(seCorner.iCol() < vNbrCoords[iNbr].iCol()) {
         seCorner.iCol(vNbrCoords[iNbr].iCol());
      }
    }

    // calc _mNbrIDMap
    if(vNbrCoords[iNbr].iRow() < 0) {
      if(vNbrCoords[iNbr].iCol() < 0) {
        _mNbrIDMap[mcPRL::NORTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcPRL::NORTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcPRL::NORTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iRow() > 0) {
      if(vNbrCoords[iNbr].iCol() < 0) {
        _mNbrIDMap[mcPRL::SOUTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcPRL::SOUTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcPRL::SOUTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() < 0) {
      _mNbrIDMap[mcPRL::WEST_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() > 0) {
      _mNbrIDMap[mcPRL::EAST_DIR].push_back(iNbr);
    }
  } // End of iNbr loop

  _MBR.nwCorner(nwCorner);
  _MBR.seCorner(seCorner);
  addnbrForGpu();
  return true;
}

bool mcPRL::Neighborhood::
init(const vector<mcPRL::WeightedCellCoord> &vNbrCoords,
     mcPRL::EdgeOption edgeOption,
     double virtualEdgeVal) {
  if(!_checkNbrCoords((const vector<mcPRL::CellCoord> &)vNbrCoords)) {
    return false;
  }
  
  _vNbrs.clear();
  _mNbrIDMap.clear();
  _mNbrIDMap.resize(8);
  
  _edgeOption = edgeOption;
  _pVirtualEdgeVal = NULL;
  if(edgeOption == mcPRL::CUSTOM_VIRTUAL_EDGES) {
    _pVirtualEdgeVal = new double;
    memcpy(_pVirtualEdgeVal, &virtualEdgeVal, sizeof(double));
  }

  mcPRL::CellCoord nwCorner, seCorner;
  for(int iNbr = 0; iNbr < vNbrCoords.size(); iNbr++) {
    if(std::find(_vNbrs.begin(), _vNbrs.end(), vNbrCoords[iNbr]) != _vNbrs.end()) {
      cerr << __FILE__ << " " << __FUNCTION__ \
      << " Error: a duplicated coordinate [" \
      << vNbrCoords[iNbr] \
      << "] has been found" << endl;
      return false;
    }
    
    _vNbrs.push_back(mcPRL::WeightedCellCoord(vNbrCoords[iNbr]));
    
    // calc _MBR
    if(0 == iNbr) {
      nwCorner = vNbrCoords[iNbr];
      seCorner = vNbrCoords[iNbr];
    }
    else{
      if(nwCorner.iRow() > vNbrCoords[iNbr].iRow()) {
        nwCorner.iRow(vNbrCoords[iNbr].iRow());
      }
      if(nwCorner.iCol() > vNbrCoords[iNbr].iCol()) {
        nwCorner.iCol(vNbrCoords[iNbr].iCol());
      }
      if(seCorner.iRow() < vNbrCoords[iNbr].iRow()) {
        seCorner.iRow(vNbrCoords[iNbr].iRow());
      }
      if(seCorner.iCol() < vNbrCoords[iNbr].iCol()) {
        seCorner.iCol(vNbrCoords[iNbr].iCol());
      }
    }
    
    // calc _mNbrIDMap
    if(vNbrCoords[iNbr].iRow() < 0) {
      if(vNbrCoords[iNbr].iCol() < 0) {
        _mNbrIDMap[mcPRL::NORTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcPRL::NORTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcPRL::NORTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iRow() > 0) {
      if(vNbrCoords[iNbr].iCol() < 0) {
        _mNbrIDMap[mcPRL::SOUTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcPRL::SOUTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcPRL::SOUTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() < 0) {
      _mNbrIDMap[mcPRL::WEST_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() > 0) {
      _mNbrIDMap[mcPRL::EAST_DIR].push_back(iNbr);
    }
  } // End of iNbr loop
  
  _MBR.nwCorner(nwCorner);
  _MBR.seCorner(seCorner);
  addnbrForGpu();
  return true;
}

bool mcPRL::Neighborhood::
initSingleCell(double weight) {
  vector<mcPRL::CellCoord> vNbrCoords;
  vNbrCoords.push_back(mcPRL::CellCoord(0, 0));
  return init(vNbrCoords, weight, mcPRL::FORBID_VIRTUAL_EDGES);
}

bool mcPRL::Neighborhood::
initVonNeumann(double weight,
               mcPRL::EdgeOption edgeOption,
               double virtualEdgeVal) {
  vector<mcPRL::CellCoord> vNbrCoords;
  vNbrCoords.push_back(mcPRL::CellCoord(0, 0));
  vNbrCoords.push_back(mcPRL::CellCoord(-1, 0));
  vNbrCoords.push_back(mcPRL::CellCoord(0, 1));
  vNbrCoords.push_back(mcPRL::CellCoord(1, 0));
  vNbrCoords.push_back(mcPRL::CellCoord(0, -1));
  return init(vNbrCoords, weight, edgeOption, virtualEdgeVal);
}

bool mcPRL::Neighborhood::
initMoore(long nEdgeLength,
          double weight,
          mcPRL::EdgeOption edgeOption,
          double virtualEdgeVal) {
  if(nEdgeLength <= 0 || nEdgeLength % 2 == 0) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid edge length (" << nEdgeLength \
         << "). A Moore neighborhood's edge length must be a positive odd number" << endl;
    return false;
  }

  vector<mcPRL::CellCoord> vNbrCoords;
  vNbrCoords.push_back(mcPRL::CellCoord(0, 0));
  for(long iRow = -(nEdgeLength/2); iRow <= nEdgeLength/2; iRow++) {
    for(long iCol = -(nEdgeLength/2); iCol <= nEdgeLength/2; iCol++) {
      if(iRow == 0 && iCol == 0) {
        continue;
      }
      vNbrCoords.push_back(mcPRL::CellCoord(iRow, iCol));
    }
  }

  return init(vNbrCoords, weight, edgeOption, virtualEdgeVal);
}

void mcPRL::Neighborhood::
clear() {
  _name.clear();
  _vNbrs.clear();
  _mNbrIDMap.clear();
  _MBR = CoordBR();
  _edgeOption = mcPRL::FORBID_VIRTUAL_EDGES;
  if(_pVirtualEdgeVal != NULL) {
    delete _pVirtualEdgeVal;
    _pVirtualEdgeVal = NULL;
  }
}

mcPRL::WeightedCellCoord& mcPRL::Neighborhood::
operator[](int iNbr) {
  return _vNbrs.at(iNbr);
}

const mcPRL::WeightedCellCoord& mcPRL::Neighborhood::
operator[](int iNbr) const {
  return _vNbrs.at(iNbr);
}

mcPRL::Neighborhood& mcPRL::Neighborhood::
operator=(const mcPRL::Neighborhood &rhs) {
  if(this != &rhs) {
    clear();
    _vNbrs = rhs._vNbrs;
    _MBR = rhs._MBR;
    _mNbrIDMap = rhs._mNbrIDMap;
    _edgeOption = rhs._edgeOption;
    if(rhs._pVirtualEdgeVal != NULL) {
      _pVirtualEdgeVal = new double(*(rhs._pVirtualEdgeVal));
    }
  }
  return *this;
}

const char* mcPRL::Neighborhood::
name() const {
  return _name.c_str();
}

void mcPRL::Neighborhood::
name(const char *aName) {
  if(aName != NULL) {
    _name.assign(aName);
  }
}

bool mcPRL::Neighborhood::
isEmpty() const{
  return _vNbrs.empty();
}

int mcPRL::Neighborhood::
size() const{
  return _vNbrs.size();  
}

bool mcPRL::Neighborhood::
isEquallyWeighted(double &weight) const {
  bool equal = true;
  weight = 0.0;
  if(_vNbrs.empty()) {
    return equal;
  }
  weight = _vNbrs[0].weight();
  for(int iNbr = 1; iNbr < _vNbrs.size(); iNbr++) {
    if(weight != _vNbrs[iNbr].weight()) {
      equal = false;
      break;
    }
  }
  return equal;
}
 vector<double> mcPRL::Neighborhood::cuWeights()
 {
	 return cuNbrWeights;
 }
  vector<int>mcPRL::Neighborhood::cuCoords()
  {
	  return cuNbrCoords;
  }
long mcPRL::Neighborhood::
minIRow() const {
  return _MBR.minIRow();
}

long mcPRL::Neighborhood::
minICol() const {
  return _MBR.minICol();
}

long mcPRL::Neighborhood::
maxIRow() const {
  return _MBR.maxIRow();
}

long mcPRL::Neighborhood::
maxICol() const {
  return _MBR.maxICol();
}

long mcPRL::Neighborhood::
nRows() const {
  return _MBR.nRows();
}

long mcPRL::Neighborhood::
nCols() const {
  return _MBR.nCols();
}

const mcPRL::CoordBR* mcPRL::Neighborhood::
getMBR() const {
  return &(_MBR);
}

mcPRL::EdgeOption mcPRL::Neighborhood::
edgeOption() const {
  return _edgeOption;
}

const double* mcPRL::Neighborhood::
virtualEdgeVal() const {
  return _pVirtualEdgeVal;
}

bool mcPRL::Neighborhood::
virtualEdgeVal(const double veVal) {
  if(_edgeOption != mcPRL::CUSTOM_VIRTUAL_EDGES) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: can NOT customize the virtual edge value," \
         << " because the Neighborhood\'s edge option is NOT CUSTOM_VIRTUAL_EDGE" \
         << endl;
    return false;
  }
  if(_pVirtualEdgeVal == NULL) {
    _pVirtualEdgeVal = new double(veVal);
  }
  else {
    memcpy(_pVirtualEdgeVal, &veVal, sizeof(double));
  }
  return true;
}

bool mcPRL::Neighborhood::
hasNbrs(mcPRL::MeshDir dir) const {
  bool nbrExist;
  if(!_validNumDirects()) {
    nbrExist = false; 
  }
  else {
    if(!_mNbrIDMap[dir].empty()) {
      nbrExist = true;
    }
    else {
      nbrExist = false;
    }
  }
  return nbrExist;
}

const mcPRL::IntVect* mcPRL::Neighborhood::
nbrIDs(mcPRL::MeshDir dir) const {
  return &(_mNbrIDMap.at(dir));
}

bool mcPRL::Neighborhood::
calcWorkBR(mcPRL::CoordBR &workBR,
           const mcPRL::SpaceDims &dims) const {
  if(isEmpty()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: unable to calculate workBR" \
         << " using an empty Neighborhood" << endl;
    return false;
  }
  if(dims.isNone()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: unable to calculate workBR" \
         << " using an empty SpaceDims" << endl;
    return false;
  }
  if(dims.nRows() < nRows() ||
     dims.nCols() < nCols()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: the dimensions (" << dims << ") is smaller than" \
         << " the dimensions of the Neighborhood (" << nRows() \
         << " " << nCols() << ")" << endl;
    return false;
  }

  if(_edgeOption == FORBID_VIRTUAL_EDGES) {
    workBR.nwCorner(-minIRow(), -minICol());
    workBR.seCorner(dims.nRows() - maxIRow() - 1,
                    dims.nCols() - maxICol() - 1);

    /*
    if((maxIRow() > 0 && workBR.nRows() < maxIRow()) ||
       (minIRow() < 0 && workBR.nRows() < -minIRow()) ||
       (maxICol() > 0 && workBR.nCols() < maxICol()) ||
       (minICol() < 0 && workBR.nCols() < -minICol())) {
      cerr << __FILE__ << " " << __FUNCTION__ \
           << " Error: workBR (" << workBR \
           << ") is too small to accommodate the Neighborhood" << endl;
      return false;
    }
    */
  }
  else {
    workBR.nwCorner(0, 0);
    workBR.seCorner(dims.nRows() - 1,
                    dims.nCols() - 1);
  }

  if(!workBR.valid(dims)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid workBR (" << workBR \
         << ") has been produced" << endl;
    return false;
  }

  return true;
}

void mcPRL::Neighborhood::
add2Buf(mcPRL::CharVect &vBuf) const {
  size_t optionSize = sizeof(mcPRL::EdgeOption);
  size_t coordSize = sizeof(mcPRL::WeightedCellCoord);
  size_t intSize = sizeof(int);

  int bufSize = vBuf.size();
  vBuf.resize(bufSize + optionSize + intSize + size()*coordSize);

  memcpy(&(vBuf[bufSize]), &(_edgeOption), optionSize);
  bufSize += optionSize;

  int nNbrs = size();
  memcpy(&(vBuf[bufSize]), &(nNbrs), intSize);
  bufSize += intSize;

  for(int iNbr = 0; iNbr < size(); iNbr++) {
    memcpy(&(vBuf[bufSize]), &(_vNbrs[iNbr]), coordSize);
    bufSize += coordSize;
  }
}

bool mcPRL::Neighborhood::
fromBuf(const mcPRL::CharVect &vBuf,
        int &iChar) {
  size_t optionSize = sizeof(mcPRL::EdgeOption);
  size_t coordSize = sizeof(mcPRL::WeightedCellCoord);
  size_t intSize = sizeof(int);

  if(vBuf.size() - iChar < optionSize) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: buffer is too small to extract a Neighborhood" << endl;
    return false;
  }
  mcPRL::EdgeOption edgeOption;
  memcpy(&edgeOption, &(vBuf[iChar]), optionSize);
  iChar += optionSize;

  if(vBuf.size() - iChar < intSize) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: buffer is too small to extract a Neighborhood" << endl;
    return false;
  }
  int nNbrs;
  memcpy(&nNbrs, &(vBuf[iChar]), intSize);
  iChar += intSize;

  if(vBuf.size() - iChar < nNbrs*coordSize) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: buffer is too small to extract a Neighborhood" << endl;
    return false;
  }
  mcPRL::WeightedCellCoord coord;
  vector<mcPRL::WeightedCellCoord> vNbrCoords;
  for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
    memcpy(&coord, &(vBuf[iChar]), coordSize);
    vNbrCoords.push_back(WeightedCellCoord(coord));
    iChar += coordSize;
  }
  return init(vNbrCoords, edgeOption);
}

ostream& mcPRL::
operator<<(ostream &os,
           const mcPRL::Neighborhood &nbrhd) {
  os << nbrhd.edgeOption() << endl;
  int nNbrs = nbrhd.size();
  os << nNbrs << endl;
  for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
    os << nbrhd[iNbr];
    os << endl;
  }
  return os;
}

istream& mcPRL::
operator>>(istream &is,
           mcPRL::Neighborhood &nbrhd) {
  mcPRL::EdgeOption edgeOption;
  vector<mcPRL::WeightedCellCoord> vNbrCoords;
  mcPRL::WeightedCellCoord coord;
  int iEdgeOption, nNbrs;
  
  is >> iEdgeOption;
  if(iEdgeOption >= mcPRL::FORBID_VIRTUAL_EDGES &&
     iEdgeOption <= mcPRL::CUSTOM_VIRTUAL_EDGES) {
    edgeOption = (mcPRL::EdgeOption)iEdgeOption;
  }
  else {
    edgeOption = mcPRL::FORBID_VIRTUAL_EDGES;
  }

  is >> nNbrs;
  for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
    is >> coord;
    vNbrCoords.push_back(mcPRL::WeightedCellCoord(coord));
  }
  nbrhd.init(vNbrCoords, edgeOption);
  
  return is;
}

