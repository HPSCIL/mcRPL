#include "mcrpl-neighborhood.h"

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
bool mcRPL::Neighborhood::
_checkNbrCoords(const vector<mcRPL::CellCoord> &vNbrCoords) const {
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

bool mcRPL::Neighborhood::
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
mcRPL::Neighborhood::
Neighborhood(const char *aName,
             mcRPL::EdgeOption edgeOption,
             double virtualEdgeVal) {
  if(aName != NULL) {
    _name.assign(aName);
  }
  else {
    _name.assign("Untitled_Neighborhood");
  }
  _edgeOption = edgeOption;
  _pVirtualEdgeVal = NULL;

  if(edgeOption == mcRPL::CUSTOM_VIRTUAL_EDGES) {
    _pVirtualEdgeVal = new double;
    memcpy(_pVirtualEdgeVal, &virtualEdgeVal, sizeof(double));
  }
}

mcRPL::Neighborhood::
Neighborhood(const mcRPL::Neighborhood &rhs) {
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

mcRPL::Neighborhood::
~Neighborhood() {
  if(_pVirtualEdgeVal != NULL) {
    delete _pVirtualEdgeVal;
  }
}
void mcRPL::Neighborhood::addnbrForGpu()
{
	 long long  nbrsize=this->size();
	 vector<mcRPL::WeightedCellCoord>coords =this->getInnerNbr(); 
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
bool mcRPL::Neighborhood::
init(const vector<mcRPL::CellCoord> &vNbrCoords,
     double weight,
     mcRPL::EdgeOption edgeOption,
     double virtualEdgeVal) {
  if(!_checkNbrCoords(vNbrCoords)) {
    return false;
  }

  _vNbrs.clear();
  _mNbrIDMap.clear();
  _mNbrIDMap.resize(8);

  _edgeOption = edgeOption;
  _pVirtualEdgeVal = NULL;
  if(edgeOption == mcRPL::CUSTOM_VIRTUAL_EDGES) {
    _pVirtualEdgeVal = new double;
    memcpy(_pVirtualEdgeVal, &virtualEdgeVal, sizeof(double));
  }

  CellCoord nwCorner, seCorner;
  for(int iNbr = 0; iNbr < vNbrCoords.size(); iNbr++) {
    vector<mcRPL::WeightedCellCoord>::iterator nbrItr = _vNbrs.begin();
    while(nbrItr != _vNbrs.end()) {
      mcRPL::CellCoord coord(nbrItr->iRow(), nbrItr->iCol());
      if(coord == vNbrCoords[iNbr]) {
        cerr << __FILE__ << " " << __FUNCTION__ \
             << " Error: a duplicated coordinate [" \
             << vNbrCoords[iNbr] \
             << "] has been found" << endl;
        return false;
      }
      nbrItr++;
    }

    _vNbrs.push_back(mcRPL::WeightedCellCoord(vNbrCoords[iNbr].iRow(), vNbrCoords[iNbr].iCol(), weight));

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
        _mNbrIDMap[mcRPL::NORTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcRPL::NORTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcRPL::NORTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iRow() > 0) {
      if(vNbrCoords[iNbr].iCol() < 0) {
        _mNbrIDMap[mcRPL::SOUTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcRPL::SOUTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcRPL::SOUTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() < 0) {
      _mNbrIDMap[mcRPL::WEST_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() > 0) {
      _mNbrIDMap[mcRPL::EAST_DIR].push_back(iNbr);
    }
  } // End of iNbr loop

  _MBR.nwCorner(nwCorner);
  _MBR.seCorner(seCorner);
  addnbrForGpu();
  return true;
}

bool mcRPL::Neighborhood::
init(const vector<mcRPL::WeightedCellCoord> &vNbrCoords,
     mcRPL::EdgeOption edgeOption,
     double virtualEdgeVal) {
  if(!_checkNbrCoords((const vector<mcRPL::CellCoord> &)vNbrCoords)) {
    return false;
  }
  
  _vNbrs.clear();
  _mNbrIDMap.clear();
  _mNbrIDMap.resize(8);
  
  _edgeOption = edgeOption;
  _pVirtualEdgeVal = NULL;
  if(edgeOption == mcRPL::CUSTOM_VIRTUAL_EDGES) {
    _pVirtualEdgeVal = new double;
    memcpy(_pVirtualEdgeVal, &virtualEdgeVal, sizeof(double));
  }

  mcRPL::CellCoord nwCorner, seCorner;
  for(int iNbr = 0; iNbr < vNbrCoords.size(); iNbr++) {
    if(std::find(_vNbrs.begin(), _vNbrs.end(), vNbrCoords[iNbr]) != _vNbrs.end()) {
      cerr << __FILE__ << " " << __FUNCTION__ \
      << " Error: a duplicated coordinate [" \
      << vNbrCoords[iNbr] \
      << "] has been found" << endl;
      return false;
    }
    
    _vNbrs.push_back(mcRPL::WeightedCellCoord(vNbrCoords[iNbr]));
    
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
        _mNbrIDMap[mcRPL::NORTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcRPL::NORTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcRPL::NORTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iRow() > 0) {
      if(vNbrCoords[iNbr].iCol() < 0) {
        _mNbrIDMap[mcRPL::SOUTHWEST_DIR].push_back(iNbr);
      }
      else if(vNbrCoords[iNbr].iCol() > 0) {
        _mNbrIDMap[mcRPL::SOUTHEAST_DIR].push_back(iNbr);
      }
      _mNbrIDMap[mcRPL::SOUTH_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() < 0) {
      _mNbrIDMap[mcRPL::WEST_DIR].push_back(iNbr);
    }
    if(vNbrCoords[iNbr].iCol() > 0) {
      _mNbrIDMap[mcRPL::EAST_DIR].push_back(iNbr);
    }
  } // End of iNbr loop
  
  _MBR.nwCorner(nwCorner);
  _MBR.seCorner(seCorner);
  addnbrForGpu();
  return true;
}

bool mcRPL::Neighborhood::
initSingleCell(double weight) {
  vector<mcRPL::CellCoord> vNbrCoords;
  vNbrCoords.push_back(mcRPL::CellCoord(0, 0));
  return init(vNbrCoords, weight, mcRPL::FORBID_VIRTUAL_EDGES);
}

bool mcRPL::Neighborhood::
initVonNeumann(double weight,
               mcRPL::EdgeOption edgeOption,
               double virtualEdgeVal) {
  vector<mcRPL::CellCoord> vNbrCoords;
  vNbrCoords.push_back(mcRPL::CellCoord(0, 0));
  vNbrCoords.push_back(mcRPL::CellCoord(-1, 0));
  vNbrCoords.push_back(mcRPL::CellCoord(0, 1));
  vNbrCoords.push_back(mcRPL::CellCoord(1, 0));
  vNbrCoords.push_back(mcRPL::CellCoord(0, -1));
  return init(vNbrCoords, weight, edgeOption, virtualEdgeVal);
}

bool mcRPL::Neighborhood::
initMoore(long nEdgeLength,
          double weight,
          mcRPL::EdgeOption edgeOption,
          double virtualEdgeVal) {
  if(nEdgeLength <= 0 || nEdgeLength % 2 == 0) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid edge length (" << nEdgeLength \
         << "). A Moore neighborhood's edge length must be a positive odd number" << endl;
    return false;
  }

  vector<mcRPL::CellCoord> vNbrCoords;
  vNbrCoords.push_back(mcRPL::CellCoord(0, 0));
  for(long iRow = -(nEdgeLength/2); iRow <= nEdgeLength/2; iRow++) {
    for(long iCol = -(nEdgeLength/2); iCol <= nEdgeLength/2; iCol++) {
      if(iRow == 0 && iCol == 0) {
        continue;
      }
      vNbrCoords.push_back(mcRPL::CellCoord(iRow, iCol));
    }
  }

  return init(vNbrCoords, weight, edgeOption, virtualEdgeVal);
}

void mcRPL::Neighborhood::
clear() {
  _name.clear();
  _vNbrs.clear();
  _mNbrIDMap.clear();
  _MBR = CoordBR();
  _edgeOption = mcRPL::FORBID_VIRTUAL_EDGES;
  if(_pVirtualEdgeVal != NULL) {
    delete _pVirtualEdgeVal;
    _pVirtualEdgeVal = NULL;
  }
}

mcRPL::WeightedCellCoord& mcRPL::Neighborhood::
operator[](int iNbr) {
  return _vNbrs.at(iNbr);
}

const mcRPL::WeightedCellCoord& mcRPL::Neighborhood::
operator[](int iNbr) const {
  return _vNbrs.at(iNbr);
}

mcRPL::Neighborhood& mcRPL::Neighborhood::
operator=(const mcRPL::Neighborhood &rhs) {
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

const char* mcRPL::Neighborhood::
name() const {
  return _name.c_str();
}

void mcRPL::Neighborhood::
name(const char *aName) {
  if(aName != NULL) {
    _name.assign(aName);
  }
}

bool mcRPL::Neighborhood::
isEmpty() const{
  return _vNbrs.empty();
}

int mcRPL::Neighborhood::
size() const{
  return _vNbrs.size();  
}

bool mcRPL::Neighborhood::
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
 vector<double> mcRPL::Neighborhood::cuWeights()
 {
	 return cuNbrWeights;
 }
  vector<int>mcRPL::Neighborhood::cuCoords()
  {
	  return cuNbrCoords;
  }
long mcRPL::Neighborhood::
minIRow() const {
  return _MBR.minIRow();
}

long mcRPL::Neighborhood::
minICol() const {
  return _MBR.minICol();
}

long mcRPL::Neighborhood::
maxIRow() const {
  return _MBR.maxIRow();
}

long mcRPL::Neighborhood::
maxICol() const {
  return _MBR.maxICol();
}

long mcRPL::Neighborhood::
nRows() const {
  return _MBR.nRows();
}

long mcRPL::Neighborhood::
nCols() const {
  return _MBR.nCols();
}

const mcRPL::CoordBR* mcRPL::Neighborhood::
getMBR() const {
  return &(_MBR);
}

mcRPL::EdgeOption mcRPL::Neighborhood::
edgeOption() const {
  return _edgeOption;
}

const double* mcRPL::Neighborhood::
virtualEdgeVal() const {
  return _pVirtualEdgeVal;
}

bool mcRPL::Neighborhood::
virtualEdgeVal(const double veVal) {
  if(_edgeOption != mcRPL::CUSTOM_VIRTUAL_EDGES) {
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

bool mcRPL::Neighborhood::
hasNbrs(mcRPL::MeshDir dir) const {
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

const mcRPL::IntVect* mcRPL::Neighborhood::
nbrIDs(mcRPL::MeshDir dir) const {
  return &(_mNbrIDMap.at(dir));
}

bool mcRPL::Neighborhood::
calcWorkBR(mcRPL::CoordBR &workBR,
           const mcRPL::SpaceDims &dims) const {
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

void mcRPL::Neighborhood::
add2Buf(mcRPL::CharVect &vBuf) const {
  size_t optionSize = sizeof(mcRPL::EdgeOption);
  size_t coordSize = sizeof(mcRPL::WeightedCellCoord);
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

bool mcRPL::Neighborhood::
fromBuf(const mcRPL::CharVect &vBuf,
        int &iChar) {
  size_t optionSize = sizeof(mcRPL::EdgeOption);
  size_t coordSize = sizeof(mcRPL::WeightedCellCoord);
  size_t intSize = sizeof(int);

  if(vBuf.size() - iChar < optionSize) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: buffer is too small to extract a Neighborhood" << endl;
    return false;
  }
  mcRPL::EdgeOption edgeOption;
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
  mcRPL::WeightedCellCoord coord;
  vector<mcRPL::WeightedCellCoord> vNbrCoords;
  for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
    memcpy(&coord, &(vBuf[iChar]), coordSize);
    vNbrCoords.push_back(WeightedCellCoord(coord));
    iChar += coordSize;
  }
  return init(vNbrCoords, edgeOption);
}

ostream& mcRPL::
operator<<(ostream &os,
           const mcRPL::Neighborhood &nbrhd) {
  os << nbrhd.edgeOption() << endl;
  int nNbrs = nbrhd.size();
  os << nNbrs << endl;
  for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
    os << nbrhd[iNbr];
    os << endl;
  }
  return os;
}

istream& mcRPL::
operator>>(istream &is,
           mcRPL::Neighborhood &nbrhd) {
  mcRPL::EdgeOption edgeOption;
  vector<mcRPL::WeightedCellCoord> vNbrCoords;
  mcRPL::WeightedCellCoord coord;
  int iEdgeOption, nNbrs;
  
  is >> iEdgeOption;
  if(iEdgeOption >= mcRPL::FORBID_VIRTUAL_EDGES &&
     iEdgeOption <= mcRPL::CUSTOM_VIRTUAL_EDGES) {
    edgeOption = (mcRPL::EdgeOption)iEdgeOption;
  }
  else {
    edgeOption = mcRPL::FORBID_VIRTUAL_EDGES;
  }

  is >> nNbrs;
  for(int iNbr = 0; iNbr < nNbrs; iNbr++) {
    is >> coord;
    vNbrCoords.push_back(mcRPL::WeightedCellCoord(coord));
  }
  nbrhd.init(vNbrCoords, edgeOption);
  
  return is;
}

