#include "mcrpl-cellspaceInfo.h"

mcRPL::CellspaceInfo::
CellspaceInfo()
  :_dTypeName("UNKNOWN"),
   _dTypeSize(0),
   _pNoDataVal(NULL),
   _pGeoInfo(NULL),
   _tileSize(0) {}

mcRPL::CellspaceInfo::
CellspaceInfo(const mcRPL::CellspaceInfo &rhs)
  :_dTypeName("UNKNOWN"),
   _dTypeSize(0),
   _pNoDataVal(NULL),
   _pGeoInfo(NULL),
   _tileSize(0) {
  if(!init(rhs._dims, rhs._dTypeName.c_str(), rhs._dTypeSize, rhs._pGeoInfo, rhs._tileSize)) {
    exit(-1);
  }

  memcpy(_pNoDataVal, rhs._pNoDataVal, _dTypeSize);
}

mcRPL::CellspaceInfo::
CellspaceInfo(const mcRPL::SpaceDims &dims,
              const char *aTypeName,
              size_t typeSize,
              const mcRPL::CellspaceGeoinfo *pGeoinfo,
              long tileSize)
  :_dTypeName("UNKNOWN"),
   _dTypeSize(0),
   _pNoDataVal(NULL),
   _pGeoInfo(NULL),
   _tileSize(0) {
  if(!init(dims, aTypeName, typeSize, pGeoinfo, tileSize)) {
    exit(-1);
  }
 
}

mcRPL::CellspaceInfo::
~CellspaceInfo() {
  clear();
}

mcRPL::CellspaceInfo& mcRPL::CellspaceInfo::
operator=(const mcRPL::CellspaceInfo &rhs) {
  if(this != &rhs) {
    if(!init(rhs._dims, rhs._dTypeName.c_str(), rhs._dTypeSize, rhs._pGeoInfo, rhs._tileSize)) {
      exit(-1);
    }

    memcpy(_pNoDataVal, rhs._pNoDataVal, _dTypeSize);
  }

  return *this;
}

bool mcRPL::CellspaceInfo::
operator==(const mcRPL::CellspaceInfo &rhs) {
  return (_dims == rhs._dims &&
          _dTypeName == rhs._dTypeName &&
          _dTypeSize == rhs._dTypeSize &&
          *_pGeoInfo == *(rhs._pGeoInfo) &&
          _tileSize == rhs._tileSize &&
          strcmp ((const char *)_pNoDataVal, (const char *)(rhs._pNoDataVal)) == 0);
}

bool mcRPL::CellspaceInfo::
operator!=(const mcRPL::CellspaceInfo &rhs) {
  return !(operator==(rhs));
}

bool mcRPL::CellspaceInfo::
init(const mcRPL::SpaceDims &dims,
     const char *aTypeName,
     size_t typeSize,
     const mcRPL::CellspaceGeoinfo *pGeoinfo,
     long tileSize) {
  clear();
  if(dims.isNone()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: invalid spatial dimensions (" << dims \
        << ") for the CellspaceInfo" \
        << endl;
    return false;
  }
  else if(typeSize <= 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: invalid data type size (" << typeSize \
        << ") for the CellspaceInfo" \
        << endl;
    return false;
  }
  else if(!initDefVal(_pNoDataVal, string(aTypeName)) ||
          _pNoDataVal == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: NoData value is not properly initialized"
         << endl;
    return false;
  }
		   
  _dims = dims;
  _dTypeName = string(aTypeName);
  _dTypeSize = typeSize; 
  if(pGeoinfo != NULL) {
    _pGeoInfo = new mcRPL::CellspaceGeoinfo(*pGeoinfo);
  }
  setTileSize(tileSize);

  return true;
}

bool mcRPL::CellspaceInfo::
initByGDAL(GDALDataset *pDataset,
           int iBand,
           bool warning) {
  if(pDataset == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL GDAL Dataset" << endl;
    }
    return false;
  }

  int nBands = pDataset->GetRasterCount();
  if(iBand < 1 || iBand > nBands) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: invalid band index (" << iBand \
           << ") in the GDAL Dataset with " << nBands << " bands" << endl;
    }
    return false;
  }

  GDALRasterBand *pBand = pDataset->GetRasterBand(iBand);
  GDALDataType gdalType = pBand->GetRasterDataType();
  string dataTypeName;
  size_t dataTypeSize;
  gdalType2cppType(dataTypeName, dataTypeSize, gdalType);
  //cout<<dataTypeName<<endl;
  if(dataTypeName == "UNKNOWN") {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: unknown data type from the GDAL Dataset" << endl;
    }
    return false;
  }

  long nRows = pDataset->GetRasterYSize();
  long nCols = pDataset->GetRasterXSize();
  if(!init(mcRPL::SpaceDims(nRows, nCols), dataTypeName.c_str(), dataTypeSize)) {
    return false;
  }

  double noDataVal = pBand->GetNoDataValue();
  setNoDataValAs<double>(noDataVal);

  mcRPL::CellspaceGeoinfo geoInfo;
  if(geoInfo.initByGDAL(pDataset, warning)) {
    georeference(geoInfo);
  }

  setTileSize(TILEWIDTH);

  return true;
}
int mcRPL::CellspaceInfo::
cudaTypeCopy()
{
	if(isDataType<bool>()) {
     _cudaType=11;
    }
    else if(isDataType<unsigned char>()) {
       _cudaType=0;
    }
    else if(isDataType<char>()) {
      _cudaType=1;
    }
    else if(isDataType<unsigned short>()) {
      _cudaType=2;     
    }
    else if(isDataType<short>()) {
      _cudaType=3; 
    }
    else if(isDataType<unsigned int>()) {
     _cudaType=4; 
    }
    else if(isDataType<int>()) {
      _cudaType=5 ;
    }
    else if(isDataType<unsigned long>()) {
      _cudaType=6; 
    }
    else if(isDataType<long>()) {
     _cudaType=7; 
    }
    else if(isDataType<float>()) {
     _cudaType=8; 
    }
    else if(isDataType<double>()) {
      _cudaType=9; 
    }
    else if(isDataType<long double>()) {
      _cudaType=10; 
    }
	return _cudaType;
}
bool mcRPL::CellspaceInfo::
initByPGTIOL(PGTIOLDataset *pDataset,
           int iBand,
           bool warning) {
  if(pDataset == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pGTIOL Dataset" << endl;
    }
    return false;
  }

  int nBands = pDataset->GetRasterCount();
  if(iBand < 1 || iBand > nBands) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: invalid band index (" << iBand \
           << ") in the pGTIOL Dataset with " << nBands << " bands" << endl;
    }
    return false;
  }

  pDataset->SetRasterBand(iBand);
  GDALDataType gdalType = pDataset->GetRasterDataType();
  string dataTypeName;
  size_t dataTypeSize;
  
  gdalType2cppType(dataTypeName, dataTypeSize,gdalType);
  
  if(dataTypeName == "UNKNOWN") {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: unknown data type from the pGTIOL Dataset" << endl;
    }
    return false;
  }
  
  long nRows = pDataset->GetRasterYSize();
  long nCols = pDataset->GetRasterXSize();
  if(!init(mcRPL::SpaceDims(nRows, nCols), dataTypeName.c_str(), dataTypeSize)) {
    return false;
  }

  double noDataVal = pDataset->GetNoDataValue();
  setNoDataValAs<double>(noDataVal);

  mcRPL::CellspaceGeoinfo geoInfo;
  if(geoInfo.initByPGTIOL(pDataset, warning)) {
     georeference(geoInfo);
  }

  long tileSize = pDataset->GetTileWidth();
  setTileSize(tileSize);

  return true;
}



void mcRPL::CellspaceInfo::
clear() {
  _dims = mcRPL::SpaceDims();
  _dTypeName.assign("UNKNOWN");
  _dTypeSize = 0;
  
  if(_pNoDataVal != NULL) {
    free(_pNoDataVal);
    _pNoDataVal = NULL;
  }

  if(_pGeoInfo != NULL) {
    delete _pGeoInfo;
    _pGeoInfo = NULL;
  }

  _tileSize = 0;
}

void mcRPL::CellspaceInfo::
add2Buf(vector<char> &buf) {
  if(_pGeoInfo != NULL) {
    _pGeoInfo->add2Buf(buf);
    buf.push_back('y'); // indicates that geoinfo exists
  }
  else {
    buf.push_back('n'); // indicates that geoinfo doesn't exist
  }

  size_t bufPtr = buf.size();
  buf.resize(bufPtr + 2*sizeof(size_t) + _dTypeName.size() + _dTypeSize + sizeof(long));

  memcpy((void *)&(buf[bufPtr]), _pNoDataVal, _dTypeSize);
  bufPtr += _dTypeSize;

  memcpy((void *)&(buf[bufPtr]), &_dTypeSize, sizeof(size_t));
  bufPtr += sizeof(size_t);

  size_t lenTypeName = _dTypeName.size();
  memcpy((void *)&(buf[bufPtr]), &(_dTypeName[0]), lenTypeName);
  bufPtr += lenTypeName;
  
  memcpy((void *)&(buf[bufPtr]), &lenTypeName, sizeof(size_t));
  bufPtr += sizeof(size_t);
  
  memcpy((void *)&(buf[bufPtr]), &_tileSize, sizeof(long));
  
  _dims.add2Buf(buf);
}

bool mcRPL::CellspaceInfo::
initFromBuf(vector<char> &buf) {
  clear();

  size_t lenTypeName;
  char ifGeoref;

  if(!_dims.initFromBuf(buf)) {
    return false;
  }
  if(!_dims.valid()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid spatial dimensions (" << _dims \
         << ")" << endl;
    return false;
  }
  
  int bufPtr = buf.size() - sizeof(long);
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the buffer has insufficient memory to extract the tile size of Cellspace " \
         << endl;
    return false;
  }
  memcpy(&_tileSize, (void *)&(buf[bufPtr]), sizeof(long));

  bufPtr -= sizeof(size_t);
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the buffer has insufficient memory to extract the length of the data type name of a Cellspace" \
         << endl;
    return false;
  }
  memcpy(&lenTypeName, (void *)&(buf[bufPtr]), sizeof(size_t));

  bufPtr -= lenTypeName;
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the buffer has insufficient memory to extract the data type name of a Cellspace" \
         << endl;
    return false;
  }
  _dTypeName.resize(lenTypeName);
  memcpy(&(_dTypeName[0]), (void *)&(buf[bufPtr]), lenTypeName);

  bufPtr -= sizeof(size_t);
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the buffer has insufficient memory to extract the data type size of a Cellspace" \
         << endl;
    return false;
  }
  memcpy(&_dTypeSize, (void *)&(buf[bufPtr]), sizeof(size_t));
  if(_dTypeSize <= 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid data type size (" << _dTypeSize << ")" \
         << endl;
    return false;
  }

  bufPtr -= _dTypeSize;
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the buffer has insufficient memory to extract the NoData value of a Cellspace" \
         << endl;
    return false;
  }
  _pNoDataVal = (void *)calloc(1, _dTypeSize);
  memcpy(_pNoDataVal, (void *)&(buf[bufPtr]), _dTypeSize);

  bufPtr -= 1;
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the buffer has insufficient memory to extract the Georeference flag of a Cellspace" \
         << endl;
    return false;
  }
  ifGeoref = buf[bufPtr];

  buf.erase(buf.begin()+bufPtr, buf.end());

  if(ifGeoref == 'y') {
    CellspaceGeoinfo geoInfo;
    if(!geoInfo.initFromBuf(buf)) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: failed to extract the Georeference information of a Cellspace" \
           << endl;
      return false;
    }
    _pGeoInfo = new mcRPL::CellspaceGeoinfo(geoInfo);
  }
  else if(ifGeoref != 'n') {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid Georeference flag (" << ifGeoref \
         << ")" << endl;
    return false;
  }

  return true;
}

void mcRPL::CellspaceInfo::
georeference(const mcRPL::CellspaceGeoinfo& geoinfo) {
  if(_pGeoInfo != NULL) {
    delete _pGeoInfo;
    _pGeoInfo = NULL;
  }
  _pGeoInfo = new mcRPL::CellspaceGeoinfo(geoinfo);
}

const mcRPL::CellspaceGeoinfo* mcRPL::CellspaceInfo::
georeference() const {
  return _pGeoInfo;
}

bool mcRPL::CellspaceInfo::
isGeoreferenced(bool warning) const {
  bool isGeofed = true;
  if(_pGeoInfo == NULL) {
    isGeofed = false;
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << "Warning: CellspaceInfo is NOT georeferenced yet" << endl;
    }
  }
  return isGeofed;
}

bool mcRPL::CellspaceInfo::
isEmpty(bool warning) const {
  bool empty = false;
  if(_dims.isNone()) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Warning: EMPTY Cellspace" << endl;
    }
    empty = true;
  }
  return empty;
}

const mcRPL::SpaceDims& mcRPL::CellspaceInfo::
dims() const {
  return _dims;
}

long mcRPL::CellspaceInfo::
nRows() const {
  return _dims.nRows();
}

long mcRPL::CellspaceInfo::
nCols() const {
  return _dims.nCols();
}

long mcRPL::CellspaceInfo::
tileSize() const {
  return _tileSize;
}

long mcRPL::CellspaceInfo::
size() const {
  return _dims.size();
}

const char* mcRPL::CellspaceInfo::
dataType() const {
  return _dTypeName.c_str();
}

size_t mcRPL::CellspaceInfo::
dataSize() const {
  return _dTypeSize;
}

void mcRPL::CellspaceInfo::
setTileSize(long tileSize) {
  _tileSize = tileSize;
  if(_tileSize <= 1 || _tileSize >= _dims.nRows() || _tileSize >= _dims.nCols()) {
    _tileSize = TILEWIDTH;
  }
}

long mcRPL::CellspaceInfo::
coord2idx(const mcRPL::CellCoord& coord) const {
  return coord.toIdx(_dims);
}

long mcRPL::CellspaceInfo::
coord2idx(long iRow, long iCol) const {
  return coord2idx(CellCoord(iRow, iCol));
}

mcRPL::CellCoord mcRPL::CellspaceInfo::
idx2coord(long idx) const {
  return mcRPL::CellCoord(idx, _dims);
}

bool mcRPL::CellspaceInfo::
validCoord(const mcRPL::CellCoord& coord,
           bool warning) const {
  bool valid = true;
  if(!coord.valid(_dims)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: coordinates [" << coord \
           << "] out of CellspaceInfo boundary [" \
           << _dims << "]" << endl;
    }
  }
  return valid;
}

bool mcRPL::CellspaceInfo::
validCoord(long iRow, long iCol,
           bool warning) const {
  return validCoord(mcRPL::CellCoord(iRow, iCol), warning);
}

bool mcRPL::CellspaceInfo::
validIdx(long idx,
         bool warning) const {
  bool valid = true;
  if(!_dims.validIdx(idx)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: index [" << idx \
           << "] out of CellspaceInfo size [" \
           << size() << "]" \
           << endl;
    }
  }
  return valid;
}

bool mcRPL::CellspaceInfo::
validBR(const mcRPL::CoordBR& rectangle,
        bool warning) const {
  bool valid = true;
  if(!rectangle.valid(_dims)) {
    valid = false;
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: rectangle [" << rectangle \
           << "] out of CellspaceInfo boundary [" << _dims  << "]" << endl;
    }
  }
  return valid;
}

mcRPL::GeoCoord mcRPL::CellspaceInfo::
cellCoord2geoCoord(const mcRPL::CellCoord &cCoord) const {
  mcRPL::GeoCoord gCoord;
  if(!isEmpty(true) &&
     isGeoreferenced(true) &&
     validCoord(cCoord, true)) {
    gCoord = _pGeoInfo->cellCoord2geoCoord(cCoord);
  }
  return gCoord;
}

mcRPL::GeoCoord mcRPL::CellspaceInfo::
cellCoord2geoCoord(long iRow, long iCol) const {
  mcRPL::GeoCoord gCoord;
  if(!isEmpty(true) &&
     isGeoreferenced(true) &&
     validCoord(iRow, iCol, true)) {
    gCoord = _pGeoInfo->cellCoord2geoCoord(iRow, iCol);
  }
  return gCoord;
}

mcRPL::CellCoord mcRPL::CellspaceInfo::
geoCoord2cellCoord(const mcRPL::GeoCoord &gCoord) const {
  mcRPL::CellCoord cCoord;
  if(!isEmpty(true) && isGeoreferenced(true)) {
    cCoord = _pGeoInfo->geoCoord2cellCoord(gCoord);
    if(!validCoord(cCoord, true)) {
      cCoord = CellCoord();
    }
  }
  return cCoord;
}

mcRPL::CellCoord mcRPL::CellspaceInfo::
geoCoord2cellCoord(double x, double y) const {
  mcRPL::CellCoord cCoord;
  if(!isEmpty(true) && isGeoreferenced(true)) {
    cCoord = _pGeoInfo->geoCoord2cellCoord(x, y);
    if(!validCoord(cCoord, true)) {
      cCoord = CellCoord();
    }
  }
  return cCoord;
}

bool mcRPL::CellspaceInfo::
calcWorkBR(mcRPL::CoordBR *pWorkBR,
           const mcRPL::Neighborhood *pNbrhd,
           bool warning) const {
  bool done = true;
  if(pWorkBR == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the work CoordBR" << endl;
    }
    done = false;
  }
  else if(pNbrhd == NULL || pNbrhd->isEmpty()) {
    pWorkBR->nwCorner(0, 0);
    pWorkBR->seCorner(_dims.nRows()-1, _dims.nCols()-1);
  }
  else {
    done = pNbrhd->calcWorkBR(*pWorkBR, _dims);
  }
  return done;
}

bool mcRPL::CellspaceInfo::
validNbrhd(const mcRPL::Neighborhood &nbrhd,
           const mcRPL::CellCoord &ctrCoord,
           bool warning) const {
  if(nbrhd.isEmpty()) {
    if(warning) {
      cerr << __FILE__ << " function: " << __FUNCTION__ \
           << " Error: empty Neighborhood" << endl;
    }
    return false;
  }

  if(nbrhd.edgeOption() == mcRPL::FORBID_VIRTUAL_EDGES) {
    if(ctrCoord.iRow() < -(nbrhd.minIRow()) ||
       ctrCoord.iRow() > nRows()-1-nbrhd.maxIRow() ||
       ctrCoord.iCol() < -(nbrhd.minICol()) ||
       ctrCoord.iCol() > nCols()-1-nbrhd.maxICol()) {
      if(warning) {
        cerr << __FILE__ << " function: " << __FUNCTION__ \
            << " Error: invalid center coordinate[" \
            << ctrCoord << "] for a Neighborhood in CellspaceInfo (" \
            << _dims << ")"<< endl;
      }
      return false;
    }
  }
  else {
    if(!ctrCoord.valid(_dims)) {
      if(warning) {
        cerr << __FILE__ << " function: " << __FUNCTION__ \
             << " Error: invalid center coordinate[" \
             << ctrCoord << "] for a Neighborhood in CellspaceInfo (" \
             << _dims << ")"<< endl;
      }
      return false;
    }
  }

  return true;
}

bool mcRPL::CellspaceInfo::
validNbrhd(const mcRPL::Neighborhood &nbrhd,
           long ctrRow, long ctrCol,
           bool warning) const {
  return validNbrhd(nbrhd, mcRPL::CellCoord(ctrRow, ctrCol), warning);
}

bool mcRPL::CellspaceInfo::
validNbrhd(const mcRPL::Neighborhood &nbrhd,
           long idx,
           bool warning) const {
  return validNbrhd(nbrhd, mcRPL::CellCoord(idx, _dims), warning);
}
