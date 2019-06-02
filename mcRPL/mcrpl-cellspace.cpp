#include "mcrpl-cellspace.h"
#include "mcrpl-CuPRL.h"
/****************************************************
*                 Protected Methods                 *
****************************************************/
bool mcRPL::Cellspace::
_checkGDALSettings(GDALDataset *pDataset,
                   int iBand,
                   const mcRPL::CellCoord *pGDALOff,
                   const mcRPL::CoordBR *pWorkBR,
                   bool warning) const {
  if(isEmpty(warning)) {
    return false;
  }

  if(pDataset == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL GDAL Dataset to read data from" << endl;
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

  GDALDataType cellDataType, gdalDataType;
  cppType2gdalType(cellDataType, string(_pInfo->dataType()));
  gdalDataType = pDataset->GetRasterBand(iBand)->GetRasterDataType();

  if(cellDataType != gdalDataType) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: Cellspace's data type (" << _pInfo->dataType() \
           << ") does NOT match GDAL data type (" \
           << GDALGetDataTypeName(gdalDataType) << ")" << endl;
    }
    return false;
  }

  int nGDALRows = pDataset->GetRasterYSize();
  int nGDALCols = pDataset->GetRasterXSize();
  long iGDALRowOff = 0, iGDALColOff = 0;
  if(pGDALOff != 0) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
  }
  if(iGDALRowOff < 0 || iGDALRowOff >= nGDALRows ||
     iGDALColOff < 0 || iGDALColOff >= nGDALCols) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: invalid row-col offset ("\
           << iGDALRowOff << ", " << iGDALColOff << ") in a GDAL Dataset with dimensions (" \
           << nGDALRows << ", " << nGDALCols << ")" << endl;
    }
    return false;
  }

  long nDataRows = _pInfo->nRows();
  long nDataCols = _pInfo->nCols();
  if(pWorkBR != NULL) {
    if(!pWorkBR->valid(_pInfo->dims())) {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: invalid work boundary (" << *pWorkBR \
             << ") in Cellspace with dimensions (" << _pInfo->dims() << ")" << endl;
      }
      return false;
    }
    nDataRows = pWorkBR->nRows();
    nDataCols = pWorkBR->nCols();
  }
  if(iGDALRowOff+nDataRows < 0 || iGDALRowOff+nDataRows > nGDALRows ||
     iGDALColOff+nDataCols < 0 || iGDALColOff+nDataCols > nGDALCols) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the Cellspace with work dimensions (" \
           << nDataRows << ", " << nDataCols << ") and row-col offset ("\
           << iGDALRowOff << ", " << iGDALColOff << ") exceeds the extent of a GDAL Dataset with dimensions (" \
           << nGDALRows << ", " << nGDALCols << ")" << endl;
    }
    return false;
  }
  return true;
}


/*-----------------PGTIOL--------------------------------*/
bool mcRPL::Cellspace::
_checkPGTIOLSettings(PGTIOLDataset *pDataset,
                     int iBand,
                     const mcRPL::CellCoord *pGDALOff ,
                     const mcRPL::CoordBR *pWorkBR ,
                     bool warning ) const {
  

  if(isEmpty(warning)) {
    return false;
  }

  if(pDataset == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL GDAL Dataset to read data from" << endl;
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

  GDALDataType cellDataType;
  cppType2gdalType(cellDataType, string(_pInfo->dataType()));
  int rnk =0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rnk);
  GDALDataType gdalDataType = pDataset->GetRasterDataType();
  
  if(cellDataType != gdalDataType) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: Cellspace's data type (" << _pInfo->dataType() \
           << ") does NOT match GDAL data type ("<< endl;
    }
    return false;
  }

  int nGDALRows = pDataset->GetRasterYSize();
  int nGDALCols = pDataset->GetRasterXSize();
  long iGDALRowOff = 0, iGDALColOff = 0;
  if(pGDALOff != 0) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
  }
  if(iGDALRowOff < 0 || iGDALRowOff >= nGDALRows ||
     iGDALColOff < 0 || iGDALColOff >= nGDALCols) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: invalid row-col offset ("\
           << iGDALRowOff << ", " << iGDALColOff << ") in a GDAL Dataset with dimensions (" \
           << nGDALRows << ", " << nGDALCols << ")" << endl;
    }
    return false;
  }

  long nDataRows = _pInfo->nRows();
  long nDataCols = _pInfo->nCols();
  if(pWorkBR != NULL) {
    if(!pWorkBR->valid(_pInfo->dims())) {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: invalid work boundary (" << *pWorkBR \
             << ") in Cellspace with dimensions (" << _pInfo->dims() << ")" << endl;
      }
      return false;
    }
    nDataRows = pWorkBR->nRows();
    nDataCols = pWorkBR->nCols();
  }
  if(iGDALRowOff+nDataRows < 0 || iGDALRowOff+nDataRows > nGDALRows ||
     iGDALColOff+nDataCols < 0 || iGDALColOff+nDataCols > nGDALCols) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the Cellspace with work dimensions (" \
           << nDataRows << ", " << nDataCols << ") and row-col offset ("\
           << iGDALRowOff << ", " << iGDALColOff << ") exceeds the extent of a GDAL Dataset with dimensions (" \
           << nGDALRows << ", " << nGDALCols << ")" << endl;
    }
    return false;
  }
  return true;
}

/****************************************************
*                 Public Methods                    *
****************************************************/
mcRPL::Cellspace::
Cellspace()
  :_pInfo(NULL),
   _pData(NULL),
   _dpData(NULL),                                      //设备数据指针
   _trackUpdt(false){}

mcRPL::Cellspace::
Cellspace(const mcRPL::Cellspace &rhs)
  :_pInfo(rhs._pInfo),
   _pData(NULL),
   _dpData(NULL),
   _trackUpdt(false){
  if(rhs._pData) {
    if(!initMem()) {
      exit(-1);
    }
    memcpy(_pData, rhs._pData, (_pInfo->dataSize())*(_pInfo->size()));
  }
}

mcRPL::Cellspace::
Cellspace(mcRPL::CellspaceInfo *pInfo,
          bool ifInitMem)
  :_pInfo(pInfo),
   _pData(NULL),
   _dpData(NULL),
   _trackUpdt(false) {
  if(ifInitMem) {
    if(!initMem()) {
      exit(-1);
    }
  }
}

mcRPL::Cellspace::
~Cellspace() {
  clear();
}

bool mcRPL::Cellspace::
initMem() {
  if(_pData) {
    free(_pData);
  }
  _pData = NULL;

  if(_pInfo == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: NULL pointer to the CellspaceInfo to initialize memory for the Cellspace" \
         << endl;
    return false;
  }
  if(_pInfo->dims().isNone()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: Invalid spatial dimensions (" \
         << _pInfo->dims() << ") for the Cellspace" \
         << endl;
    return false;
  }
  if(_pInfo->dataSize() <= 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid data type size (" \
         << _pInfo->dataSize() << ") for the Cellspace" \
         << endl;
    return false;
  }

  _pData = (void *)calloc(_pInfo->size(), _pInfo->dataSize());
  if(_pData == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: failed to allocate memory for the Cellspace" \
         << endl;
    return false;
  }

  return true;
}

bool mcRPL::Cellspace::
init(mcRPL::CellspaceInfo *pInfo,
     bool ifInitMem) {
  _pInfo = pInfo;
  if(ifInitMem) {
    if(!initMem()) {
      return false;
    }
  }
  return true;
}
void mcRPL::Cellspace::
transDataToGPU()
{
	long nWidth=info()->dims().nCols();
	long nHeight=info()->dims().nRows();
	checkCudaErrors(cudaMalloc((void**)&_dpData,nHeight*nWidth*info()->dataSize()));
	checkCudaErrors(cudaMemcpy(_dpData,_pData,nHeight*nWidth*info()->dataSize(),cudaMemcpyHostToDevice));
}
void mcRPL::Cellspace::
getGPUMalloc()
{
	long nWidth=info()->dims().nCols();
	long nHeight=info()->dims().nRows();
	checkCudaErrors(cudaMalloc((void**)&_dpData,nHeight*nWidth*info()->dataSize()));
}
void *mcRPL::Cellspace::
getGPUData()
{
	return _dpData;
}
void mcRPL::Cellspace::
deleteGPUData()
{
	if(_dpData) {
	cudaFree(_dpData);
	}
	_dpData=NULL;
}
bool mcRPL::Cellspace::
	createrandForCells(long ncols,long nrows)
{
	 bool done = false;
  if(_pInfo->isDataType<bool>()) {
	  done = createrandForCellsAs<bool>(ncols, nrows);
  }
  else if(_pInfo->isDataType<unsigned char>()) {
   done = createrandForCellsAs<unsigned char>(ncols, nrows);
  }
  else if(_pInfo->isDataType<char>()) {                                                                                    
    done = createrandForCellsAs<char>(ncols, nrows);
  }
  else if(_pInfo->isDataType<unsigned short>()) {
    done = createrandForCellsAs<unsigned short>(ncols, nrows);
  }
  else if(_pInfo->isDataType<short>()) {
   done = createrandForCellsAs<short>(ncols, nrows);
  }
  else if(_pInfo->isDataType<unsigned int>()) {
    done = createrandForCellsAs<unsigned int>(ncols, nrows);
  }
  else if(_pInfo->isDataType<int>()) {
   done = createrandForCellsAs<int>(ncols, nrows);
  }
  else if(_pInfo->isDataType<unsigned long>()) {
    done = createrandForCellsAs<unsigned long>(ncols, nrows);
  }
  else if(_pInfo->isDataType<long>()) {
   done = createrandForCellsAs<long>(ncols, nrows);
  }
  else if(_pInfo->isDataType<float>()) {
    done = createrandForCellsAs<float>(ncols, nrows);
  }
  else if(_pInfo->isDataType<double>()) {
   done = createrandForCellsAs<double>(ncols, nrows);
  }
  else if(_pInfo->isDataType<long double>()) 
  {
    done = createrandForCellsAs<long double>(ncols, nrows);
  }
  return done;
}
bool mcRPL::Cellspace::
brupdateCell(const mcRPL::CoordBR &br,void *output)
{
	 bool done = false;
  if(_pInfo->isDataType<bool>()) {
    done = brupdateCellAs<bool>(br, output);
  }
  else if(_pInfo->isDataType<unsigned char>()) {
   done = brupdateCellAs<unsigned char>(br, output);
  }
  else if(_pInfo->isDataType<char>()) {                                                                                    
    done = brupdateCellAs<char>(br, output);
  }
  else if(_pInfo->isDataType<unsigned short>()) {
    done = brupdateCellAs<unsigned short>(br, output);
  }
  else if(_pInfo->isDataType<short>()) {
   done = brupdateCellAs<short>(br, output);
  }
  else if(_pInfo->isDataType<unsigned int>()) {
    done = brupdateCellAs<unsigned int>(br, output);
  }
  else if(_pInfo->isDataType<int>()) {
   done = brupdateCellAs<int>(br, output);
  }
  else if(_pInfo->isDataType<unsigned long>()) {
    done = brupdateCellAs<unsigned long>(br, output);
  }
  else if(_pInfo->isDataType<long>()) {
   done = brupdateCellAs<long>(br, output);
  }
  else if(_pInfo->isDataType<float>()) {
    done = brupdateCellAs<float>(br, output);
  }
  else if(_pInfo->isDataType<double>()) {
   done = brupdateCellAs<double>(br, output);
  }
  else if(_pInfo->isDataType<long double>()) 
  {
    done = brupdateCellAs<long double>(br, output);
  }
  return done;
}
void mcRPL::Cellspace::
clear() {
  _pInfo = NULL;

  if(_pData) {
    free(_pData);
  }
  _pData = NULL;

  clearUpdatedIdxs();
}
      
mcRPL::Cellspace& mcRPL::Cellspace::
operator=(const mcRPL::Cellspace& rhs) {
  if(this != &rhs) {
    clear();
    _pInfo = rhs._pInfo;

    if(rhs._pData) {
      if(!initMem()) {
        exit(-1);
      }
      memcpy(_pData, rhs._pData, (_pInfo->dataSize())*(_pInfo->size()));
    }
  }
  return *this;
}

const mcRPL::CellspaceInfo* mcRPL::Cellspace::
info() const {
  return _pInfo;
}

mcRPL::CellspaceInfo* mcRPL::Cellspace::
info() {
  return _pInfo;
}

void mcRPL::Cellspace::
info(mcRPL::CellspaceInfo *pInfo) {
  _pInfo = pInfo;
}

bool mcRPL::Cellspace::
isEmpty(bool warning) const {
  bool empty = false;
  if(_pData == NULL ||
     _pInfo == NULL ||
     _pInfo->dims().isNone()) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Warning: EMPTY Cellspace" << endl;
    }
    empty = true;
  }
  return empty;
}

void* mcRPL::Cellspace::
at(long idx,
   bool warning) {
  void *pData = NULL;
  if(!isEmpty(warning) &&
     _pInfo->validIdx(idx, warning)) {
    pData = (char *)_pData + idx*_pInfo->dataSize();
  }
  return pData;
}

const void* mcRPL::Cellspace::
at(long idx,
   bool warning) const {
  void *pData = NULL;
  if(!isEmpty(warning) &&
      _pInfo->validIdx(idx, warning)) {
    pData = (char *)_pData + idx*_pInfo->dataSize();
  }
  return pData;
}

void mcRPL::Cellspace::
setUpdateTracking(bool toTrack) {
  _trackUpdt = toTrack;
}

const mcRPL::LongVect& mcRPL::Cellspace::
getUpdatedIdxs() const {
  return _vUpdtIdxs;
}

void mcRPL::Cellspace::
clearUpdatedIdxs() {
  //_trackUpdt = false;
  _vUpdtIdxs.clear();
}


/*----------------GDAL--------------*/
bool mcRPL::Cellspace::
createGdalDS(GDALDataset *&pDataset,
             const char *aFileName,
             GDALDriver *pGdalDriver,
             char **aGdalOptions) const {
  if(_pInfo == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: NULL CellspaceInfo, unable to create a GDAL dataset" \
        << endl;
    return false;
  }
  if(_pInfo->isEmpty(true)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create a GDAL dataset with an empty CellspaceInfo" \
        << endl;
    return false;
  }
  if(aFileName == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create a GDAL dataset using a NULL file name" \
        << endl;
    return false;
  }
  if(pGdalDriver == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create a GDAL dataset using a NULL GDAL driver" \
        << endl;
    return false;
  }

  GDALDataType gdalDataType;
  mcRPL::cppType2gdalType(gdalDataType, string(_pInfo->dataType()));
  if(gdalDataType == GDT_Unknown) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: unable to determine appropriate data type for the GDAL dataset" \
         << endl;
    return false;
  }

  pDataset = pGdalDriver->Create(aFileName,
                                 _pInfo->nCols(), _pInfo->nRows(), 1,
                                 gdalDataType, aGdalOptions);
  if(pDataset == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create GDAL dataset (" << aFileName << ")" << endl;
    return false;
  }

  pDataset->GetRasterBand(1)->SetNoDataValue(_pInfo->getNoDataValAs<double>());

  if(_pInfo->isGeoreferenced(false)) {
    pDataset->SetProjection(_pInfo->georeference()->projection().c_str());
    double aGeoTransform[6];
    _pInfo->georeference()->geoTransform(aGeoTransform);
    pDataset->SetGeoTransform(aGeoTransform);
  }
  return true;
}

bool mcRPL::Cellspace::
createGdalDS(GDALDataset *&pDataset,
             const char *aFileName,
             const char *aGdalFormat,
             char **aGdalOptions) const {
  if(aGdalFormat == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: unable to create a GDAL dataset using a NULL GDAL format" \
         << endl;
    return false;
  }

  GDALDriver *pGdalDriver = GetGDALDriverManager()->GetDriverByName(aGdalFormat);
  if(pGdalDriver == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: failed to load the GDAL Driver for (" \
         << aGdalFormat << ")" << endl;
    return false;
  }

  return createGdalDS(pDataset, aFileName, pGdalDriver, aGdalOptions);
}
bool mcRPL::Cellspace::
creatRandData(const mcRPL::CellCoord *pGDALOff,
             bool warning)
{
	 long iGDALRowOff = 0, iGDALColOff = 0;
	 if(pGDALOff != NULL) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
  }
   long ncols=_pInfo->nCols();
    long nrows=_pInfo->nRows();
	return createrandForCells(ncols,nrows);
}
bool mcRPL::Cellspace::
readGDALData(GDALDataset *pDataset,
             int iBand,
             const mcRPL::CellCoord *pGDALOff,
             bool warning) {
  if(!_checkGDALSettings(pDataset, iBand, pGDALOff, NULL, warning)) {
    return false;
  }
  GDALRasterBand *pBand = pDataset->GetRasterBand(iBand);
  GDALDataType inputType;
  cppType2gdalType(inputType, string(_pInfo->dataType()));
  long iGDALRowOff = 0, iGDALColOff = 0;
  if(pGDALOff != NULL) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
  }
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
  if(pBand->RasterIO(GF_Read, iGDALColOff, iGDALRowOff, _pInfo->nCols(), _pInfo->nRows(),
                     _pData, _pInfo->nCols(), _pInfo->nRows(), inputType, 0, 0) == CE_Failure) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: failed to read Cellspace from GDAL Dataset" << endl;
    }
    return false;
  }
    
  return true;
}
bool mcRPL::Cellspace::
writeGDALData(GDALDataset *pDataset,
              int iBand,
              const mcRPL::CellCoord *pGDALOff,
              const mcRPL::CoordBR *pWorkBR,
              bool warning) {
  if(!_checkGDALSettings(pDataset, iBand, pGDALOff, pWorkBR, warning)) {
    return false;
  }
  GDALRasterBand *pBand = pDataset->GetRasterBand(iBand);
  GDALDataType outputType;
  cppType2gdalType(outputType,string(_pInfo->dataType()));
  size_t dataSize = _pInfo->dataSize();

  long iGDALRowOff = 0, iGDALColOff = 0;
  if(pGDALOff != NULL) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
    if(pWorkBR != NULL) {
      iGDALRowOff = iGDALRowOff + pWorkBR->minIRow();
      iGDALColOff = iGDALColOff + pWorkBR->minICol();
    }
  }

  long nDataRows = _pInfo->nRows();
  long nDataCols = _pInfo->nCols();
  if(pWorkBR != NULL) {
    nDataRows = pWorkBR->nRows();
    nDataCols = pWorkBR->nCols();
  }

  long lineSpace =   dataSize * _pInfo->nCols();
  /*int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  cout<<"rank"<<rank<<"iGDALColOff"<<iGDALColOff
      <<"iGDALRowOff"<<iGDALRowOff
	  <<"nDataCols"<<nDataCols
	  <<"nDataRows"<<nDataRows
      <<"nDataCols"<<nDataCols<<"outputType"<<outputType<<"linespace"<<lineSpace<<endl;  */



  void *pData = _pData;
  if(pWorkBR != NULL) {
    pData = (char *)_pData + pWorkBR->nwCorner().toIdx(_pInfo->dims())*dataSize;
  }
 
  if(pBand->RasterIO(GF_Write, iGDALColOff, iGDALRowOff, nDataCols, nDataRows,
                     pData, nDataCols, nDataRows, outputType, 0, lineSpace) == CE_Failure) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: failed to write Cellspace to GDAL Dataset" << endl;
    }
    return false;
  }

  return true;
}

/*-------------------PGTIOL----------------------*/
bool mcRPL::Cellspace::
createPgtiolDS(PGTIOLDataset *&pDataset,
               const char *aFileName,
               char **aPgtiolOptions) const {
  if(_pInfo == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: NULL CellspaceInfo, unable to create a GDAL dataset" \
        << endl;
    return false;
  }
  if(_pInfo->isEmpty(true)) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create a GDAL dataset with an empty CellspaceInfo" \
        << endl;
    return false;
  }
  if(aFileName == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create a GDAL dataset using a NULL file name" \
        << endl;
    return false;
  }

  /*
  PGTIOLDriver *pDriver = new PGTIOLDriver();
  if(pDriver == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create a GDAL dataset using a NULL GDAL driver" \
        << endl;
    return false;
  }
  */

  GDALDataType gdalDataType;
  mcRPL::cppType2gdalType(gdalDataType, string(_pInfo->dataType()));
  if(gdalDataType == GDT_Unknown) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: unable to determine appropriate data type for the GDAL dataset" \
         << endl;
    return false;
  }

  /*
  pDataset = pDriver->Create(aFileName,
                             _pInfo->nCols(), _pInfo->nRows(), 1,
                             gdalDataType, aGdalOptions);
  */

  pDataset = new PGTIOLDataset(aFileName, PG_Update, aPgtiolOptions,
                               _pInfo->nCols(),_pInfo->nRows(),1,gdalDataType);

  if(pDataset == NULL) {
    cerr << __FILE__ << " " << __FUNCTION__ \
        << " Error: unable to create pGTIOL dataset (" << aFileName << ")" << endl;
    return false;
  }

  pDataset->SetNoDataValue(_pInfo->getNoDataValAs<double>());

  if(_pInfo->isGeoreferenced(false)) {
    pDataset->SetProjection(_pInfo->georeference()->projection().c_str());
    double aGeoTransform[6];
    _pInfo->georeference()->geoTransform(aGeoTransform);
    pDataset->SetGeoTransform(aGeoTransform);
    pDataset->SetTileWidth(_pInfo->tileSize());
	  pDataset->SetTileLength(_pInfo->tileSize());
  }
  return true;
}

bool mcRPL::Cellspace::
readPGTIOLData(PGTIOLDataset *pDataset,
                int iBand,
                const mcRPL::CellCoord *pGDALOff,
                bool warning) {
  if(!_checkPGTIOLSettings(pDataset, iBand, pGDALOff, NULL, warning)) {
    return false;
  }
 // GDALRasterBand *pBand = pDataset->GetRasterBand(iBand);
  GDALDataType inputType;
  cppType2gdalType(inputType, string(_pInfo->dataType()));
  long iGDALRowOff = 0, iGDALColOff = 0;
  if(pGDALOff != NULL) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
  }
  if(pDataset->RasterIO(PG_Read, iGDALColOff, iGDALRowOff, _pInfo->nCols(), _pInfo->nRows(),
                        _pData, _pInfo->nCols(), _pInfo->nRows(), inputType, 1,0, 0) == PG_Failure) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: failed to read Cellspace from pGTIOL Dataset" << endl;
    }
    return false;
  }
  return true;
}

bool mcRPL::Cellspace::
writePGTIOLData(PGTIOLDataset *pDataset,
              int iBand,
              const mcRPL::CellCoord *pGDALOff,
              const mcRPL::CoordBR *pWorkBR,
              bool warning) {
  if(!_checkPGTIOLSettings(pDataset, iBand, pGDALOff, pWorkBR, warning)) {
    return false;
  }
  //GDALRasterBand *pBand = pDataset->GetRasterBand(iBand);
  GDALDataType outputType;
  cppType2gdalType(outputType,string(_pInfo->dataType()));
  size_t dataSize = _pInfo->dataSize();

  long iGDALRowOff = 0, iGDALColOff = 0;
  if(pGDALOff != NULL) {
    iGDALRowOff = pGDALOff->iRow();
    iGDALColOff = pGDALOff->iCol();
    if(pWorkBR != NULL) {
      iGDALRowOff = iGDALRowOff + pWorkBR->minIRow();
      iGDALColOff = iGDALColOff + pWorkBR->minICol();
    }
  }

  long nDataRows = _pInfo->nRows();
  long nDataCols = _pInfo->nCols();
  if(pWorkBR != NULL) {
    nDataRows = pWorkBR->nRows();
    nDataCols = pWorkBR->nCols();

  }
 
  long lineSpace =  _pInfo->nCols();
  
  /*int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  cout<<"rank"<<rank<<"iGDALColOff"<<iGDALColOff
      <<"iGDALRowOff"<<iGDALRowOff
	  <<"nDataCols"<<nDataCols
	  <<"nDataRows"<<nDataRows
      <<"nDataCols"<<nDataCols<<"outputType"<<outputType<<"linespace"<<lineSpace<<endl;*/ 
  
  
  void *pData = _pData;
  if(pWorkBR != NULL) {
    pData = (char *)_pData + pWorkBR->nwCorner().toIdx(_pInfo->dims())*dataSize;
  }
  
  if(pDataset->RasterIO(PG_Write, iGDALColOff, iGDALRowOff, nDataCols, nDataRows,
                        pData, nDataCols, nDataRows, outputType, 1,0, lineSpace) == PG_Failure) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: failed to write Cellspace to pGTIOL Dataset" << endl;
    }
    return false;
  }
  
  return true;
}


