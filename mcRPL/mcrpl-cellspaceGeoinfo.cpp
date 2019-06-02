#include "mcrpl-cellspaceGeoinfo.h"

mcRPL::CellspaceGeoinfo::
CellspaceGeoinfo(const mcRPL::GeoCoord &nwCorner,
                 const mcRPL::GeoCoord &cellSize,
                 const string &projection)
  :_nwCorner(nwCorner),
   _cellSize(cellSize),
   _projection(projection) {}

mcRPL::CellspaceGeoinfo::
CellspaceGeoinfo(const mcRPL::CellspaceGeoinfo &rhs)
  :_nwCorner(rhs._nwCorner),
   _cellSize(rhs._cellSize),
   _projection(rhs._projection) {}

mcRPL::CellspaceGeoinfo& mcRPL::CellspaceGeoinfo::
operator=(const mcRPL::CellspaceGeoinfo &rhs) {
  if(this != &rhs) {
    _nwCorner = rhs._nwCorner;
    _cellSize = rhs._cellSize;
    _projection = rhs._projection;
  }
  return *this;
}

bool mcRPL::CellspaceGeoinfo::
operator==(const mcRPL::CellspaceGeoinfo &rhs) const {
  return (_nwCorner == rhs._nwCorner &&
          _cellSize == rhs._cellSize &&
          _projection == rhs._projection);
}

void mcRPL::CellspaceGeoinfo::
add2Buf(vector<char> &buf) {
  size_t bufPtr = buf.size();
  size_t lenProj = _projection.size();

  if(lenProj > 0) {
    buf.resize(bufPtr + lenProj);
    memcpy((void *)&(buf[bufPtr]), &(_projection[0]), lenProj);
  }

  bufPtr = buf.size();
  buf.resize(bufPtr + sizeof(size_t));
  memcpy((void *)&(buf[bufPtr]), &lenProj, sizeof(size_t));
  
  _cellSize.add2Buf(buf);
  _nwCorner.add2Buf(buf);
}

bool mcRPL::CellspaceGeoinfo::
initFromBuf(vector<char> &buf) {
  if(!_nwCorner.initFromBuf(buf) ||
     !_cellSize.initFromBuf(buf)) {
    return false;
  }
  
  _projection.clear();

  int bufPtr = buf.size() - sizeof(size_t);
  if(bufPtr < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: the buffer has insufficient memory to extract the length of the Projection a Cellspace" \
        << endl;
    return false;
  }
  size_t lenProj;
  memcpy(&lenProj, (void *)&(buf[bufPtr]), sizeof(size_t));
  buf.erase(buf.begin()+bufPtr, buf.end());

  if(lenProj > 0) {
    bufPtr = buf.size() - lenProj;
    if(bufPtr < 0) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the buffer has insufficient memory to extract the Projection a Cellspace" \
           << endl;
      return false;
    }
    _projection.resize(lenProj);
    memcpy(&(_projection[0]), (void *)&(buf[bufPtr]), lenProj);
    buf.erase(buf.begin()+bufPtr, buf.end());
  }

  return true;
}

void mcRPL::CellspaceGeoinfo::
nwCorner(const mcRPL::GeoCoord &nw) {
  _nwCorner(nw);
}

void mcRPL::CellspaceGeoinfo::
nwCorner(double x, double y) {
  _nwCorner(x, y);
}

void mcRPL::CellspaceGeoinfo::
cellSize(const mcRPL::GeoCoord &size) {
  _cellSize(size);
}

void mcRPL::CellspaceGeoinfo::
cellSize(double xSize, double ySize) {
  _cellSize(xSize, ySize);
}

void mcRPL::CellspaceGeoinfo::
projection(const string &proj) {
  _projection = proj;
}

const mcRPL::GeoCoord& mcRPL::CellspaceGeoinfo::
nwCorner() const {
  return _nwCorner;
}

const mcRPL::GeoCoord& mcRPL::CellspaceGeoinfo::
cellSize() const {
  return _cellSize;
}

const string& mcRPL::CellspaceGeoinfo::
projection() const {
  return _projection;
}


void mcRPL::CellspaceGeoinfo::
geoTransform(double aGeoTransform[6]) const {
  aGeoTransform[0] = _nwCorner.x();
  aGeoTransform[1] = _cellSize.x();
  aGeoTransform[2] = 0;
  aGeoTransform[3] = _nwCorner.y();
  aGeoTransform[4] = 0;
  aGeoTransform[5] = _cellSize.y();
}

mcRPL::GeoCoord mcRPL::CellspaceGeoinfo::
cellCoord2geoCoord(const mcRPL::CellCoord &cCoord) const {
  return mcRPL::GeoCoord(_nwCorner.x() + cCoord.iCol()*_cellSize.x(),
                        _nwCorner.y() + cCoord.iRow()*_cellSize.y() );
}

mcRPL::GeoCoord mcRPL::CellspaceGeoinfo::
cellCoord2geoCoord(long iRow, long iCol) const {
  return mcRPL::GeoCoord(_nwCorner.x() + iCol*_cellSize.x(),
                        _nwCorner.y() + iRow*_cellSize.y() );
}

mcRPL::CellCoord mcRPL::CellspaceGeoinfo::
geoCoord2cellCoord(const mcRPL::GeoCoord &gCoord) const {
  return mcRPL::CellCoord(floor((gCoord.y() - _nwCorner.y())/_cellSize.y()),
                         floor((gCoord.x() - _nwCorner.x())/_cellSize.x()));
}

mcRPL::CellCoord mcRPL::CellspaceGeoinfo::
geoCoord2cellCoord(double x, double y) const {
  return mcRPL::CellCoord(floor((y - _nwCorner.y())/_cellSize.y()),
                         floor((x - _nwCorner.x())/_cellSize.x()));
}

/*-------------GDAL-----*/
bool mcRPL::CellspaceGeoinfo::
initByGDAL(GDALDataset *pDataset,
           bool warning) {
  if(pDataset == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL GDAL dataset" << endl;
    }
    return false;
  }

  double aGeoTransform[6];
  if(pDataset->GetGeoTransform(aGeoTransform) == CE_None) {
    if(aGeoTransform[2] != 0 || aGeoTransform[4] != 0) {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: GDAL dataset is NOT north-up" << endl;
      }
      return false;
    }
    
    nwCorner(aGeoTransform[0], aGeoTransform[3]);
    cellSize(aGeoTransform[1], aGeoTransform[5]);
  }

  if(pDataset->GetProjectionRef() != NULL) {
    _projection.assign(pDataset->GetProjectionRef());
  }

  /*
  for(int i = 0; i < 6; i++) {
    cout << aGeoTransform[i] << "\t";
  }
  cout << endl;
  cout << _projection << endl;
  */

  return true;
}

/*----------PGTIOL-------------*/
bool mcRPL::CellspaceGeoinfo::initByPGTIOL(PGTIOLDataset *pDataset,
                                          bool warning){
 if(pDataset == NULL) {
	if(warning) {
	  cerr << __FILE__ << " function:" << __FUNCTION__ \
		   << " Error: NULL dataset" << endl;
	}
	return false;
  }

  double aGeoTransform[6];
  pDataset->GetGeoTransform(aGeoTransform);
	if(aGeoTransform[2] != 0 || aGeoTransform[4] != 0) {
	  if(warning) {
		cerr << __FILE__ << " function:" << __FUNCTION__ \
			 << " Error: dataset is NOT north-up" << endl;
	  }
	  return false;
	}
	nwCorner(aGeoTransform[0], aGeoTransform[3]);
	cellSize(aGeoTransform[1], aGeoTransform[5]);
 
  if(pDataset->GetProjectionRef() != NULL) {
	_projection.assign(pDataset->GetProjectionRef());
  }
 
  return true;
}

mcRPL::CellspaceGeoinfo mcRPL::CellspaceGeoinfo::
subSpaceGeoinfo(const mcRPL::SubCellspaceInfo &subInfo) const {
  return mcRPL::CellspaceGeoinfo(cellCoord2geoCoord(subInfo.MBR().nwCorner()),
                                cellSize(),
                                projection());
}
