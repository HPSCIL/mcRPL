#include "mcPRL-cellspaceGeoinfo.h"

mcPRL::CellspaceGeoinfo::
CellspaceGeoinfo(const mcPRL::GeoCoord &nwCorner,
                 const mcPRL::GeoCoord &cellSize,
                 const string &projection)
  :_nwCorner(nwCorner),
   _cellSize(cellSize),
   _projection(projection) {}

mcPRL::CellspaceGeoinfo::
CellspaceGeoinfo(const mcPRL::CellspaceGeoinfo &rhs)
  :_nwCorner(rhs._nwCorner),
   _cellSize(rhs._cellSize),
   _projection(rhs._projection) {}

mcPRL::CellspaceGeoinfo& mcPRL::CellspaceGeoinfo::
operator=(const mcPRL::CellspaceGeoinfo &rhs) {
  if(this != &rhs) {
    _nwCorner = rhs._nwCorner;
    _cellSize = rhs._cellSize;
    _projection = rhs._projection;
  }
  return *this;
}

bool mcPRL::CellspaceGeoinfo::
operator==(const mcPRL::CellspaceGeoinfo &rhs) const {
  return (_nwCorner == rhs._nwCorner &&
          _cellSize == rhs._cellSize &&
          _projection == rhs._projection);
}

void mcPRL::CellspaceGeoinfo::
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

bool mcPRL::CellspaceGeoinfo::
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

void mcPRL::CellspaceGeoinfo::
nwCorner(const mcPRL::GeoCoord &nw) {
  _nwCorner(nw);
}

void mcPRL::CellspaceGeoinfo::
nwCorner(double x, double y) {
  _nwCorner(x, y);
}

void mcPRL::CellspaceGeoinfo::
cellSize(const mcPRL::GeoCoord &size) {
  _cellSize(size);
}

void mcPRL::CellspaceGeoinfo::
cellSize(double xSize, double ySize) {
  _cellSize(xSize, ySize);
}

void mcPRL::CellspaceGeoinfo::
projection(const string &proj) {
  _projection = proj;
}

const mcPRL::GeoCoord& mcPRL::CellspaceGeoinfo::
nwCorner() const {
  return _nwCorner;
}

const mcPRL::GeoCoord& mcPRL::CellspaceGeoinfo::
cellSize() const {
  return _cellSize;
}

const string& mcPRL::CellspaceGeoinfo::
projection() const {
  return _projection;
}


void mcPRL::CellspaceGeoinfo::
geoTransform(double aGeoTransform[6]) const {
  aGeoTransform[0] = _nwCorner.x();
  aGeoTransform[1] = _cellSize.x();
  aGeoTransform[2] = 0;
  aGeoTransform[3] = _nwCorner.y();
  aGeoTransform[4] = 0;
  aGeoTransform[5] = _cellSize.y();
}

mcPRL::GeoCoord mcPRL::CellspaceGeoinfo::
cellCoord2geoCoord(const mcPRL::CellCoord &cCoord) const {
  return mcPRL::GeoCoord(_nwCorner.x() + cCoord.iCol()*_cellSize.x(),
                        _nwCorner.y() + cCoord.iRow()*_cellSize.y() );
}

mcPRL::GeoCoord mcPRL::CellspaceGeoinfo::
cellCoord2geoCoord(long iRow, long iCol) const {
  return mcPRL::GeoCoord(_nwCorner.x() + iCol*_cellSize.x(),
                        _nwCorner.y() + iRow*_cellSize.y() );
}

mcPRL::CellCoord mcPRL::CellspaceGeoinfo::
geoCoord2cellCoord(const mcPRL::GeoCoord &gCoord) const {
  return mcPRL::CellCoord(floor((gCoord.y() - _nwCorner.y())/_cellSize.y()),
                         floor((gCoord.x() - _nwCorner.x())/_cellSize.x()));
}

mcPRL::CellCoord mcPRL::CellspaceGeoinfo::
geoCoord2cellCoord(double x, double y) const {
  return mcPRL::CellCoord(floor((y - _nwCorner.y())/_cellSize.y()),
                         floor((x - _nwCorner.x())/_cellSize.x()));
}

/*-------------GDAL-----*/
bool mcPRL::CellspaceGeoinfo::
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
bool mcPRL::CellspaceGeoinfo::initByPGTIOL(PGTIOLDataset *pDataset,
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

mcPRL::CellspaceGeoinfo mcPRL::CellspaceGeoinfo::
subSpaceGeoinfo(const mcPRL::SubCellspaceInfo &subInfo) const {
  return mcPRL::CellspaceGeoinfo(cellCoord2geoCoord(subInfo.MBR().nwCorner()),
                                cellSize(),
                                projection());
}
