#include "mcPRL-basicTypes.h"

void mcPRL::
gdalType2cppType(string &cppTypeName,
                 size_t &cppTypeSize,
                 GDALDataType gdalType) {
  cppTypeName.clear();
  cppTypeSize = 0;

  switch(gdalType) {
    case GDT_Unknown:
      cppTypeName.assign("UNKNOWN");
      cppTypeSize = 0;
      break;
    case GDT_Byte:
      cppTypeName.assign(typeid(unsigned char).name());
      cppTypeSize = sizeof(unsigned char);
      break;
    case GDT_UInt16:
      cppTypeName.assign(typeid(unsigned short int).name());
      cppTypeSize = sizeof(unsigned short int);
      break;
    case GDT_Int16:
      cppTypeName.assign(typeid(short int).name());
      cppTypeSize = sizeof(short int);
      break;
    case GDT_UInt32:
      cppTypeName.assign(typeid(unsigned int).name());
      cppTypeSize = sizeof(unsigned int);
      break;
    case GDT_Int32:
      cppTypeName.assign(typeid(int).name());
      cppTypeSize = sizeof(int);
      break;
    case GDT_Float32:
      cppTypeName.assign(typeid(float).name());
      cppTypeSize = sizeof(float);
      break;
    case GDT_Float64:
      cppTypeName.assign(typeid(double).name());
      cppTypeSize = sizeof(double);
      break;
    default:
	    cppTypeName.assign("UNKNOWN");
      cppTypeSize = 0;
      break;
  }
}

void mcPRL::
cppType2gdalType(GDALDataType &gdalType,
                 const string &cppTypeName) {
  gdalType = GDT_Unknown;
  if(cppTypeName == typeid(unsigned char).name()) {
    gdalType = GDT_Byte;
  }
  else if(cppTypeName == typeid(unsigned short int).name()) {
    gdalType = GDT_UInt16;
  }
  else if(cppTypeName == typeid(short int).name()) {
    gdalType = GDT_Int16;
  }
  else if(cppTypeName == typeid(unsigned int).name()) {
    gdalType = GDT_UInt32;
  }
  else if(cppTypeName == typeid(int).name()) {
    gdalType = GDT_Int32;
  }
  else if(cppTypeName == typeid(float).name()) {
    gdalType = GDT_Float32;
  }
  else if(cppTypeName == typeid(double).name()) {
    gdalType = GDT_Float64;
  }
}

bool mcPRL::
initDefVal(void *&pVal,
           const string &cppTypeName) {
  if(pVal != NULL) {
    free(pVal);
    pVal = NULL;
  }

  if(cppTypeName == typeid(bool).name()) {
    pVal = (void *)calloc(1, sizeof(bool));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_BOOL, sizeof(bool));
  }
  else if(cppTypeName == typeid(unsigned char).name()) {
    pVal = (void *)calloc(1, sizeof(unsigned char));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_UCHAR, sizeof(unsigned char));
  }
  else if(cppTypeName == typeid(char).name()) {
    pVal = (void *)calloc(1, sizeof(char));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_CHAR, sizeof(char));
  }
  else if(cppTypeName == typeid(unsigned short int).name()) {
    pVal = (void *)calloc(1, sizeof(unsigned short int));
    memcpy(pVal, &DEFAULT_NODATA_USHORT, sizeof(unsigned short int));
  }
  else if(cppTypeName == typeid(short int).name()) {
    pVal = (void *)calloc(1, sizeof(short int));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_SHORT, sizeof(short int));
  }
  else if(cppTypeName == typeid(unsigned int).name()) {
    pVal = (void *)calloc(1, sizeof(unsigned int));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_UINT, sizeof(unsigned int));
  }
  else if(cppTypeName == typeid(int).name()) {
    pVal = (void *)calloc(1, sizeof(int));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_INT, sizeof(int));
  }
  else if(cppTypeName == typeid(unsigned long).name()) {
    pVal = (void *)calloc(1, sizeof(unsigned long));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_ULONG, sizeof(unsigned long));
  }
  else if(cppTypeName == typeid(long).name()) {
    pVal = (void *)calloc(1, sizeof(long));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_LONG, sizeof(long));
  }
  else if(cppTypeName == typeid(float).name()) {
    pVal = (void *)calloc(1, sizeof(float));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_FLOAT, sizeof(float));
  }
  else if(cppTypeName == typeid(double).name()) {
    pVal = (void *)calloc(1, sizeof(double));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_DOUBLE, sizeof(double));
  }
  else if(cppTypeName == typeid(long double).name()) {
    pVal = (void *)calloc(1, sizeof(long double));
    memcpy(pVal, &mcPRL::DEFAULT_NODATA_LDOUBLE, sizeof(long double));
  }
  else {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid data type (" << cppTypeName \
         << ")." << endl;
    return false;
  }

  return true;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::DomDcmpMethod &dcmpMethod) {
  switch(dcmpMethod) {
    case mcPRL::SNGL_PRC:
      os << "NONE_DCMP";
      break;
    case mcPRL::SMPL_ROW:
      os << "SIMPLE_ROW-WISE";
      break;
    case mcPRL::SMPL_COL:
      os << "SIMPLE_COL-WISE";
      break;
    case mcPRL::SMPL_BLK:
      os << "SIMPLE_BLK-WISE";
      break;
    case mcPRL::WKLD_ROW:
      os << "WORKLOAD_ROW-WISE";
      break;
    case mcPRL::WKLD_COL:
      os << "WORKLOAD_COL-WISE";
      break;
    case mcPRL::WKLD_BLK:
      os << "WORKLOAD_BLK-WISE";
      break;
    case mcPRL::WKLD_QTB:
      os << "WORKLOAD_QUADTREE";
      break;
    default:
      os << "UNKNOWN_DCMP (" << dcmpMethod << ")";
      break;
  }
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::MappingMethod &mapMethod) {
  switch(mapMethod) {
    case mcPRL::CYLC_MAP:
      os << "CYCLIC_MAPPING";
      break;
    case mcPRL::BKFR_MAP:
      os << "BACK-FORTH_MAPPING";
      break;
    default:
      os << "UNKNOWN_MAPPING (" << mapMethod << ")";
      break;
  }
  return os;
}

mcPRL::MeshDir mcPRL::
oppositeDir(mcPRL::MeshDir dir) {
  mcPRL::MeshDir opDir = mcPRL::NONE_DIR;
  if(dir >= mcPRL::NORTH_DIR &&
     dir <= mcPRL::NORTHWEST_DIR) {
    int tmpDir = dir + 4;
    if(tmpDir >= 8) {
      tmpDir -= 8;
    }
    opDir = static_cast<mcPRL::MeshDir>(tmpDir);
  }
  return opDir;
}

mcPRL::PrimeDir mcPRL::
oppositePrmDir(mcPRL::PrimeDir dir) {
  mcPRL::PrimeDir opPrmDir = mcPRL::NONE_PDIR;
  if(dir >= mcPRL::NORTH_PDIR &&
     dir <= mcPRL::WEST_PDIR) {
    int tmpDir = dir + 2;
    if(tmpDir >= 4) {
      tmpDir -= 4;
    }
    opPrmDir = static_cast<mcPRL::PrimeDir>(tmpDir);
  }
  return opPrmDir;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::BasicNeighborhoodType &val) {
  switch(val) {
    case mcPRL::SINGLE_CELL_NBRHD:
      os << "SINGLE_CELL";
      break;
    case mcPRL::VON_NEUMANN_NBRHD:
      os << "VON_NEUMANN";
      break;
    case mcPRL::MOORE_NBRHD:
      os << "MOORE";
      break;
    default:
      break;
  }
  return os;
}

istream& mcPRL::
operator>>(istream &is, mcPRL::BasicNeighborhoodType &val) {
  string sVal;
  is >> sVal;
  if(sVal == "SINGLE_CELL") {
    val = mcPRL::SINGLE_CELL_NBRHD;
  }
  else if(sVal == "VON_NEUMANN") {
    val = mcPRL::VON_NEUMANN_NBRHD;
  }
  else if(sVal == "MOORE") {
    val = mcPRL::MOORE_NBRHD;
  }
  else {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid BasicNeighborhoodType (" << val \
         << "). SINGLE_CELL_NBRHD is used as the value." << endl;
    val = mcPRL::SINGLE_CELL_NBRHD;
  }
  return is;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::IntVect &vInt) {
  for(size_t iVal = 0; iVal < vInt.size(); iVal++) {
    os << vInt[iVal];
    if(iVal < vInt.size() - 1) {
      os << "\t";
    }
  }
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::LongVect &vLong) {
  for(size_t iVal = 0; iVal < vLong.size(); iVal++) {
    os << vLong[iVal];
    if(iVal < vLong.size() - 1) {
      os << "\t";
    }
  }
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::DblVect &vDbl) {
  for(size_t iVal = 0; iVal < vDbl.size(); iVal++) {
    os << vDbl[iVal];
    if(iVal < vDbl.size() - 1) {
      os << "\t";
    }
  }
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::CharVect &vChar) {
  for(size_t iVal = 0; iVal < vChar.size(); iVal++) {
    os << vChar[iVal];
    if(iVal < vChar.size() - 1) {
      os << "\t";
    }
  }
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::IntPair &pair) {
  os << pair.first << " " << pair.second;
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::IntMap &mInt) {
  mcPRL::IntMap::const_iterator iVal = mInt.begin();
  while(iVal != mInt.end()) {
    os << "{" << (*iVal).first << ":\t" << (*iVal).second << "}";
    os << "\t";
    iVal++;
  }
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::SpaceDims &dims) {
  os << dims.nRows() << " " << dims.nCols();
  return os;
}

istream& mcPRL::
operator>>(istream &is, mcPRL::SpaceDims &dims) {
  long numRows, numCols;
  is >> numRows;
  is >> numCols;
  dims.nRows(numRows);
  dims.nCols(numCols);
  return is;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::CellCoord &coord) {
  os << coord.iRow() << " " << coord.iCol();
  return os;
}

istream& mcPRL::
operator>>(istream &is, mcPRL::CellCoord &coord) {
  long rowIdx, colIdx;
  is >> rowIdx;
  is >> colIdx;
  coord.iRow(rowIdx);
  coord.iCol(colIdx);
  return is;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::WeightedCellCoord &coord) {
  os << coord.iRow() << " " << coord.iCol() << " " << coord.weight();
  return os;
}

istream& mcPRL::
operator>>(istream &is, mcPRL::WeightedCellCoord &coord) {
  long rowIdx, colIdx;
  double wVal;
  is >> rowIdx;
  is >> colIdx;
  is >> wVal;
  coord.iRow(rowIdx);
  coord.iCol(colIdx);
  coord.weight(wVal);
  return is;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::CoordBR &br) {
  os << br.nwCorner() << "; " << br.seCorner();
  return os;
}

ostream& mcPRL::
operator<<(ostream &os, const mcPRL::GeoCoord &gcoord) {
  os << gcoord.x() << " " << gcoord.y();
  return os;
}

istream& mcPRL::
operator>>(istream &is, mcPRL::GeoCoord &gcoord) {
  double xCoord, yCoord;
  is >> xCoord;
  is >> yCoord;
  gcoord.x(xCoord);
  gcoord.y(yCoord);
  return is;
}
