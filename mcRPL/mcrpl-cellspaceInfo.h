#ifndef MCRPL_CELLSPACEINFO_H
#define MCRPL_CELLSPACEINFO_H

#include "mcrpl-basicTypes.h"
#include "mcrpl-cellspaceGeoinfo.h"
#include "mcrpl-neighborhood.h"

namespace mcRPL {
  class CellspaceInfo {
    public:
      CellspaceInfo();
      CellspaceInfo(const mcRPL::CellspaceInfo &rhs);
      CellspaceInfo(const mcRPL::SpaceDims &dims,
                    const char *aTypeName,
                    size_t typeSize,
                    const mcRPL::CellspaceGeoinfo *pGeoinfo = NULL,
                    long tileSize = TILEWIDTH);
      ~CellspaceInfo();

      mcRPL::CellspaceInfo& operator=(const mcRPL::CellspaceInfo &rhs);
      bool operator==(const mcRPL::CellspaceInfo &rhs);
      bool operator!=(const mcRPL::CellspaceInfo &rhs);

      bool init(const mcRPL::SpaceDims &dims,
                const char *aTypeName,
                size_t typeSize,
                const mcRPL::CellspaceGeoinfo *pGeoinfo = NULL,
                long tileSize = TILEWIDTH);

      bool initByGDAL(GDALDataset *pDataset,
                      int iBand,
                      bool warning = true);
     bool initByPGTIOL(PGTIOLDataset *pDataset,
                       int iBand,
                       bool warning = true);

      void clear();

      void add2Buf(vector<char> &buf);
      bool initFromBuf(vector<char> &buf);

      bool isEmpty(bool warning = false) const;
      void dims(const mcRPL::SpaceDims& cellDims);
      const mcRPL::SpaceDims& dims() const;
      long nRows() const;
      long nCols() const;

      long size() const;
      const char* dataType() const;
      size_t dataSize() const;

      void georeference(const mcRPL::CellspaceGeoinfo &geoinfo);
      const mcRPL::CellspaceGeoinfo* georeference() const;
      bool isGeoreferenced(bool warning = false) const;

      template<class CmprType>
      bool isDataType(bool warning = false) const;

      template<class RtrType>
      RtrType& getNoDataVal(bool warning = true) const;
      template<class RtrType>
      const RtrType getNoDataValAs(bool warning = true) const;

      template<class ElemType>
      bool setNoDataVal(const ElemType &val,
                        bool warning = true);
      template<class ElemType>
      bool setNoDataValAs(const ElemType &val,
                          bool warning = true);

      long tileSize() const;
      void setTileSize(long size);
	  int cudaTypeCopy();
      /* Coordinate */
      long coord2idx(const mcRPL::CellCoord &coord) const;
      long coord2idx(long iRow, long iCol) const;
      mcRPL::CellCoord idx2coord(long idx) const;
      bool validCoord(const mcRPL::CellCoord &coord,
                      bool warning = true) const;
      bool validCoord(long iRow, long iCol,
                      bool warning = true) const;
      bool validIdx(long idx,
                    bool warning = true) const;
      bool validBR(const mcRPL::CoordBR &rectangle,
                   bool warning = true) const;

      mcRPL::GeoCoord cellCoord2geoCoord(const mcRPL::CellCoord &cCoord) const;
      mcRPL::GeoCoord cellCoord2geoCoord(long iRow, long iCol) const;
      mcRPL::CellCoord geoCoord2cellCoord(const mcRPL::GeoCoord &gCoord) const;
      mcRPL::CellCoord geoCoord2cellCoord(double x, double y) const;

      /* Neighborhood operations */
      bool calcWorkBR(mcRPL::CoordBR *pWorkBR,
                      const mcRPL::Neighborhood *pNbrhd,
                      bool warning = true) const;

      bool validNbrhd(const mcRPL::Neighborhood &nbrhd,
                      const mcRPL::CellCoord &ctrCoord,
                      bool warning = true) const;
      bool validNbrhd(const mcRPL::Neighborhood &nbrhd,
                      long ctrRow, long ctrCol,
                      bool warning = true) const;
      bool validNbrhd(const mcRPL::Neighborhood &nbrhd,
                      long idx,
                      bool warning = true) const;

    private:
	  int _cudaType;
      mcRPL::SpaceDims _dims;
      string _dTypeName;
      size_t _dTypeSize;
      void *_pNoDataVal;
      mcRPL::CellspaceGeoinfo *_pGeoInfo;
      long _tileSize;
  };
};

template<class CmprType>
bool mcRPL::CellspaceInfo::
isDataType(bool warning) const {
  bool valid = true;
  if(_dTypeName.compare(typeid(CmprType).name()) || _dTypeSize != sizeof(CmprType)) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: Cellspace element data type (" << _dTypeName \
           << ") does NOT match requested data type (" << typeid(CmprType).name() \
           << ")" << endl;
    }
    valid = false;
  }
  return valid;
}
template<class RtrType>
RtrType& mcRPL::CellspaceInfo::
getNoDataVal(bool warning) const {
  RtrType *pNoData = NULL;
  if(isDataType<RtrType>(warning) && _pNoDataVal != NULL) {
    pNoData = (RtrType*)_pNoDataVal;
  }
  return *pNoData;
}

template<class RtrType>
const RtrType mcRPL::CellspaceInfo::
getNoDataValAs(bool warning) const {
  RtrType val = (RtrType)mcRPL::ERROR_VAL;

  if(_pNoDataVal != NULL) {
    if(isDataType<bool>()) {
      val = (RtrType)(getNoDataVal<bool>());
    }
    else if(isDataType<unsigned char>()) {
      val = (RtrType)(getNoDataVal<unsigned char>());
    }
    else if(isDataType<char>()) {
      val = (RtrType)(getNoDataVal<char>());
    }
    else if(isDataType<unsigned short>()) {
      val = (RtrType)(getNoDataVal<unsigned short>());
    }
    else if(isDataType<short>()) {
      val = (RtrType)(getNoDataVal<short>());
    }
    else if(isDataType<unsigned int>()) {
      val = (RtrType)(getNoDataVal<unsigned int>());
    }
    else if(isDataType<int>()) {
      val = (RtrType)(getNoDataVal<int>());
    }
    else if(isDataType<unsigned long>()) {
      val = (RtrType)(getNoDataVal<unsigned long>());
    }
    else if(isDataType<long>()) {
      val = (RtrType)(getNoDataVal<long>());
    }
    else if(isDataType<float>()) {
      val = (RtrType)(getNoDataVal<float>());
    }
    else if(isDataType<double>()) {
      val = (RtrType)(getNoDataVal<double>());
    }
    else if(isDataType<long double>()) {
      val = (RtrType)(getNoDataVal<long double>());
    }
    else {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
            << " Error: unable to convert Cellspace data type (" \
            << dataType() << ") to output data type (" << typeid(RtrType).name() \
            << "). Returned an INVALID value. "<< endl;
      }
    }
  } // end -- if(_pNoDataVal != NULL)
  return val;
}

template<class ElemType>
bool mcRPL::CellspaceInfo::
setNoDataVal(const ElemType &val,
             bool warning) {
  bool done = false;
  if(isDataType<ElemType>(warning) && _pNoDataVal != NULL) {
    (ElemType&)(*((ElemType*)_pNoDataVal)) = val;
    done = true;
  }
  return done;
}

template<class ElemType>
bool mcRPL::CellspaceInfo::
setNoDataValAs(const ElemType &val,
               bool warning) {  bool done = false;
  if(!isEmpty(warning) && _pNoDataVal != NULL) {
    if(isDataType<bool>()) {
      done = setNoDataVal(static_cast<bool>(val), warning);
    }
    else if(isDataType<unsigned char>()) {
      done = setNoDataVal(static_cast<unsigned char>(val), warning);
    }
    else if(isDataType<char>()) {
      done = setNoDataVal(static_cast<char>(val), warning);
    }
    else if(isDataType<unsigned short>()) {
      done = setNoDataVal(static_cast<unsigned short>(val), warning);
    }
    else if(isDataType<short>()) {
      done = setNoDataVal(static_cast<short>(val), warning);
    }
    else if(isDataType<unsigned int>()) {
      done = setNoDataVal(static_cast<unsigned int>(val), warning);
    }
    else if(isDataType<int>()) {
      done = setNoDataVal(static_cast<int>(val), warning);
    }
    else if(isDataType<unsigned long>()) {
      done = setNoDataVal(static_cast<unsigned long>(val), warning);
    }
    else if(isDataType<long>()) {
      done = setNoDataVal(static_cast<long>(val), warning);
    }
    else if(isDataType<float>()) {
      done = setNoDataVal(static_cast<float>(val), warning);
    }
    else if(isDataType<double>()) {
      done = setNoDataVal(static_cast<double>(val), warning);
    }
    else if(isDataType<long double>()) {
      done = setNoDataVal(static_cast<long double>(val), warning);
    }
    else {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: unable to convert input data type (" \
             << typeid(ElemType).name() << ") to Cellspace data type (" \
             << dataType() << ")" << endl;
        done = false;
      }
    }
  } // end -- if(!isEmpty(warning) && _pNoDataVal != NULL)

  return done;
}
#endif /* CELLSPACEINFO_H */
