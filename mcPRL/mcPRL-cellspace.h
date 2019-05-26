#ifndef mcPRL_CELLSPACE_H
#define mcPRL_CELLSPACE_H

#include "mcPRL-basicTypes.h"
#include "mcPRL-cellspaceInfo.h"
#include "mcPRL-neighborhood.h"
#include <chrono>
//#include <boost/chrono/include.hpp>
using namespace std;

namespace mcPRL {
  class Cellspace {
    public:
      /* Constructors and Destructor */
      Cellspace();
      Cellspace(const mcPRL::Cellspace &rhs);
      Cellspace(mcPRL::CellspaceInfo *pInfo,
                bool ifInitMem = true);
      
      ~Cellspace();
      
      /* Memory allocation and de-allocation */
      bool initMem();

      bool init(mcPRL::CellspaceInfo *pInfo,
                bool ifInitMem = true);
      
      template <class ElemType>
      bool initVal(const ElemType *pVal = NULL);
      
      template<class InputType>
      bool initValAs(const InputType *pVal = NULL,
                     bool warning = true);

      void clear();
	  template<class DataType>
	  DataType* getData(){DataType* pElem = NULL; pElem = (DataType*)_pData;return  pElem ; };
      /* Operators */
      mcPRL::Cellspace& operator=(const mcPRL::Cellspace& rhs);
	  template<class InputType>
	 bool brupdateCellAs(const mcPRL::CoordBR &br,void *output);
	 template<class InputType>
     bool createrandForCellsAs(long ncols, long nrows);
	 bool createrandForCells(long ncols,long nrows);
      void* getGPUData();
	   void transDataToGPU();
	  void getGPUMalloc();
	  void deleteGPUData();
      /* Information */
      const mcPRL::CellspaceInfo* info() const;
      mcPRL::CellspaceInfo* info();
      void info(mcPRL::CellspaceInfo *pInfo);

      bool isEmpty(bool warning = false) const;
      
      /* Elements */
      void* at(long idx,
               bool warning = true);
      const void* at(long idx,
                     bool warning = true) const;

      template<class RtrType>
      RtrType* row(long iRow,
                   bool warning = true);
      template<class RtrType>
      const RtrType* row(long iRow,
                         bool warning = true) const;

      template<class RtrType>
      RtrType& at(const mcPRL::CellCoord &coord,
                  bool warning = true);
      template<class RtrType>
      const RtrType& at(const mcPRL::CellCoord &coord,
                        bool warning = true) const;

      template<class RtrType>
      RtrType& at(long iRow, long iCol,
                  bool warning = true);
      template<class RtrType>
      const RtrType& at(long iRow, long iCol,
                        bool warning = true) const;

      template<class RtrType>
      RtrType& at(long idx,
                  bool warning = true);
      template<class RtrType>
      const RtrType& at(long idx,
                        bool warning = true) const;

      template<class RtrType>
      RtrType atAs(const mcPRL::CellCoord &coord,
                   bool warning = true) const;
      template<class RtrType>
      RtrType atAs(long iRow, long iCol,
                   bool warning = true) const;
      template<class RtrType>
      RtrType atAs(long idx,
                   bool warning = true) const;

      template<class RtrType>
      bool values(vector<RtrType> *pvVals,
                  const mcPRL::CoordBR *pRectangle = NULL,
                  bool warning = true) const;

      template<class RtrType>
      bool valuesAs(vector<RtrType> *pvVals,
                    const mcPRL::CoordBR *pRectangle = NULL,
                    bool warning = true) const;

      /*
       * example: myCellspace.find(&vIdxs, &val, &br);
       */
      template<class ElemType>
      bool find(mcPRL::LongVect *pvFoundIdxs,
                const ElemType *pVal,
                const mcPRL::CoordBR *pRectangle = NULL,
                bool warning = true) const;

      /*
       * example: myCellspace.find(&vIdx, &vVals, &br);
       */
      template<class ElemType>
      bool find(mcPRL::LongVect *pvFoundIdxs,
                const vector<ElemType> *pvVals,
                const mcPRL::CoordBR *pRectangle = NULL,
                bool warning = true) const;

      /*
       * example: myCellspace.find<int>(&vIdxs, bind2nd(greater<int>(), 10), &br);
       */
      template<class ElemType, class Predicate>
      bool find(mcPRL::LongVect *pvFoundIdxs,
                Predicate pred,
                const mcPRL::CoordBR *pRectangle = NULL,
                bool excldNoData = true,
                bool warning = true) const;

      /*
       * example: myCellspace.count(&val, &br);
       */
      template<class ElemType>
      long count(const ElemType *pVal,
                 const mcPRL::CoordBR *pRectangle = NULL,
                 bool warning = true) const;

      /*
       * example: myCellspace.count(&vVals, &br);
       */
      template<class ElemType>
      long count(const vector<ElemType> *pvVals,
                 const mcPRL::CoordBR *pRectangle = NULL,
                 bool warning = true) const;

      /*
       * example: myCellspace.count<int>(bind2nd(greater<int>(), 10), &br);
       */
      template<class ElemType, class Predicate>
      long count(Predicate pred,
                 const mcPRL::CoordBR *pRectangle = NULL,
                 bool excldNoData = true,
                 bool warning = true) const;


      template<class ElemType>
      bool frequences(map<ElemType, long> *pmFreqs,
                      const mcPRL::CoordBR *pRectangle = NULL,
                      bool warning = true) const;

      template<class RtrType>
      bool frequencesAs(map<RtrType, long> *pmFreqs,
                        const mcPRL::CoordBR *pRectangle = NULL,
                        bool warning = true) const;

      template<class RtrType>
      bool nbrhdVals(vector<RtrType> &vVals,
                     const mcPRL::Neighborhood &nbrhd,
                     const mcPRL::CellCoord &ctrCoord,
                     bool includeCtr = true,
                     bool warning = true) const;
      template<class RtrType>
      bool nbrhdVals(vector<RtrType> &vVals,
                     const mcPRL::Neighborhood &nbrhd,
                     long ctrRow, long ctrCol,
                     bool includeCtr = true,
                     bool warning = true) const;
      template<class RtrType>
      bool nbrhdVals(vector<RtrType> &vVals,
                     const mcPRL::Neighborhood &nbrhd,
                     long idx,
                     bool includeCtr = true,
                     bool warning = true) const;

      template<class RtrType>
      bool nbrhdValsAs(vector<RtrType> &vVals,
                       const mcPRL::Neighborhood &nbrhd,
                       const mcPRL::CellCoord &ctrCoord,
                       bool includeCtr = true,
                       bool warning = true) const;
      template<class RtrType>
      bool nbrhdValsAs(vector<RtrType> &vVals,
                       const mcPRL::Neighborhood &nbrhd,
                       long ctrRow, long ctrCol,
                       bool includeCtr = true,
                       bool warning = true) const;
      template<class RtrType>
      bool nbrhdValsAs(vector<RtrType> &vVals,
                       const mcPRL::Neighborhood &nbrhd,
                       long idx,
                       bool includeCtr = true,
                       bool warning = true) const;

      template<class RtrType>
      bool nbrhdFreqs(map<RtrType, int> &mFreqs,
                      const mcPRL::Neighborhood &nbrhd,
                      const mcPRL::CellCoord &ctrCoord,
                      bool includeCtr = true,
                      bool warning = true) const;
      template<class RtrType>
      bool nbrhdFreqs(map<RtrType, int> &mFreqs,
                      const mcPRL::Neighborhood &nbrhd,
                      long ctrRow, long ctrCol,
                      bool includeCtr = true,
                      bool warning = true) const;
      template<class RtrType>
      bool nbrhdFreqs(map<RtrType, int> &mFreqs,
                      const mcPRL::Neighborhood &nbrhd,
                      long idx,
                      bool includeCtr = true,
                      bool warning = true) const;

      template<class RtrType>
      bool wnbrhdFreqs(map<RtrType, double> &mFreqs,
                       const mcPRL::Neighborhood &nbrhd,
                       const mcPRL::CellCoord &ctrCoord,
                       bool includeCtr = true,
                       bool warning = true) const;
      template<class RtrType>
      bool wnbrhdFreqs(map<RtrType, double> &mFreqs,
                       const mcPRL::Neighborhood &nbrhd,
                       long ctrRow, long ctrCol,
                       bool includeCtr = true,
                       bool warning = true) const;
      template<class RtrType>
      bool wnbrhdFreqs(map<RtrType, double> &mFreqs,
                       const mcPRL::Neighborhood &nbrhd,
                       long idx,
                       bool includeCtr = true,
                       bool warning = true) const;

      template<class RtrType>
      bool nbrhdFreqsAs(map<RtrType, int> &mFreqs,
                        const mcPRL::Neighborhood &nbrhd,
                        const mcPRL::CellCoord &ctrCoord,
                        bool includeCtr = true,
                        bool warning = true) const;
      template<class RtrType>
      bool nbrhdFreqsAs(map<RtrType, int> &mFreqs,
                        const mcPRL::Neighborhood &nbrhd,
                        long ctrRow, long ctrCol,
                        bool includeCtr = true,
                        bool warning = true) const;
      template<class RtrType>
      bool nbrhdFreqsAs(map<RtrType, int> &mFreqs,
                        const mcPRL::Neighborhood &nbrhd,
                        long idx,
                        bool includeCtr = true,
                        bool warning = true) const;

      template<class RtrType>
      bool wnbrhdFreqsAs(map<RtrType, double> &mFreqs,
                         const mcPRL::Neighborhood &nbrhd,
                         const mcPRL::CellCoord &ctrCoord,
                         bool includeCtr = true,
                         bool warning = true) const;
      template<class RtrType>
      bool wnbrhdFreqsAs(map<RtrType, double> &mFreqs,
                         const mcPRL::Neighborhood &nbrhd,
                         long ctrRow, long ctrCol,
                         bool includeCtr = true,
                         bool warning = true) const;
      template<class RtrType>
      bool wnbrhdFreqsAs(map<RtrType, double> &mFreqs,
                         const mcPRL::Neighborhood &nbrhd,
                         long idx,
                         bool includeCtr = true,
                         bool warning = true) const;

      template<class RtrType>
      RtrType nbrhdTotalVal(const mcPRL::Neighborhood &nbrhd,
                            const mcPRL::CellCoord &ctrCoord,
                            bool includeCtr = true,
                            bool warning = true) const;
      template<class RtrType>
      RtrType nbrhdTotalVal(const mcPRL::Neighborhood &nbrhd,
                            long ctrRow, long ctrCol,
                            bool includeCtr = true,
                            bool warning = true) const;
      template<class RtrType>
      RtrType nbrhdTotalVal(const mcPRL::Neighborhood &nbrhd,
                            long idx,
                            bool includeCtr = true,
                            bool warning = true) const;

      template<class RtrType>
      RtrType nbrhdTotalValAs(const mcPRL::Neighborhood &nbrhd,
                              const mcPRL::CellCoord &ctrCoord,
                              bool includeCtr = true,
                              bool warning = true) const;
      template<class RtrType>
      RtrType nbrhdTotalValAs(const mcPRL::Neighborhood &nbrhd,
                              long ctrRow, long ctrCol,
                              bool includeCtr = true,
                              bool warning = true) const;
      template<class RtrType>
      RtrType nbrhdTotalValAs(const mcPRL::Neighborhood &nbrhd,
                              long idx,
                              bool includeCtr = true,
                              bool warning = true) const;

      /* Update */
      void setUpdateTracking(bool toTrack);
      const mcPRL::LongVect& getUpdatedIdxs() const;
      void clearUpdatedIdxs();
      bool brupdateCell(const mcPRL::CoordBR &br,void *output);
      template<class ElemType>
      bool updateCell(const mcPRL::CellCoord &coord, const ElemType &val,
                      bool warning = true);
      template<class ElemType>
      bool updateCell(long iRow, long iCol, const ElemType &val,
                      bool warning = true);
      template<class ElemType>
      bool updateCell(long idx, const ElemType &val,
                      bool warning = true);

      template<class InputType>
      bool updateCellAs(const mcPRL::CellCoord &coord, const InputType &val,
                        bool warning = true);
      template<class InputType>
      bool updateCellAs(long iRow, long iCol, const InputType &val,
                        bool warning = true);
      template<class InputType>
      bool updateCellAs(long idx, const InputType &val,
                        bool warning = true);

      template<class ElemType>
      bool updateCells(const vector< pair<mcPRL::CellCoord, ElemType> > &vCells,
                       bool warning = true);
      template<class ElemType>
      bool updateCells(const vector< pair<long, ElemType> > &vCells,
                       bool warning = true);
      template<class ElemType>
      bool updateCells(const vector< pair< ElemType, vector<long> > > &vCells,
                       bool warning = true);

      template<class InputType>
      bool updateCellsAs(const vector< pair<mcPRL::CellCoord, InputType> > &vCells,
                         bool warning = true);
      template<class InputType>
      bool updateCellsAs(const vector< pair<long, InputType> > &vCells,
                         bool warning = true);
      template<class InputType>
      bool updateCellsAs(const vector< pair< InputType, vector<long> > > &vCells,
                         bool warning = true);
      
      /*
      template<class ElemType>
      bool update(Transition *pTrans,
                  const CoordBR *pWorkBR = NULL);
      */

      bool createGdalDS(GDALDataset *&pDataset,
                        const char *aFileName,
                        GDALDriver *pGdalDriver,
                        char **aGdalOptions = NULL) const;
      bool createGdalDS(GDALDataset *&pDataset,
                        const char *aFileName,
                        const char *aGdalFormat,
                        char **aGdalOptions = NULL) const;
      bool readGDALData(GDALDataset *pDataset,
                        int iBand = 1,
                        const mcPRL::CellCoord *pGDALOff = NULL,
                        bool warning = true);
	  bool creatRandData( const mcPRL::CellCoord *pGDALOff = NULL,
                        bool warning = true);
      bool writeGDALData(GDALDataset *pDataset,
                         int iBand = 1,
                         const mcPRL::CellCoord *pGDALOff = NULL,
                         const mcPRL::CoordBR *pWorkBR = NULL,
                         bool warning = true);
	  /*pgtiol--------------*/
	   bool createPgtiolDS(PGTIOLDataset *&pDataset,
                         const char *aFileName,
                         char **aPgtiolOptions = NULL) const;
      bool readPGTIOLData(PGTIOLDataset *pDataset,
                          int iBand = 1,
                          const mcPRL::CellCoord *pGDALOff = NULL,
                          bool warning = true);
      bool writePGTIOLData(PGTIOLDataset *pDataset,
                           int iBand = 1,
                           const mcPRL::CellCoord *pGDALOff = NULL,
                           const mcPRL::CoordBR *pWorkBR = NULL,
                           bool warning = true);


    protected:
	  /*-------GDAL--------------*/
      bool _checkGDALSettings(GDALDataset *pDataset,
                              int iBand,
                              const mcPRL::CellCoord *pGDALOff = NULL,
                              const mcPRL::CoordBR *pWorkBR = NULL,
                              bool warning = true) const;

	  /*-------PGTIOL--------------*/
      bool _checkPGTIOLSettings(PGTIOLDataset *pDataset,
                              int iBand,
                              const mcPRL::CellCoord *pGDALOff = NULL,
                              const mcPRL::CoordBR *pWorkBR = NULL,
                              bool warning = true) const;

    protected:
      mcPRL::CellspaceInfo *_pInfo;
      void *_pData;
	  void *_dpData;
      bool _trackUpdt;
      mcPRL::LongVect _vUpdtIdxs;

    friend class DataManager;
  };
};

/****************************************************
*                 Public Methods                    *
****************************************************/
template <class ElemType>
bool mcPRL::Cellspace::
initVal(const ElemType *pVal) {
  bool done = true;
  if(isEmpty(true) ||
     !(_pInfo->isDataType<ElemType>(true))) {
    done = false;
  }
  else {
    long mySize = _pInfo->size();
    ElemType val = (pVal == NULL)?_pInfo->getNoDataVal<ElemType>(true):*pVal;
    for(long iElem = 0; iElem < mySize; iElem++) {
      (ElemType&)(*((ElemType*)_pData + iElem)) = val;
    }
  }
  return done;
}

template<class InputType>
bool mcPRL::Cellspace::
initValAs(const InputType *pVal,
          bool warning) {
  bool done = true;
  if(isEmpty(true)) {
    done = false;
  }
  else {
    InputType val = (pVal == NULL) ? _pInfo->getNoDataValAs<InputType>(true) : *pVal;
    long mySize = _pInfo->size();
    for(long iElem = 0; iElem < mySize; iElem++) {
      if(_pInfo->isDataType<bool>()) {
        (bool&)(*((bool*)_pData + iElem)) = (bool)val;
      }
      else if(_pInfo->isDataType<char>()) {
        (char&)(*((char*)_pData + iElem)) = (char)val;
      }
      else if(_pInfo->isDataType<unsigned short>()) {
        (unsigned short&)(*((unsigned short*)_pData + iElem)) = (unsigned short)val;
      }
      else if(_pInfo->isDataType<short>()) {
        (short&)(*((short*)_pData + iElem)) = (short)val;
      }
      else if(_pInfo->isDataType<unsigned int>()) {
        (unsigned int&)(*((unsigned int*)_pData + iElem)) = (unsigned int)val;
      }
      else if(_pInfo->isDataType<int>()) {
        (int&)(*((int*)_pData + iElem)) = (int)val;
      }
      else if(_pInfo->isDataType<unsigned long>()) {
        (unsigned long&)(*((unsigned long*)_pData + iElem)) = (unsigned long)val;
      }
      else if(_pInfo->isDataType<long>()) {
        (long&)(*((long*)_pData + iElem)) = (long)val;
      }
      else if(_pInfo->isDataType<float>()) {
        (float&)(*((float*)_pData + iElem)) = (float)val;
      }
      else if(_pInfo->isDataType<double>()) {
        (double&)(*((double*)_pData + iElem)) = (double)val;
      }
      else if(_pInfo->isDataType<long double>()) {
        (long double&)(*((long double*)_pData + iElem)) = (long double)val;
      }
      else {
        if(warning) {
          cerr << __FILE__ << " function:" << __FUNCTION__ \
              << " Error: unable to convert input data type (" \
              << typeid(InputType).name() << ") to Cellspace data type (" \
              << _pInfo->dataType() << ")" << endl;
          done = false;
        }
      }
    } // end -- for(iElem)
  }
  return done;
}

template<class RtrType>
RtrType* mcPRL::Cellspace::
row(long iRow,
    bool warning) {
  RtrType* pRowElems = NULL;
  if(!isEmpty(warning) && _pInfo->isDataType<RtrType>(warning)) {
    if(iRow < 0 || iRow >= _pInfo->nRows()) {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: row index [" \
             << iRow << "] out of Cellspace boundary [" \
             << _pInfo->dims() << "]" << endl;
      }
    }
    else {
      pRowElems = (RtrType*)_pData + iRow*(_pInfo->nCols());
    }
  }
  return pRowElems;
}

template<class RtrType>
const RtrType* mcPRL::Cellspace::
row(long iRow,
    bool warning) const {
  RtrType* pRowElems = NULL;
  if(!isEmpty(warning) && _pInfo->isDataType<RtrType>(warning)) {
    if(iRow < 0 || iRow >= _pInfo->nRows()) {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: row index [" \
             << iRow << "] out of Cellspace boundary [" \
             << _pInfo->dims() << "]" << endl;
      }
    }
    else {
      pRowElems = (RtrType*)_pData + iRow*(_pInfo->nCols());
    }
  }
  return pRowElems;
}


template <class RtrType> 
RtrType& mcPRL::Cellspace::
at(const mcPRL::CellCoord& coord,
   bool warning) {
  RtrType* pElem = NULL;
  if(!isEmpty(warning) &&
      _pInfo->isDataType<RtrType>(warning) &&
      _pInfo->validCoord(coord, warning)) {
    pElem = (RtrType*)_pData + coord.iRow()*(_pInfo->nCols()) + coord.iCol();
  }
  return *pElem;
}


template <class RtrType> 
const RtrType& mcPRL::Cellspace::
at(const mcPRL::CellCoord& coord,
   bool warning) const {
  RtrType* pElem = NULL;
  if(!isEmpty(warning) &&
     _pInfo->isDataType<RtrType>(warning) &&
     _pInfo->validCoord(coord, warning)) {
    pElem = (RtrType*)_pData + coord.iRow()*(_pInfo->nCols()) + coord.iCol();
  }
  return *pElem;
}


template <class RtrType> 
RtrType& mcPRL::Cellspace::
at(long iRow, long iCol,
   bool warning) {
  RtrType* pElem = NULL;
  if(!isEmpty(warning) &&
     _pInfo->isDataType<RtrType>(warning) &&
     _pInfo->validCoord(iRow, iCol, warning)) {
    pElem = (RtrType*)_pData + iRow*(_pInfo->nCols()) + iCol;
  }
  return *pElem;
}





template <class RtrType> 
RtrType& mcPRL::Cellspace::
at(long idx,
   bool warning) {
  RtrType* pElem = NULL;
  if(!isEmpty(warning) &&
     _pInfo->isDataType<RtrType>(warning) &&
     _pInfo->validIdx(idx, warning)) {
    pElem = (RtrType*)_pData + idx;
  }
  return *pElem;
}


template <class RtrType> 
const RtrType& mcPRL::Cellspace::
at(long idx,
   bool warning) const {
  RtrType *pElem = NULL;
  if(!isEmpty(warning) &&
     _pInfo->isDataType<RtrType>(warning) &&
     _pInfo->validIdx(idx, warning)) {
    pElem = (RtrType*)_pData + idx;
  }
  return *pElem;
}

template<class RtrType>
RtrType mcPRL::Cellspace::
atAs(const mcPRL::CellCoord &coord,
     bool warning) const {
  return atAs<RtrType>(_pInfo->coord2idx(coord), warning);
}

template<class RtrType>
RtrType mcPRL::Cellspace::
atAs(long iRow, long iCol,
     bool warning) const {
  return atAs<RtrType>(_pInfo->coord2idx(iRow, iCol), warning);
}

template<class RtrType>
RtrType mcPRL::Cellspace::
atAs(long idx,
     bool warning) const {
  RtrType val = (RtrType)mcPRL::ERROR_VAL;
  if(!isEmpty(warning) && _pInfo->validIdx(idx, warning)) {
    if(_pInfo->isDataType<bool>()) {
      val = (RtrType)(at<bool>(idx));
    }
    else if(_pInfo->isDataType<unsigned char>()) {
      val = (RtrType)(at<unsigned char>(idx));
    }
    else if(_pInfo->isDataType<char>()) {
      val = (RtrType)(at<char>(idx));
    }
    else if(_pInfo->isDataType<unsigned short>()) {
      val = (RtrType)(at<unsigned short>(idx));
    }
    else if(_pInfo->isDataType<short>()) {
      val = (RtrType)(at<short>(idx));
    }
    else if(_pInfo->isDataType<unsigned int>()) {
      val = (RtrType)(at<unsigned int>(idx));
    }
    else if(_pInfo->isDataType<int>()) {
      val = (RtrType)(at<int>(idx));
    }
    else if(_pInfo->isDataType<unsigned long>()) {
      val = (RtrType)(at<unsigned long>(idx));
    }
    else if(_pInfo->isDataType<long>()) {
      val = (RtrType)(at<long>(idx));
    }
    else if(_pInfo->isDataType<float>()) {
      val = (RtrType)(at<float>(idx));
    }
    else if(_pInfo->isDataType<double>()) {
      val = (RtrType)(at<double>(idx));
    }
    else if(_pInfo->isDataType<long double>()) {
      val = (RtrType)(at<long double>(idx));
    }
    else {
      if(warning) {
        cerr << __FILE__ << " function:" << __FUNCTION__ \
             << " Error: unable to convert Cellspace data type (" \
             << _pInfo->dataType() << ") to output data type (" << typeid(RtrType).name() \
             << "). Returned an INVALID value. "<< endl;
      }
    }
  }
  return val;
}

template<class RtrType>
bool mcPRL::Cellspace::
values(vector<RtrType> *pvVals,
       const mcPRL::CoordBR *pRectangle,
       bool warning) const {
  bool done = true;
  if(pvVals == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the vector of values" << endl;
    }
    done = false;
  }
  else {
    pvVals->clear();
    if(!_pInfo->isDataType<RtrType>(warning)) {
      done = false;
    }
    else if(!isEmpty(warning)) {
      if(pRectangle == NULL) {
        long mySize = _pInfo->size();
        for(long iElem = 0; iElem < mySize; iElem++) {
          const RtrType &val = *((const RtrType*)_pData + iElem);
          if(std::find(pvVals->begin(), pvVals->end(), val) == pvVals->end()) {
            pvVals->push_back(val);
          }
        } // End of iElem loop
      } // if(pRectangle == NULL)
      else if(_pInfo->validBR(*pRectangle, warning)) {
        for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
          for(long iCol = pRectangle->minICol(); iCol <= pRectangle->maxICol(); iCol++) {
            long iElem = _pInfo->coord2idx(iRow, iCol);
            const RtrType &val = *((const RtrType*)_pData + iElem);
            if(std::find(pvVals->begin(), pvVals->end(), val) == pvVals->end()) {
              pvVals->push_back(val);
            }
          }
        }
      }
      stable_sort(pvVals->begin(), pvVals->end());
    }
  }
  return done;
}

template<class RtrType>
bool mcPRL::Cellspace::
valuesAs(vector<RtrType> *pvVals,
         const mcPRL::CoordBR *pRectangle,
         bool warning) const {
  if(pvVals == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the vector of values" << endl;
    }
    return false;
  }
  pvVals->clear();

  if(isEmpty(warning)) {
    return false;
  }

  if(pRectangle == NULL) {
    long mySize = _pInfo->size();
    for(long iElem = 0; iElem < mySize; iElem++) {
      RtrType val = atAs<RtrType>(iElem, warning);
      if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
        return false;
      }
      else if(std::find(pvVals->begin(), pvVals->end(), val) == pvVals->end()) {
        pvVals->push_back(val);
      }
    } // End of iElem loop
  }
  else if(_pInfo->validBR(*pRectangle, warning)) {
    for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
      for(long iCol = pRectangle->minICol(); iCol <= pRectangle->maxICol(); iCol++) {
        long iElem = _pInfo->coord2idx(iRow, iCol);
        RtrType val = atAs<RtrType>(iElem, warning);
        if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
          return false;
        }
        if(std::find(pvVals->begin(), pvVals->end(), val) == pvVals->end()) {
          pvVals->push_back(val);
        }
      }
    }
  }
  else {
    return false;
  }

  stable_sort(pvVals->begin(), pvVals->end());

  return true;
}

template<class ElemType>
bool mcPRL::Cellspace::
find(LongVect *pvFoundIdxs,
     const ElemType *pVal,
     const mcPRL::CoordBR *pRectangle,
     bool warning) const {
  bool found = false;
  if(pvFoundIdxs == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the vector of found indices" << endl;
    }
    found = false;
  }
  else {
    pvFoundIdxs->clear();
    if(_pInfo->isDataType<ElemType>(warning) && !isEmpty(warning)) {
      if(pRectangle == NULL) {
        long mySize = _pInfo->size();
        const ElemType *pElem = (const ElemType*)_pData;
        while(pElem != (const ElemType*)_pData + mySize) {
          pElem = (const ElemType*)std::find(pElem, (const ElemType*)_pData+mySize, *pVal);
          if(pElem != (const ElemType*)_pData + mySize) {
            pvFoundIdxs->push_back(pElem - (const ElemType*)_pData);
            if(!found) {
              found = true;
            }
            pElem++;
          }
        } // End of while loop
      }
      else if(_pInfo->validBR(*pRectangle, warning)){
        long nRecCols = pRectangle->nCols();
        for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
          const ElemType *aRow = (const ElemType*)_pData + iRow*(_pInfo->nCols());
          const ElemType *pElem = aRow+pRectangle->minICol();
          while(pElem != aRow + pRectangle->maxICol() + 1) {
            pElem = std::find(pElem, aRow+pRectangle->maxICol()+1, *pVal);
            if(pElem != aRow + pRectangle->maxICol() + 1) {
              long idx = _pInfo->coord2idx(iRow, pElem-aRow);
              pvFoundIdxs->push_back(idx);
              if(!found) {
                found = true;
              }
              pElem++;
            }
          } // End of while loop
        }
      }
    }
  }
  return found;
}


template<class ElemType>
bool mcPRL::Cellspace::
find(mcPRL::LongVect *pvFoundIdxs,
     const vector<ElemType> *pvVals,
     const mcPRL::CoordBR *pRectangle,
     bool warning) const {
  bool found = false;
  if(pvFoundIdxs == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the vector of found indices" << endl;
    }
    found = false;
  }
  else if(pvVals == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the vector of values" << endl;
    }
    found = false;
  }
  else {
    pvFoundIdxs->clear();
    if(_pInfo->isDataType<ElemType>(warning) && !isEmpty(warning)) {
      if(pRectangle == NULL) {
        long mySize = _pInfo->size();
        for(long iElem = 0; iElem < mySize; iElem++) {
          const ElemType &elemVal = *((const ElemType*)_pData + iElem);
          if(std::find(pvVals->begin(), pvVals->end(), elemVal) != pvVals->end()) {
            pvFoundIdxs->push_back(iElem);
            if(!found) {
              found = true;
            }
          }
        }
      }
      else if(_pInfo->validBR(*pRectangle, warning)){
        for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
          for(long iCol = pRectangle->minICol(); iCol <= pRectangle->maxICol(); iCol++) {
            long iElem = _pInfo->coord2idx(iRow, iCol);
            const ElemType& elemVal = *((const ElemType*)_pData + iElem);
            if(std::find(pvVals->begin(), pvVals->end(), elemVal) != pvVals->end()) {
              pvFoundIdxs->push_back(iElem);
              if(!found) {
                found = true;
              }
            }
          }
        }
      }
    }
  }
  return found;
}


template<class ElemType, class Predicate>
bool mcPRL::Cellspace::
find(LongVect *pvFoundIdxs,
     Predicate pred,
     const mcPRL::CoordBR *pRectangle,
     bool excldNoData,
     bool warning) const {
  bool found = false;
  if(pvFoundIdxs == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: NULL pointer to the vector of found indices" << endl;
    found = false;
  }
  else {
    pvFoundIdxs->clear();
    if(_pInfo->isDataType<ElemType>(warning) && !isEmpty(warning)) {
      if(pRectangle == NULL) {
        long mySize = _pInfo->size();
        const ElemType *pElem = (const ElemType*)_pData;
        while(pElem != (const ElemType*)_pData + mySize) {
          pElem = (const ElemType*)std::find_if(pElem, (const ElemType*)_pData+mySize, pred);
          if(pElem != (const ElemType*)_pData + mySize) {
            if(excldNoData &&
               fabs(*pElem - _pInfo->getNoDataValAs<ElemType>()) < mcPRL::EPSINON) {
              continue;
            }
            else {
              pvFoundIdxs->push_back(pElem - (const ElemType*)_pData);
              if(!found) {
                found = true;
              }
            }
            pElem++;
          }
        } // End of while loop
      }
      else if(_pInfo->validBR(*pRectangle, warning)) {
        for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
          const ElemType* aRow = (const ElemType*)_pData + iRow*(_pInfo->nCols());
          const ElemType* pElem = aRow+pRectangle->minICol();
          while(pElem != aRow + pRectangle->maxICol() + 1) {
            pElem = std::find_if(pElem, aRow+pRectangle->maxICol()+1, pred);
            if(pElem != aRow + pRectangle->maxICol() + 1) {
              if(excldNoData &&
                 fabs(*pElem - _pInfo->getNoDataValAs<ElemType>()) < mcPRL::EPSINON) {
                continue;
              }
              else {
                long idx = _pInfo->coord2idx(iRow, pElem-aRow);
                pvFoundIdxs->push_back(idx);
                if(!found) {
                  found = true;
                }
              }
              pElem++;
            }
          } // End of while loop
        }
      }
    }
  }
  return found;
}

template<class ElemType>
long mcPRL::Cellspace::
count(const ElemType *pVal,
      const mcPRL::CoordBR *pRectangle,
      bool warning) const {
  long num = 0;

  if(pVal == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the value to count" << endl;
    }
  }
  else if(_pInfo->isDataType<ElemType>(warning) && !isEmpty(warning)) {
    if(pRectangle == NULL) {
      num = std::count((const ElemType*)_pData, (const ElemType*)_pData+_pInfo->size(), *pVal);
    }
    else if(_pInfo->validBR(*pRectangle, true)) {
      for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
        long iColStart = pRectangle->minICol();
        long iColEnd = pRectangle->maxICol();
        long iElemStart = _pInfo->coord2idx(iRow, iColStart);
        long iElemEnd = _pInfo->coord2idx(iRow, iColEnd) + 1;
        num += std::count((const ElemType*)_pData+iElemStart, (const ElemType*)_pData+iElemEnd, *pVal);
      } // End of iRow loop
    }
  }

  return num;
}

template<class ElemType>
long mcPRL::Cellspace::
count(const vector<ElemType> *pvVals,
      const mcPRL::CoordBR *pRectangle,
      bool warning) const {
  long num = 0;
  if(pvVals == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the vector of values" << endl;
    }
  }
  else if(_pInfo->isDataType<ElemType>(warning) && !isEmpty(warning)) {
    if(pRectangle == NULL) {
      long mySize = _pInfo->size();
      for(long iElem = 0; iElem < mySize; iElem++) {
        const ElemType& elemVal = *((const ElemType*)_pData + iElem);
        if(std::find(pvVals->begin(), pvVals->end(), elemVal) != pvVals->end()) {
          num++;
        }
      }
    }
    else if(_pInfo->validBR(*pRectangle, warning)) {
      for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
        for(long iCol = pRectangle->minICol(); iCol <= pRectangle->maxICol(); iCol++) {
          long iElem = _pInfo->coord2idx(iRow, iCol);
          const ElemType& elemVal = *((const ElemType*)_pData + iElem);
          if(std::find(pvVals->begin(), pvVals->end(), elemVal) != pvVals->end()) {
            num++;
          }
        }
      } // End of iRow loop
    }
  }
  return num;
}

template<class ElemType, class Predicate>
long mcPRL::Cellspace::
count(Predicate pred,
      const mcPRL::CoordBR *pRectangle,
      bool excldNoData,
      bool warning) const {
  long num = 0;
  mcPRL::LongVect vFoundIdxs;
  if(find(&vFoundIdxs, pred, pRectangle, excldNoData, warning)) {
    num = vFoundIdxs.size();
  }
  return num;
}

template<class ElemType>
bool mcPRL::Cellspace::
frequences(map<ElemType, long> *pmFreqs,
           const mcPRL::CoordBR *pRectangle,
           bool warning) const {
  if(pmFreqs == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the map of frequences" << endl;
    }
    return false;
  }
  pmFreqs->clear();

  if(_pInfo->isDataType<ElemType>(warning) && !isEmpty(warning)) {
    if(pRectangle == NULL) {
      long mySize = _pInfo->size();
      for(long iElem = 0; iElem < mySize; iElem++) {
        const ElemType &val = *((const ElemType*)_pData + iElem);
        pmFreqs->operator[](val)++; // may include NoData
      } // End of iElem loop
    } // if(pRectangle == NULL)
    else if(_pInfo->validBR(*pRectangle, warning)) {
      for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
        for(long iCol = pRectangle->minICol(); iCol <= pRectangle->maxICol(); iCol++) {
          long iElem = _pInfo->coord2idx(iRow, iCol);
          const ElemType &val = *((const ElemType*)_pData + iElem);
          pmFreqs->operator[](val)++; // may include NoData
        }
      }
    }
  }
  return true;
}

template<class RtrType>
bool mcPRL::Cellspace::
frequencesAs(map<RtrType, long> *pmFreqs,
             const mcPRL::CoordBR *pRectangle,
             bool warning) const {
  if(pmFreqs == NULL) {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL pointer to the map of frequences" << endl;
    }
    return false;
  }
  pmFreqs->clear();

  if(isEmpty(warning)) {
    return false;
  }

  if(pRectangle == NULL) {
    long mySize = _pInfo->size();
    for(long iElem = 0; iElem < mySize; iElem++) {
      RtrType val = atAs<RtrType>(iElem, warning);
      if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
        return false;
      }
      pmFreqs->operator[](val)++; // may include NoData
    } // End of iElem loop
  }
  else if(_pInfo->validBR(*pRectangle, warning)) {
    for(long iRow = pRectangle->minIRow(); iRow <= pRectangle->maxIRow(); iRow++) {
      for(long iCol = pRectangle->minICol(); iCol <= pRectangle->maxICol(); iCol++) {
        long iElem = _pInfo->coord2idx(iRow, iCol);
        RtrType val = atAs<RtrType>(iElem, warning);
        if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
          return false;
        }
        pmFreqs->operator[](val)++; // may include NoData
      }
    }
  }
  else {
    return false;
  }

  return true;
}

                         
template<class RtrType>
bool mcPRL::Cellspace::
nbrhdVals(vector<RtrType>& vVals,
          const mcPRL::Neighborhood& nbrhd,
          const mcPRL::CellCoord& ctrCoord,
          bool includeCtr,
          bool warning) const {
  vVals.clear();
  if(_pInfo->isDataType<RtrType>(warning) &&
     !isEmpty(warning) &&
     _pInfo->validNbrhd(nbrhd, ctrCoord, warning)) {
    for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
      if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
        continue;
      }
      mcPRL::CellCoord nbrCoord(ctrCoord+nbrhd[iNbr]);
      if(nbrCoord.valid(_pInfo->dims())) {
        const RtrType& val = *((const RtrType*)_pData+nbrCoord.iRow()*(_pInfo->nCols())+nbrCoord.iCol());
        vVals.push_back(val); // may include NoData
      }
      else switch(nbrhd.edgeOption()) {
        case mcPRL::NODATA_VIRTUAL_EDGES:
          vVals.push_back(_pInfo->getNoDataVal<RtrType>());
          break;
        case mcPRL::CUSTOM_VIRTUAL_EDGES:
          if(nbrhd.virtualEdgeVal() != NULL) {
            vVals.push_back(RtrType(*(nbrhd.virtualEdgeVal())));
          }
          else {
            if(warning) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                   << " Error: customized virtual edge value has not been set" \
                   << endl;
            }
            return false;
          }
          break;
        default:
          break;
      }
    }
  }
  else {
    return false;
  }

  return true;
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdVals(vector<RtrType>& vVals,
          const mcPRL::Neighborhood& nbrhd,
          long ctrRow, long ctrCol,
          bool includeCtr,
          bool warning) const {
  return nbrhdVals<RtrType>(vVals, nbrhd, mcPRL::CellCoord(ctrRow, ctrCol), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdVals(vector<RtrType>& vVals,
              const mcPRL::Neighborhood& nbrhd,
              long idx,
              bool includeCtr,
              bool warning) const {
  return nbrhdVals<RtrType>(vVals, nbrhd, _pInfo->idx2coord(idx), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdValsAs(vector<RtrType> &vVals,
            const mcPRL::Neighborhood &nbrhd,
            const mcPRL::CellCoord &ctrCoord,
            bool includeCtr,
            bool warning) const {
  vVals.clear();
  if(isEmpty(warning) ||
     !(_pInfo->validNbrhd(nbrhd, ctrCoord, warning))) {
    return false;
  }
  for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
    if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
      continue;
    }
    mcPRL::CellCoord nbrCoord(ctrCoord+nbrhd[iNbr]);
    if(nbrCoord.valid(_pInfo->dims())) {
      RtrType val = atAs<RtrType>(nbrCoord, warning);
      /*
      if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
        return false;
      }
      */
      vVals.push_back(val); // may include NoData
    }
    else switch(nbrhd.edgeOption()) {
      case mcPRL::NODATA_VIRTUAL_EDGES:
        vVals.push_back(_pInfo->getNoDataValAs<RtrType>());
        break;
      case mcPRL::CUSTOM_VIRTUAL_EDGES:
        if(nbrhd.virtualEdgeVal() != NULL) {
          vVals.push_back(RtrType(*(nbrhd.virtualEdgeVal())));
        }
        else {
          if(warning) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                 << " Error: customized virtual edge value has not been set" \
                 << endl;
          }
          return false;
        }
        break;
      default:
        break;
    }
  }

  return true;
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdValsAs(vector<RtrType> &vVals,
            const mcPRL::Neighborhood &nbrhd,
            long ctrRow, long ctrCol,
            bool includeCtr,
            bool warning) const {
  return nbrhdValsAs<RtrType>(vVals, nbrhd, mcPRL::CellCoord(ctrRow, ctrCol), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdValsAs(vector<RtrType> &vVals,
            const mcPRL::Neighborhood &nbrhd,
            long idx,
            bool includeCtr,
            bool warning) const {
  return nbrhdValsAs<RtrType>(vVals, nbrhd, _pInfo->idx2coord(idx), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdFreqs(map<RtrType, int> &mFreqs,
           const mcPRL::Neighborhood &nbrhd,
           const mcPRL::CellCoord &ctrCoord,
           bool includeCtr,
           bool warning) const {
  mFreqs.clear();
  if(_pInfo->isDataType<RtrType>(warning) &&
     !isEmpty(warning) &&
     _pInfo->validNbrhd(nbrhd, ctrCoord, warning)) {
    for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
      if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
        continue;
      }
      mcPRL::CellCoord nbrCoord(ctrCoord+nbrhd[iNbr]);
      if(nbrCoord.valid(_pInfo->dims())) {
        const RtrType &val = *((const RtrType*)_pData+nbrCoord.iRow()*(_pInfo->nCols())+nbrCoord.iCol());
        mFreqs[val]++; // may include NoData
      }
      else switch(nbrhd.edgeOption()) {
        case mcPRL::NODATA_VIRTUAL_EDGES:
          mFreqs[_pInfo->getNoDataVal<RtrType>()]++;
          break;
        case mcPRL::CUSTOM_VIRTUAL_EDGES:
          if(nbrhd.virtualEdgeVal() != NULL) {
            mFreqs[RtrType(*(nbrhd.virtualEdgeVal()))]++;
          }
          else {
            if(warning) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                   << " Error: customized virtual edge value has not been set" \
                   << endl;
            }
            return false;
          }
          break;
        default:
          break;
      }
    }
  }
  else {
    return false;
  }

  return true;
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdFreqs(map<RtrType, int> &mFreqs,
           const mcPRL::Neighborhood &nbrhd,
           long ctrRow, long ctrCol,
           bool includeCtr,
           bool warning) const {
  return nbrhdFreqs<RtrType>(mFreqs, nbrhd, mcPRL::CellCoord(ctrRow, ctrCol), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdFreqs(map<RtrType, int> &mFreqs,
           const mcPRL::Neighborhood &nbrhd,
           long idx,
           bool includeCtr,
           bool warning) const {
  return nbrhdFreqs<RtrType>(mFreqs, nbrhd, _pInfo->idx2coord(idx), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
wnbrhdFreqs(map<RtrType, double> &mFreqs,
            const mcPRL::Neighborhood &nbrhd,
            const mcPRL::CellCoord &ctrCoord,
            bool includeCtr,
            bool warning) const {
  mFreqs.clear();
  if(_pInfo->isDataType<RtrType>(warning) &&
      !isEmpty(warning) &&
      _pInfo->validNbrhd(nbrhd, ctrCoord, warning)) {
    for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
      if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
        continue;
      }
      mcPRL::CellCoord nbrCoord(ctrCoord+nbrhd[iNbr]);
      if(nbrCoord.valid(_pInfo->dims())) {
        const RtrType &val = *((const RtrType*)_pData+nbrCoord.iRow()*(_pInfo->nCols())+nbrCoord.iCol());
        mFreqs[val] += nbrhd[iNbr].weight(); // may include NoData
      }
      else switch(nbrhd.edgeOption()){
        case mcPRL::NODATA_VIRTUAL_EDGES:
          mFreqs[_pInfo->getNoDataVal<RtrType>()] += nbrhd[iNbr].weight();
          break;
        case mcPRL::CUSTOM_VIRTUAL_EDGES:
          if(nbrhd.virtualEdgeVal() != NULL) {
            mFreqs[RtrType(*(nbrhd.virtualEdgeVal()))] += nbrhd[iNbr].weight();
          }
          else {
            if(warning) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                   << " Error: customized virtual edge value has not been set" \
                   << endl;
            }
            return false;
          }
          break;
        default:
          break;
      }
    }
  }
  else {
    return false;
  }
  return true;
}

template<class RtrType>
bool mcPRL::Cellspace::
wnbrhdFreqs(map<RtrType, double> &mFreqs,
            const mcPRL::Neighborhood &nbrhd,
            long ctrRow, long ctrCol,
            bool includeCtr,
            bool warning) const {
  return wnbrhdFreqs<RtrType>(mFreqs, nbrhd, mcPRL::CellCoord(ctrRow, ctrCol), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
wnbrhdFreqs(map<RtrType, double> &mFreqs,
            const mcPRL::Neighborhood &nbrhd,
            long idx,
            bool includeCtr,
            bool warning) const {
  return wnbrhdFreqs<RtrType>(mFreqs, nbrhd, _pInfo->idx2coord(idx), includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
nbrhdFreqsAs(map<RtrType, int> &mFreqs,
             const mcPRL::Neighborhood &nbrhd,
             const CellCoord &ctrCoord,
             bool includeCtr,
             bool warning) const {
  mFreqs.clear();
  if(isEmpty(warning) ||
      !(_pInfo->validNbrhd(nbrhd, ctrCoord, warning))) {
    return false;
  }
  for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
    if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
      continue;
    }
    mcPRL::CellCoord nbrCoord(ctrCoord+nbrhd[iNbr]);
    if(nbrCoord.valid(_pInfo->dims())) {
      RtrType val = atAs<RtrType>(nbrCoord, warning);
      /*
      if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
        return false;
      }
      */
      mFreqs[val]++; // may include NoData
    }
    else switch(nbrhd.edgeOption()){
      case mcPRL::NODATA_VIRTUAL_EDGES:
        mFreqs[_pInfo->getNoDataValAs<RtrType>()]++;
        break;
      case mcPRL::CUSTOM_VIRTUAL_EDGES:
        if(nbrhd.virtualEdgeVal() != NULL) {
          mFreqs[RtrType(*(nbrhd.virtualEdgeVal()))]++;
        }
        else {
          if(warning) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                << " Error: customized virtual edge value has not been set" \
                << endl;
          }
          return false;
        }
        break;
      default:
        break;
    }
  }
  return true;
}


template<class RtrType>
bool mcPRL::Cellspace::
nbrhdFreqsAs(map<RtrType, int> &mFreqs,
             const mcPRL::Neighborhood &nbrhd,
             long ctrRow, long ctrCol,
             bool includeCtr,
             bool warning) const {
  return nbrhdFreqsAs<RtrType>(mFreqs, nbrhd,
                               mcPRL::CellCoord(ctrRow, ctrCol),
                               includeCtr, warning);
}


template<class RtrType>
bool mcPRL::Cellspace::
nbrhdFreqsAs(map<RtrType, int> &mFreqs,
             const mcPRL::Neighborhood &nbrhd,
             long idx,
             bool includeCtr,
             bool warning) const {
  return nbrhdFreqsAs<RtrType>(mFreqs, nbrhd,
                               _pInfo->idx2coord(idx),
                               includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
wnbrhdFreqsAs(map<RtrType, double> &mFreqs,
              const mcPRL::Neighborhood &nbrhd,
              const mcPRL::CellCoord &ctrCoord,
              bool includeCtr,
              bool warning) const {
  mFreqs.clear();
  if(isEmpty(warning) ||
     !(_pInfo->validNbrhd(nbrhd, ctrCoord, warning))) {
    return false;
  }
  for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
    if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
      continue;
    }
    mcPRL::CellCoord nbrCoord(ctrCoord+nbrhd[iNbr]);
    if(nbrCoord.valid(_pInfo->dims())) {
      RtrType val = atAs<RtrType>(nbrCoord, warning);
      /*
      if(fabs(val - (RtrType)mcPRL::ERROR_VAL) < mcPRL::EPSINON) {
        return false;
      }
      */
      mFreqs[val] += nbrhd[iNbr].weight(); // may include NoData
    }
    else switch(nbrhd.edgeOption()){
      case mcPRL::NODATA_VIRTUAL_EDGES:
        mFreqs[_pInfo->getNoDataValAs<RtrType>()] += nbrhd[iNbr].weight();
        break;
      case mcPRL::CUSTOM_VIRTUAL_EDGES:
        if(nbrhd.virtualEdgeVal() != NULL) {
          mFreqs[RtrType(*(nbrhd.virtualEdgeVal()))] += nbrhd[iNbr].weight();
        }
        else {
          if(warning) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                 << " Error: customized virtual edge value has not been set" \
                 << endl;
          }
          return false;
        }
        break;
      default:
        break;
    }
  }
  return true;
}

template<class RtrType>
bool mcPRL::Cellspace::
wnbrhdFreqsAs(map<RtrType, double> &mFreqs,
              const mcPRL::Neighborhood &nbrhd,
              long ctrRow, long ctrCol,
              bool includeCtr,
              bool warning) const {
  return wnbrhdFreqsAs<RtrType>(mFreqs, nbrhd,
                                mcPRL::CellCoord(ctrRow, ctrCol),
                                includeCtr, warning);
}

template<class RtrType>
bool mcPRL::Cellspace::
wnbrhdFreqsAs(map<RtrType, double> &mFreqs,
              const mcPRL::Neighborhood &nbrhd,
              long idx,
              bool includeCtr,
              bool warning) const {
  return wnbrhdFreqsAs<RtrType>(mFreqs, nbrhd,
                                _pInfo->idx2coord(idx),
                                includeCtr, warning);
}

template<class RtrType>
RtrType mcPRL::Cellspace::
nbrhdTotalVal(const mcPRL::Neighborhood& nbrhd,
              const mcPRL::CellCoord& ctrCoord,
              bool includeCtr,
              bool warning) const {
  RtrType total = RtrType(0.0);
  if(_pInfo->isDataType<RtrType>(warning) &&
     !isEmpty(warning) &&
     _pInfo->validNbrhd(nbrhd, ctrCoord, warning)) {
    for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
      if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
        continue;
      }
      const mcPRL::WeightedCellCoord& wCoord = nbrhd[iNbr];
      mcPRL::CellCoord nbrCoord(ctrCoord+wCoord);
      if(nbrCoord.valid(_pInfo->dims())) {
        const RtrType& val = *((const RtrType*)_pData+nbrCoord.iRow()*(_pInfo->nCols())+nbrCoord.iCol());
        if(fabs(val - _pInfo->getNoDataValAs<RtrType>()) >= mcPRL::EPSINON) {
          total += val * wCoord.weight();
        }
      }
      else if(nbrhd.edgeOption() == mcPRL::CUSTOM_VIRTUAL_EDGES) {
        if(nbrhd.virtualEdgeVal() != NULL) {
          const RtrType val = RtrType(*(nbrhd.virtualEdgeVal()));
          if(fabs(val - _pInfo->getNoDataValAs<RtrType>()) >= mcPRL::EPSINON) {
            total += val * wCoord.weight();
          }
        }
        else if(warning) {
          cerr << __FILE__ << " function:" << __FUNCTION__ \
               << " Error: customized virtual edge value has not been set" \
               << endl;
        }
      }
    }
  }
  return total;
}

template<class RtrType>
RtrType mcPRL::Cellspace::
nbrhdTotalVal(const mcPRL::Neighborhood& nbrhd,
                long ctrRow, long ctrCol,
                bool includeCtr,
                bool warning) const {
  return nbrhdTotalVal<RtrType>(nbrhd, mcPRL::CellCoord(ctrRow, ctrCol),
                                includeCtr, warning);
}

template<class RtrType>
RtrType mcPRL::Cellspace::
nbrhdTotalVal(const mcPRL::Neighborhood& nbrhd,
                long idx,
                bool includeCtr,
                bool warning) const {
  return nbrhdTotalVal<RtrType>(nbrhd, _pInfo->idx2coord(idx),
                                includeCtr, warning);
}

template<class RtrType>
RtrType mcPRL::Cellspace::
nbrhdTotalValAs(const mcPRL::Neighborhood& nbrhd,
                const mcPRL::CellCoord& ctrCoord,
                bool includeCtr,
                bool warning) const {
  RtrType total = RtrType(0.0);
  if(!isEmpty(warning) &&
      _pInfo->validNbrhd(nbrhd, ctrCoord, warning)) {
    for(int iNbr = 0; iNbr < nbrhd.size(); iNbr++) {
      if(nbrhd[iNbr].iRow() == 0 && nbrhd[iNbr].iCol() == 0 && !includeCtr) {
        continue;
      }
      const mcPRL::WeightedCellCoord& wCoord = nbrhd[iNbr];
      mcPRL::CellCoord nbrCoord(ctrCoord+wCoord);
      if(nbrCoord.valid(_pInfo->dims())) {
        RtrType val = atAs<RtrType>(nbrCoord, warning);
        if(fabs(val - _pInfo->getNoDataValAs<RtrType>()) >= mcPRL::EPSINON) {
          total += val * wCoord.weight();
        }
      }
      else if(nbrhd.edgeOption() == mcPRL::CUSTOM_VIRTUAL_EDGES) {
        if(nbrhd.virtualEdgeVal() != NULL) {
          const RtrType val = RtrType(*(nbrhd.virtualEdgeVal()));
          if(fabs(val - _pInfo->getNoDataValAs<RtrType>()) >= mcPRL::EPSINON) {
            total += val * wCoord.weight();
          }
        }
        else if(warning) {
          cerr << __FILE__ << " function:" << __FUNCTION__ \
              << " Error: customized virtual edge value has not been set" \
              << endl;
        }
      }
    }
  }
  return total;
}

template<class RtrType>
RtrType mcPRL::Cellspace::
nbrhdTotalValAs(const mcPRL::Neighborhood& nbrhd,
                  long ctrRow, long ctrCol,
                  bool includeCtr,
                  bool warning) const {
  return nbrhdTotalValAs<RtrType>(nbrhd, mcPRL::CellCoord(ctrRow, ctrCol),
                                  includeCtr, warning);
}

template<class RtrType>
RtrType mcPRL::Cellspace::
nbrhdTotalValAs(const mcPRL::Neighborhood& nbrhd,
                long idx,
                bool includeCtr,
                bool warning) const {
  return nbrhdTotalValAs<RtrType>(nbrhd, _pInfo->idx2coord(idx),
                                  includeCtr, warning);
}

template<class ElemType>
bool mcPRL::Cellspace::
updateCell(const mcPRL::CellCoord &coord, const ElemType &val,
           bool warning) {
  bool done = false;
  if(_pInfo->isDataType<ElemType>(warning) &&
     !isEmpty(warning) &&
     _pInfo->validCoord(coord, warning)) {
    ElemType &cellVal = (ElemType &)(*((ElemType*)_pData + coord.iRow()*(_pInfo->nCols()) + coord.iCol()));

    if(_trackUpdt && cellVal != val) {
      _vUpdtIdxs.push_back(_pInfo->coord2idx(coord));
    }

    cellVal = val;
    done = true;
  }
  return done;
}


template<class ElemType>
bool mcPRL::Cellspace::
updateCell(long iRow, long iCol, const ElemType &val,
           bool warning) {
  return updateCell(mcPRL::CellCoord(iRow, iCol), val, warning);
}


template<class ElemType>
bool mcPRL::Cellspace::
updateCell(long idx, const ElemType &val,
           bool warning) {
  return updateCell(_pInfo->idx2coord(idx), val, warning);
}

template<class InputType>
bool mcPRL::Cellspace::
updateCellAs(const mcPRL::CellCoord &coord, const InputType &val,
             bool warning) {
  bool done = false;
  if(_pInfo->isDataType<bool>()) {
    done = updateCell(coord, (bool)val, warning);
  }
  else if(_pInfo->isDataType<unsigned char>()) {
    done = updateCell(coord, (unsigned char)val, warning);
  }
  else if(_pInfo->isDataType<char>()) {                                                                                    
    done = updateCell(coord, (char)val, warning);
  }
  else if(_pInfo->isDataType<unsigned short>()) {
    done = updateCell(coord, (unsigned short)val, warning);
  }
  else if(_pInfo->isDataType<short>()) {
    done = updateCell(coord, (short)val, warning);
  }
  else if(_pInfo->isDataType<unsigned int>()) {
    done = updateCell(coord, (unsigned int)val, warning);
  }
  else if(_pInfo->isDataType<int>()) {
    done = updateCell(coord, (int)val, warning);
  }
  else if(_pInfo->isDataType<unsigned long>()) {
    done = updateCell(coord, (unsigned long)val, warning);
  }
  else if(_pInfo->isDataType<long>()) {
    done = updateCell(coord, (long)val, warning);
  }
  else if(_pInfo->isDataType<float>()) {
    done = updateCell(coord, (float)val, warning);
  }
  else if(_pInfo->isDataType<double>()) {
    done = updateCell(coord, (double)val, warning);
  }
  else if(_pInfo->isDataType<long double>()) {
    done = updateCell(coord, (long double)val, warning);
  }
  else {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: unable to convert input data type (" \
              << typeid(InputType).name() << ") to Cellspace data type (" \
              << _pInfo->dataType() << ")" << endl;
      done = false;
    }
  }
  return done;
}

template<class InputType>
bool mcPRL::Cellspace::
updateCellAs(long iRow, long iCol, const InputType &val,
             bool warning) {
  return updateCellAs<InputType>(mcPRL::CellCoord(iRow, iCol), val, warning);
}

template<class InputType>
bool mcPRL::Cellspace::
updateCellAs(long idx, const InputType &val,
             bool warning) {
  return updateCellAs<InputType>(_pInfo->idx2coord(idx), val, warning);
}

template<class ElemType>
bool mcPRL::Cellspace::
updateCells(const vector< pair<mcPRL::CellCoord, ElemType> > &vCells,
            bool warning) {
  bool done = true;
  for(size_t iCell = 0; iCell < vCells.size(); iCell++) {
    if(!updateCell(vCells[iCell].first, vCells[iCell].second, warning)) {
      done = false;
    }
  }
  return done;
}

template<class ElemType>
bool mcPRL::Cellspace::
updateCells(const vector< pair<long, ElemType> > &vCells,
            bool warning) {
  bool done = true;
  for(size_t iCell = 0; iCell < vCells.size(); iCell++) {
    if(!updateCell(vCells[iCell].first, vCells[iCell].second, warning)) {
      done = false;
    }
  }
  return done;
}

template<class ElemType>
bool mcPRL::Cellspace::
updateCells(const vector< pair< ElemType, vector<long> > >& vCells,
            bool warning) {
  bool done = true;
  for(size_t iCell = 0; iCell < vCells.size(); iCell++) {
    const ElemType &val = vCells[iCell].first;
    const vector<long> &vIdxs = vCells[iCell].second;
    vector<long>::const_iterator iIdx = vIdxs.begin();
    while(iIdx != vIdxs.end()) {
      const long &idx = *iIdx;
      if(!updateCell(idx, val, warning)) {
        done = false;
      }
      iIdx++;
    }
  }
  return done;
}

template<class InputType>
bool mcPRL::Cellspace::
updateCellsAs(const vector< pair<mcPRL::CellCoord, InputType> > &vCells,
              bool warning) {
  bool done = true;
  for(size_t iCell = 0; iCell < vCells.size(); iCell++) {
    if(!updateCellAs<InputType>(vCells[iCell].first,
                                vCells[iCell].second,
                                warning)) {
      done = false;
    }
  }
  return done;
}

template<class InputType>
bool mcPRL::Cellspace::
updateCellsAs(const vector< pair<long, InputType> > &vCells,
              bool warning) {
  bool done = true;
  for(size_t iCell = 0; iCell < vCells.size(); iCell++) {
    if(!updateCellAs<InputType>(vCells[iCell].first,
                                vCells[iCell].second,
                                warning)) {
      done = false;
    }
  }
  return done;
}

template<class InputType>
bool mcPRL::Cellspace::
brupdateCellAs(const mcPRL::CoordBR &br,void *output)
{
	 bool done = false;
	 for(long iRow = br.minIRow(); iRow <= br.maxIRow(); iRow++) 
	  {
		  for(long iCol = br.minICol(); iCol <= br.maxICol(); iCol++) 
		  {
			  /* done = evaluate(mcPRL::CellCoord(iRow, iCol));
			  if(done == mcPRL::EVAL_FAILED ||
			  done == mcPRL::EVAL_TERMINATED) {
			  return done;
			  }*/

				 done=updateCellAs<InputType>(mcPRL::CellCoord(iRow, iCol),((InputType*)output)[iRow*_pInfo->nCols()+iCol], true);                         //”–Œ Ã‚
				 if(!done)
					 return done;
		  }
	  }
	 return true;
}
template<class InputType>
bool mcPRL::Cellspace::
createrandForCellsAs(long ncols, long nrows)
{
	  std::default_random_engine _randomGen;
     std::uniform_real_distribution<float> *_pRandomDistribution;
	_randomGen.seed(chrono::system_clock::now().time_since_epoch().count());
	_pRandomDistribution = new std::uniform_real_distribution<float>(0.0, 1.0);
	 bool done = true;
	for(int _ncol=0;_ncol<ncols;_ncol++)
	{
		for(int _nrow=0;_nrow<nrows;_nrow++)
		{
			//srand((int)time(0));
		InputType nRand;
				nRand=(InputType) (*_pRandomDistribution)(_randomGen);
			//	cout<<nRand<<endl;
			InputType &cellVal = (InputType &)(*((InputType*)_pData + _nrow*ncols + _ncol));
			 cellVal = nRand;
		}
	}
	return done;
	
}
template <class RtrType> 
const RtrType& mcPRL::Cellspace::
at(long iRow, long iCol,
   bool warning) const {
  RtrType* pElem = NULL;
  if(!isEmpty(warning) &&
     _pInfo->isDataType<RtrType>(warning) &&
     _pInfo->validCoord(iRow, iCol, warning)) {
      pElem = (RtrType*)_pData + iRow*(_pInfo->nCols()) + iCol;
  }
  return *pElem;
}
template<class InputType>
bool mcPRL::Cellspace::
updateCellsAs(const vector< pair< InputType, vector<long> > >& vCells,
              bool warning) {
  bool done = true;
  for(size_t iCell = 0; iCell < vCells.size(); iCell++) {
    const InputType &val = vCells[iCell].first;
    const vector<long> &vIdxs = vCells[iCell].second;
    vector<long>::const_iterator iIdx = vIdxs.begin();
    while(iIdx != vIdxs.end()) {
      const long &idx = *iIdx;
      if(!updateCellAs<InputType>(idx, val, warning)) {
        done = false;
      }
      iIdx++;
    }
  }
  return done;
}



/*
template<class ElemType>
bool mcPRL::Cellspace::
update(Transition *pTrans,
       const CoordBR *pWorkBR) {
  if(!pTrans) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: missing Transition" << endl;
    return false;
  }
  
  Neighborhood *pNbrhd = pTrans->getNbrhood();
  const CoordBR *pWBR = pWorkBR;
  CoordBR workBR;
  if(!pWorkBR) {
    if(!nbrhoodWorkBR(workBR, pNbrhd)) {
      return false;
    }
    pWBR = &workBR;
  }
  if(!pWBR->valid(dims())) {
    cerr << __FILE__ << " function" << __FUNCTION__ \
    << " Error: invalid workBR (" \
    << *pWBR << ")" << endl;
    return false;
  }
  
  vector< pair<long, ElemType> > vUpdtedCells;
  for(int iRow = pWBR->minIRow(); iRow <= pWBR->maxIRow(); iRow++) {
    for(int iCol = pWBR->minICol(); iCol <= pWBR->maxICol(); iCol++) {
      vUpdtedCells.clear();
      CellCoord coord(iRow, iCol);
      if(!pTrans->evaluate(vUpdtedCells, coord)) {
        return false;
      }
      if(!vUpdtedCells.isEmpty()) {
        for(int iCell = 0; iCell < vUpdtedCells.size(); iCell++) {
          if(pTrans->needFinalize()) {
            if(!_addUpdtCell(vUpdtedCells[iCell])) {
              return false;
            }
          }
          else {
            if(validIdx(vUpdtedCells[iCell].first, true)) {
              (ElemType&)(*((const ElemType*)_pData + vUpdtedCells[iCell].first)) = vUpdtedCells[iCell].second;
            }
          }
        }
      }
    } // End of iCol loop
  } // End of iRow loop
  
  return true;
}
*/

#endif
