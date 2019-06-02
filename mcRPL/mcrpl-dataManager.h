#ifndef MCRPL_DATAMANAGER_H
#define MCRPL_DATAMANAGER_H

#include <time.h>
#include "mcrpl-basicTypes.h"
#include "mcrpl-cellspace.h"
#include "mcrpl-layer.h"
#include "mcrpl-neighborhood.h"
#include "mcrpl-transition.h"
#include "mcrpl-smplDcmp.h"
#include "mcrpl-ownershipMap.h"
#include "mcrpl-exchangeMap.h"
#include "mcrpl-process.h"
#include "mcrpl-transferInfo.h"

namespace mcRPL {
  class DataManager {
    public:
      /* Constructor and destructor */
      DataManager();
      ~DataManager();
      
      /* MPI Process */
      bool initMPI(MPI_Comm comm = MPI_COMM_WORLD,
                   bool hasWriter = false);
      void finalizeMPI();
      mcRPL::Process& mpiPrc();

      /* Layers */
      int nLayers() const;
      mcRPL::Layer* addLayer(const char *aLyrName);

      /*ByGDAL----*/
      mcRPL::Layer* addLayerByGDAL(const char *aLyrName,
                                  const char *aGdalFileName,
                                  int iBand,
                                  bool pReading = true);
      bool createLayerGDAL(const char *aLyrName,
                           const char *aGdalFileName,
                           const char *aGdalFormat,
                           char **aGdalOptions = NULL);
      void closeGDAL();

      /*ByPGTIOL--hsj-*/
      mcRPL::Layer* addLayerByPGTIOL(const char *aLyrName,
                                    const char *aPgtiolFileName,
                                    int iBand,
                                    bool pReading = true);
      bool createLayerPGTIOL(const char *aLyrName,
                             const char *aPgtiolFileName,
                             char **aPgtiolOptions = NULL);
      void closePGTIOL();
      void closeDatasets();

      bool rmvLayerByIdx(int lyrID,
                         bool warning = true);
      bool rmvLayerByName(const char *aLyrName,
                          bool warning = true);
      void clearLayers();
      mcRPL::Layer* getLayerByIdx(int lyrID,
                                 bool wrning = true);
      const mcRPL::Layer* getLayerByIdx(int lyrID,
                                       bool wrning = true) const ;
      int getLayerIdxByName(const char *aLyrName,
                            bool warning = true) const ;
      mcRPL::Layer* getLayerByName(const char *aLyrName,
                                  bool warning = true);
      const mcRPL::Layer* getLayerByName(const char *aLyrName,
                                        bool warning = true) const;
      const char* getLayerNameByIdx(int lyrID,
                                    bool warning = true) const;
      bool beginReadingLayer(const char *aLyrName,
                             mcRPL::ReadingOption readOpt);
	  
	  bool  beginGetRand(const char *aLyrName);
      void finishReadingLayers(mcRPL::ReadingOption readOpt);

      /* Neighborhoods */
      int nNbrhds() const;
      mcRPL::Neighborhood* addNbrhd(const char *aNbrhdName);
      mcRPL::Neighborhood* addNbrhd(const mcRPL::Neighborhood &rhs);
      bool rmvNbrhdByIdx(int nbrhdID,
                         bool warning = true);
      bool rmvNbrhdByName(const char *aNbrhdName,
                          bool warning = true);
      void clearNbrhds();
      mcRPL::Neighborhood* getNbrhdByIdx(int nbrhdID = 0,
                                        bool warning = true);
      const mcRPL::Neighborhood* getNbrhdByIdx(int nbrhdID = 0,
                                              bool warning = true) const;
      int getNbrhdIdxByName(const char *aNbrhdName,
                            bool warning = true);
      mcRPL::Neighborhood* getNbrhdByName(const char *aNbrhdName,
                                         bool warning = true);
      const mcRPL::Neighborhood* getNbrhdByName(const char *aNbrhdName,
                                               bool warning = true) const;
      
      /* Decomposition */
      bool dcmpLayer(int lyrID,
                     int nbrhdID,
                     int nRowSubspcs,
                     int nColSubspcs);
      bool dcmpLayer(const char *aLyrName,
                     const char *aNbrhdName,
                     int nRowSubspcs,
                     int nColSubspcs);

      bool propagateDcmp(int fromLyrID,
                         int toLyrID);
      bool propagateDcmp(int fromLyrID,
                         const mcRPL::IntVect &vToLyrIDs);
      bool propagateDcmp(const string &fromLyrName,
                         const string &toLyrName);
      bool propagateDcmp(const string &fromLyrName,
                         const vector<string> &vToLyrNames);

      bool dcmpLayers(const mcRPL::IntVect &vLyrIDs,
                      int nbrhdID,
                      int nRowSubspcs,
                      int nColSubspcs);
      bool dcmpLayers(const vector<string> &vLyrNames,
                      const char *aNbrhdName,
                      int nRowSubspcs,
                      int nColSubspcs);
      bool dcmpLayers(mcRPL::Transition &trans,
                      int nRowSubspcs,
                      int nColSubspcs);
      bool dcmpAllLayers(const char *aNbrhdName,
                         int nRowSubspcs,
                         int nColSubspcs);

      const mcRPL::OwnershipMap& ownershipMap() const;
      bool syncOwnershipMap();

      /* Evaluate - Dynamic Tasking (Task Farm) */
      bool initTaskFarm(mcRPL::Transition &trans,
                        mcRPL::MappingMethod mapMethod = mcRPL::CYLC_MAP,
                        int nSubspcs2Map = 0,
                        mcRPL::ReadingOption readOpt = mcRPL::PARA_READING);
      mcRPL::EvaluateReturn evaluate_TF(mcRPL::EvaluateType evalType,
                                       mcRPL::Transition &trans,
                                       mcRPL::ReadingOption readOpt,
                                       mcRPL::WritingOption writeOpt,
									   bool  isGPUCompute=true,
									   mcRPL::pCuf pf=NULL,
                                       bool ifInitNoData = false,
                                       bool ifSyncOwnershipMap = false,
                                       const mcRPL::LongVect *pvGlbIdxs = NULL);

      /* Evaluate - Static Tasking */

      bool initStaticTask(mcRPL::Transition &trans,
                          mcRPL::MappingMethod mapMethod = mcRPL::CYLC_MAP,
                          mcRPL::ReadingOption readOpt = mcRPL::PARA_READING);
      mcRPL::EvaluateReturn evaluate_ST(mcRPL::EvaluateType evalType,
                                       mcRPL::Transition &trans,
                                       mcRPL::WritingOption writeOpt,
									   bool isGPUCompute=true,
									   mcRPL::pCuf pf=NULL,
                                       bool ifInitNoData = false,
                                       const mcRPL::LongVect *pvGlbIdxs = NULL);

      /* Utilities */
      bool mergeTmpSubspcByGDAL(mcRPL::Transition &trans,
                                int subspcGlbID);
      bool mergeAllTmpSubspcsByGDAL(mcRPL::Transition &trans);

    protected:
      bool _initReadInData(mcRPL::Transition &trans,
                           mcRPL::ReadingOption readOpt = mcRPL::PARA_READING);

      bool _toMsgTag(int &msgTag,
                     int lyrID,
                     int subspcID,
                     int maxSubspcID);

      bool _iBcastCellspaceBegin(const string &lyrName);
      bool _iTransfSubspaceBegin(int fromPrcID,
                                 int toPrcID,
                                 const string &lyrName,
                                 int subspcGlbID);

      // returns the number of completed non-blocking (Sub)Cellspace communications
      // returns MPI_UNDEFINED if there is no active request
      int _iTransfSpaceTest(mcRPL::IntVect &vCmpltIDs,
                            mcRPL::TransferType transfType,
                            bool delSentSpaces = false,
                            bool writeRecvSpaces = false,
                            bool delRecvSpaces = false);

      void _clearTransfSpaceReqs();

      int _checkTransInDataReady(mcRPL::Transition &trans,
                                 mcRPL::ReadingOption readOpt);
      bool _initTransOutData(mcRPL::Transition &trans,
                             int subspcGlbID,
                             bool ifInitNoData = false);

      bool _setTransCellspaces(mcRPL::Transition &trans,
                               int subspcGlbID = mcRPL::ERROR_ID);
      bool _setTransNbrhd(mcRPL::Transition &trans);

      bool _initEvaluate(mcRPL::Transition &trans,
                         int subspcGlbID,
                         mcRPL::EvaluateBR br2Eval = mcRPL::EVAL_WORKBR,
                         bool ifInitNoData = false);
      bool _fnlzEvaluate(mcRPL::Transition &trans,
                         int subspcGlbID);

      bool _workerReadSubspcs(mcRPL::Transition &trans,
                              int subspcGlbID,
                              mcRPL::ReadingOption readOpt);
      bool _workerWriteSubspcs(mcRPL::Transition &trans,
                               int subspcGlbID,
                               mcRPL::WritingOption writeOpt);

      bool _dymWriteSubspcs(mcRPL::Transition &trans,
                            int subspcGlbID,
                            mcRPL::WritingOption writeOpt);
      bool _finalWriteSubspcs(mcRPL::Transition &trans,
                              mcRPL::WritingOption writeOpt);

      mcRPL::EvaluateReturn _evalNone(mcRPL::Transition &trans,
                                     int subspcGlbID = mcRPL::ERROR_ID,
                                     bool ifInitNoData = false);
      mcRPL::EvaluateReturn _evalAll(mcRPL::Transition &trans, bool isGPUCompute=true,
		                            pCuf pf=NULL,
                                    int subspcGlbID = mcRPL::ERROR_ID,
                                    mcRPL::EvaluateBR br2Eval = mcRPL::EVAL_WORKBR,
									bool ifInitNoData = false
									);
      mcRPL::EvaluateReturn _evalRandomly(mcRPL::Transition &trans,
                                         int subspcGlbID = mcRPL::ERROR_ID,
                                         bool ifInitNoData = false);
      mcRPL::EvaluateReturn _evalSelected(mcRPL::Transition &trans,
                                         const mcRPL::LongVect &vGlbIdxs,
                                         int subspcGlbID = mcRPL::ERROR_ID,
                                         mcRPL::EvaluateBR br2Eval = mcRPL::EVAL_WORKBR,
                                         bool ifInitNoData = false);

      mcRPL::EvaluateReturn _masterTF(mcRPL::Transition &trans,
                                     mcRPL::ReadingOption readOpt,
                                     mcRPL::WritingOption writeOpt);
      mcRPL::EvaluateReturn _writerTF(mcRPL::Transition &trans,
                                     mcRPL::WritingOption writeOpt);
      mcRPL::EvaluateReturn _workerTF(mcRPL::EvaluateType evalType,
                                     mcRPL::Transition &trans,
                                     mcRPL::ReadingOption readOpt,
                                     mcRPL::WritingOption writeOpt,
									 bool isGPUCompute=true,
									  mcRPL::pCuf pf=NULL,
                                     bool ifInitNoData = false,
									 const mcRPL::LongVect *pvGlbIdxs = NULL);

      mcRPL::EvaluateReturn _masterST(mcRPL::Transition &trans,
                                     mcRPL::WritingOption writeOpt);
      mcRPL::EvaluateReturn _writerST(mcRPL::Transition &trans,
                                     mcRPL::WritingOption writeOpt);
      mcRPL::EvaluateReturn _workerST(mcRPL::EvaluateType evalType,
                                     mcRPL::Transition &trans,
                                     mcRPL::WritingOption writeOpt,
									  bool isGPUCompute=true,
									 mcRPL::pCuf pf=NULL,
                                     bool ifInitNoData = false,
                                     const mcRPL::LongVect *pvGlbIdxs = NULL);

      bool _calcExchangeBRs(mcRPL::Transition &trans);
      bool _calcExchangeRoutes(mcRPL::Transition &trans);

      bool _loadCellStream(mcRPL::CellStream &cells2load);
      bool _makeCellStream(mcRPL::Transition &trans);
      bool _iExchangeBegin();
      bool _iExchangeEnd();
    private:
      mcRPL::Process _prc;
      int _nActiveWorkers;

      list<mcRPL::Layer> _lLayers;
      list<mcRPL::Neighborhood> _lNbrhds;
      mcRPL::OwnershipMap _mOwnerships;

      vector<MPI_Request> _vBcstSpcReqs;
      mcRPL::TransferInfoVect _vBcstSpcInfos;
      vector<MPI_Request> _vSendSpcReqs;
      mcRPL::TransferInfoVect _vSendSpcInfos;
      vector<MPI_Request> _vRecvSpcReqs;
      mcRPL::TransferInfoVect _vRecvSpcInfos;
      mcRPL::IntVect _vEvaledSubspcIDs;
      mcRPL::ExchangeMap _mSendRoutes;
      mcRPL::ExchangeMap _mRecvRoutes;
      map<int, mcRPL::CellStream> _mSendCells;
      map<int, mcRPL::CellStream> _mRecvCells;
      vector<MPI_Request> _vSendCellReqs;
      vector<MPI_Request> _vRecvCellReqs;
  };

};

#endif
