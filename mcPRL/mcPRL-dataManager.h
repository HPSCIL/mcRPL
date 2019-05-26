#ifndef mcPRL_DATAMANAGER_H
#define mcPRL_DATAMANAGER_H

#include <time.h>
#include "mcPRL-basicTypes.h"
#include "mcPRL-cellspace.h"
#include "mcPRL-layer.h"
#include "mcPRL-neighborhood.h"
#include "mcPRL-transition.h"
#include "mcPRL-smplDcmp.h"
#include "mcPRL-ownershipMap.h"
#include "mcPRL-exchangeMap.h"
#include "mcPRL-process.h"
#include "mcPRL-transferInfo.h"

namespace mcPRL {
  class DataManager {
    public:
      /* Constructor and destructor */
      DataManager();
      ~DataManager();
      
      /* MPI Process */
      bool initMPI(MPI_Comm comm = MPI_COMM_WORLD,
                   bool hasWriter = false);
      void finalizeMPI();
      mcPRL::Process& mpiPrc();

      /* Layers */
      int nLayers() const;
      mcPRL::Layer* addLayer(const char *aLyrName);

      /*ByGDAL----*/
      mcPRL::Layer* addLayerByGDAL(const char *aLyrName,
                                  const char *aGdalFileName,
                                  int iBand,
                                  bool pReading = true);
      bool createLayerGDAL(const char *aLyrName,
                           const char *aGdalFileName,
                           const char *aGdalFormat,
                           char **aGdalOptions = NULL);
      void closeGDAL();

      /*ByPGTIOL--hsj-*/
      mcPRL::Layer* addLayerByPGTIOL(const char *aLyrName,
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
      mcPRL::Layer* getLayerByIdx(int lyrID,
                                 bool wrning = true);
      const mcPRL::Layer* getLayerByIdx(int lyrID,
                                       bool wrning = true) const ;
      int getLayerIdxByName(const char *aLyrName,
                            bool warning = true) const ;
      mcPRL::Layer* getLayerByName(const char *aLyrName,
                                  bool warning = true);
      const mcPRL::Layer* getLayerByName(const char *aLyrName,
                                        bool warning = true) const;
      const char* getLayerNameByIdx(int lyrID,
                                    bool warning = true) const;
      bool beginReadingLayer(const char *aLyrName,
                             mcPRL::ReadingOption readOpt);
	  
	  bool  beginGetRand(const char *aLyrName);
      void finishReadingLayers(mcPRL::ReadingOption readOpt);

      /* Neighborhoods */
      int nNbrhds() const;
      mcPRL::Neighborhood* addNbrhd(const char *aNbrhdName);
      mcPRL::Neighborhood* addNbrhd(const mcPRL::Neighborhood &rhs);
      bool rmvNbrhdByIdx(int nbrhdID,
                         bool warning = true);
      bool rmvNbrhdByName(const char *aNbrhdName,
                          bool warning = true);
      void clearNbrhds();
      mcPRL::Neighborhood* getNbrhdByIdx(int nbrhdID = 0,
                                        bool warning = true);
      const mcPRL::Neighborhood* getNbrhdByIdx(int nbrhdID = 0,
                                              bool warning = true) const;
      int getNbrhdIdxByName(const char *aNbrhdName,
                            bool warning = true);
      mcPRL::Neighborhood* getNbrhdByName(const char *aNbrhdName,
                                         bool warning = true);
      const mcPRL::Neighborhood* getNbrhdByName(const char *aNbrhdName,
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
                         const mcPRL::IntVect &vToLyrIDs);
      bool propagateDcmp(const string &fromLyrName,
                         const string &toLyrName);
      bool propagateDcmp(const string &fromLyrName,
                         const vector<string> &vToLyrNames);

      bool dcmpLayers(const mcPRL::IntVect &vLyrIDs,
                      int nbrhdID,
                      int nRowSubspcs,
                      int nColSubspcs);
      bool dcmpLayers(const vector<string> &vLyrNames,
                      const char *aNbrhdName,
                      int nRowSubspcs,
                      int nColSubspcs);
      bool dcmpLayers(mcPRL::Transition &trans,
                      int nRowSubspcs,
                      int nColSubspcs);
      bool dcmpAllLayers(const char *aNbrhdName,
                         int nRowSubspcs,
                         int nColSubspcs);

      const mcPRL::OwnershipMap& ownershipMap() const;
      bool syncOwnershipMap();

      /* Evaluate - Dynamic Tasking (Task Farm) */
      bool initTaskFarm(mcPRL::Transition &trans,
                        mcPRL::MappingMethod mapMethod = mcPRL::CYLC_MAP,
                        int nSubspcs2Map = 0,
                        mcPRL::ReadingOption readOpt = mcPRL::PARA_READING);
      mcPRL::EvaluateReturn evaluate_TF(mcPRL::EvaluateType evalType,
                                       mcPRL::Transition &trans,
                                       mcPRL::ReadingOption readOpt,
                                       mcPRL::WritingOption writeOpt,
									   bool  isGPUCompute=true,
									   mcPRL::pCuf pf=NULL,
                                       bool ifInitNoData = false,
                                       bool ifSyncOwnershipMap = false,
                                       const mcPRL::LongVect *pvGlbIdxs = NULL);

      /* Evaluate - Static Tasking */

      bool initStaticTask(mcPRL::Transition &trans,
                          mcPRL::MappingMethod mapMethod = mcPRL::CYLC_MAP,
                          mcPRL::ReadingOption readOpt = mcPRL::PARA_READING);
      mcPRL::EvaluateReturn evaluate_ST(mcPRL::EvaluateType evalType,
                                       mcPRL::Transition &trans,
                                       mcPRL::WritingOption writeOpt,
									   bool isGPUCompute=true,
									   mcPRL::pCuf pf=NULL,
                                       bool ifInitNoData = false,
                                       const mcPRL::LongVect *pvGlbIdxs = NULL);

      /* Utilities */
      bool mergeTmpSubspcByGDAL(mcPRL::Transition &trans,
                                int subspcGlbID);
      bool mergeAllTmpSubspcsByGDAL(mcPRL::Transition &trans);

    protected:
      bool _initReadInData(mcPRL::Transition &trans,
                           mcPRL::ReadingOption readOpt = mcPRL::PARA_READING);

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
      int _iTransfSpaceTest(mcPRL::IntVect &vCmpltIDs,
                            mcPRL::TransferType transfType,
                            bool delSentSpaces = false,
                            bool writeRecvSpaces = false,
                            bool delRecvSpaces = false);

      void _clearTransfSpaceReqs();

      int _checkTransInDataReady(mcPRL::Transition &trans,
                                 mcPRL::ReadingOption readOpt);
      bool _initTransOutData(mcPRL::Transition &trans,
                             int subspcGlbID,
                             bool ifInitNoData = false);

      bool _setTransCellspaces(mcPRL::Transition &trans,
                               int subspcGlbID = mcPRL::ERROR_ID);
      bool _setTransNbrhd(mcPRL::Transition &trans);

      bool _initEvaluate(mcPRL::Transition &trans,
                         int subspcGlbID,
                         mcPRL::EvaluateBR br2Eval = mcPRL::EVAL_WORKBR,
                         bool ifInitNoData = false);
      bool _fnlzEvaluate(mcPRL::Transition &trans,
                         int subspcGlbID);

      bool _workerReadSubspcs(mcPRL::Transition &trans,
                              int subspcGlbID,
                              mcPRL::ReadingOption readOpt);
      bool _workerWriteSubspcs(mcPRL::Transition &trans,
                               int subspcGlbID,
                               mcPRL::WritingOption writeOpt);

      bool _dymWriteSubspcs(mcPRL::Transition &trans,
                            int subspcGlbID,
                            mcPRL::WritingOption writeOpt);
      bool _finalWriteSubspcs(mcPRL::Transition &trans,
                              mcPRL::WritingOption writeOpt);

      mcPRL::EvaluateReturn _evalNone(mcPRL::Transition &trans,
                                     int subspcGlbID = mcPRL::ERROR_ID,
                                     bool ifInitNoData = false);
      mcPRL::EvaluateReturn _evalAll(mcPRL::Transition &trans, bool isGPUCompute=true,
		                            pCuf pf=NULL,
                                    int subspcGlbID = mcPRL::ERROR_ID,
                                    mcPRL::EvaluateBR br2Eval = mcPRL::EVAL_WORKBR,
									bool ifInitNoData = false
									);
      mcPRL::EvaluateReturn _evalRandomly(mcPRL::Transition &trans,
                                         int subspcGlbID = mcPRL::ERROR_ID,
                                         bool ifInitNoData = false);
      mcPRL::EvaluateReturn _evalSelected(mcPRL::Transition &trans,
                                         const mcPRL::LongVect &vGlbIdxs,
                                         int subspcGlbID = mcPRL::ERROR_ID,
                                         mcPRL::EvaluateBR br2Eval = mcPRL::EVAL_WORKBR,
                                         bool ifInitNoData = false);

      mcPRL::EvaluateReturn _masterTF(mcPRL::Transition &trans,
                                     mcPRL::ReadingOption readOpt,
                                     mcPRL::WritingOption writeOpt);
      mcPRL::EvaluateReturn _writerTF(mcPRL::Transition &trans,
                                     mcPRL::WritingOption writeOpt);
      mcPRL::EvaluateReturn _workerTF(mcPRL::EvaluateType evalType,
                                     mcPRL::Transition &trans,
                                     mcPRL::ReadingOption readOpt,
                                     mcPRL::WritingOption writeOpt,
									 bool isGPUCompute=true,
									  mcPRL::pCuf pf=NULL,
                                     bool ifInitNoData = false,
									 const mcPRL::LongVect *pvGlbIdxs = NULL);

      mcPRL::EvaluateReturn _masterST(mcPRL::Transition &trans,
                                     mcPRL::WritingOption writeOpt);
      mcPRL::EvaluateReturn _writerST(mcPRL::Transition &trans,
                                     mcPRL::WritingOption writeOpt);
      mcPRL::EvaluateReturn _workerST(mcPRL::EvaluateType evalType,
                                     mcPRL::Transition &trans,
                                     mcPRL::WritingOption writeOpt,
									  bool isGPUCompute=true,
									 mcPRL::pCuf pf=NULL,
                                     bool ifInitNoData = false,
                                     const mcPRL::LongVect *pvGlbIdxs = NULL);

      bool _calcExchangeBRs(mcPRL::Transition &trans);
      bool _calcExchangeRoutes(mcPRL::Transition &trans);

      bool _loadCellStream(mcPRL::CellStream &cells2load);
      bool _makeCellStream(mcPRL::Transition &trans);
      bool _iExchangeBegin();
      bool _iExchangeEnd();
    private:
      mcPRL::Process _prc;
      int _nActiveWorkers;

      list<mcPRL::Layer> _lLayers;
      list<mcPRL::Neighborhood> _lNbrhds;
      mcPRL::OwnershipMap _mOwnerships;

      vector<MPI_Request> _vBcstSpcReqs;
      mcPRL::TransferInfoVect _vBcstSpcInfos;
      vector<MPI_Request> _vSendSpcReqs;
      mcPRL::TransferInfoVect _vSendSpcInfos;
      vector<MPI_Request> _vRecvSpcReqs;
      mcPRL::TransferInfoVect _vRecvSpcInfos;
      mcPRL::IntVect _vEvaledSubspcIDs;
      mcPRL::ExchangeMap _mSendRoutes;
      mcPRL::ExchangeMap _mRecvRoutes;
      map<int, mcPRL::CellStream> _mSendCells;
      map<int, mcPRL::CellStream> _mRecvCells;
      vector<MPI_Request> _vSendCellReqs;
      vector<MPI_Request> _vRecvCellReqs;
  };

};

#endif
