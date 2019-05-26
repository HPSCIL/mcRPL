#include "mcPRL-dataManager.h"

/****************************************************
*                 Protected Methods                 *
****************************************************/
bool mcPRL::DataManager::
_initReadInData(mcPRL::Transition &trans,
                mcPRL::ReadingOption readOpt) {
  const vector<string> &vInLyrNames = trans.getInLyrNames();
  for(int iInLyr = 0; iInLyr < vInLyrNames.size(); iInLyr++) {
    const string &inLyrName = vInLyrNames.at(iInLyr);
    if(!beginReadingLayer(inLyrName.c_str(), readOpt)) {
      return false;
    }
  } // end -- for(iInLyr) loop

  return true;
}
mcPRL::EvaluateReturn mcPRL::DataManager::
evaluate_ST(mcPRL::EvaluateType evalType,
            mcPRL::Transition &trans,
            mcPRL::WritingOption writeOpt,
			bool isGPUCompute,
			mcPRL::pCuf pf,
            bool ifInitNoData,
            const mcPRL::LongVect *pvGlbIdxs) {
				if(evalType!=EVAL_ALL&&isGPUCompute==true)
	{
		 cerr << __FILE__ << " function:" << __FUNCTION__ \
			  << " Error: when GPU is to be used," \
			  << " only EVAL_ALL  be used" \
			   << endl;
    _prc.abort();
    return mcPRL::EVAL_FAILED;
	}
  if(!trans.onlyUpdtCtrCell() &&
     trans.needExchange() &&
     writeOpt != mcPRL::NO_WRITING &&
     (_prc.hasWriter() ||
      (writeOpt != mcPRL::CENT_WRITING || writeOpt != mcPRL::CENTDEL_WRITING))) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: when a non-centralized neighborhood-scope Transition is to be used," \
        << " and data exchange and writing are required, " \
        << " only the centralized writing without a writer can be used" \
        << endl;
    _prc.abort();
    return mcPRL::EVAL_FAILED;
  }
  int nSMcount=_prc.getDeive().getDeviceinfo()->smCount();
  trans.setSMcount(nSMcount);
  if(trans.needExchange()) {
    if(!_calcExchangeBRs(trans) ||
       !_calcExchangeRoutes(trans)) {
      _prc.abort();
      return mcPRL::EVAL_FAILED;
    }
	
    /*
      for(int prcID = 0; prcID < _prc.nProcesses(); prcID++) {
        if(_prc.id() == prcID) {
          cout << "Send Route on process " << prcID << endl;
          mcPRL::ExchangeMap::iterator itrSendRoute = _mSendRoutes.begin();
          while(itrSendRoute != _mSendRoutes.end()) {
            cout << itrSendRoute->first << ": ";
            vector<mcPRL::ExchangeNode> &vSendNodes = itrSendRoute->second;
            for(int iSend = 0; iSend < vSendNodes.size(); iSend++) {
              mcPRL::ExchangeNode &sendNode = vSendNodes[iSend];
              cout << "(" << sendNode.subspcGlbID << ", " \
                  << sendNode.iDirection << ", "
                  << sendNode.iNeighbor << ")";
            }
            cout << endl;
            itrSendRoute++;
          }


          cout << "Recv Route on process " << prcID << endl;
          mcPRL::ExchangeMap::iterator itrRecvRoute = _mRecvRoutes.begin();
          while(itrRecvRoute != _mRecvRoutes.end()) {
            cout << itrRecvRoute->first << ": ";
            vector<mcPRL::ExchangeNode> &vRecvNodes = itrRecvRoute->second;
            for(int iRecv = 0; iRecv < vRecvNodes.size(); iRecv++) {
              mcPRL::ExchangeNode &recvNode = vRecvNodes[iRecv];
              cout << "(" << recvNode.subspcGlbID << ", " \
                  << recvNode.iDirection << ", "
                  << recvNode.iNeighbor << ")";
            }
            cout << endl;
            itrRecvRoute++;
          }
        }
        _prc.sync();
      }
     */
  } // end -- if(trans.needExchange())
  if(_prc.isMaster()) {
    if(_prc.hasWriter()) {
      _nActiveWorkers = _prc.nProcesses() - 2; // exclude the master and writer processes
    }
    else {
      _nActiveWorkers = _prc.nProcesses() - 1; // exclude the master process
    }
  }
  else if(_prc.isWriter()) {
    _nActiveWorkers = _prc.nProcesses() - 1; // exclude the writer process
  }
  mcPRL::EvaluateReturn done;
  if(_prc.isWriter()) { // writer process
    done = _writerST(trans, writeOpt);
  }
  else { // worker processes
	  done = _workerST(evalType, trans, writeOpt, isGPUCompute,pf,ifInitNoData, pvGlbIdxs);
    if(_prc.isMaster() && done != mcPRL::EVAL_FAILED) { // master process
      done = _masterST(trans, writeOpt);
    }
  }
  return done;
}
mcPRL::EvaluateReturn mcPRL::DataManager::
evaluate_TF(mcPRL::EvaluateType evalType,
            mcPRL::Transition &trans,
            mcPRL::ReadingOption readOpt,
            mcPRL::WritingOption writeOpt,
			bool  isGPUCompute,
			mcPRL::pCuf pf,
            bool ifInitNoData,
            bool ifSyncOwnershipMap,
            const mcPRL::LongVect *pvGlbIdxs) {
				if(evalType!=EVAL_ALL&&isGPUCompute==true)
	{
		 cerr << __FILE__ << " function:" << __FUNCTION__ \
			  << " Error: when GPU is to be used," \
			  << " only EVAL_ALL  be used" \
			   << endl;
    _prc.abort();
    return mcPRL::EVAL_FAILED;
	}
  if(trans.needExchange()) {
    if(!trans.onlyUpdtCtrCell() &&
        writeOpt != mcPRL::NO_WRITING) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: when a non-centralized neighborhood-scope Transition is to be used," \
          << " and data exchange and writing are required, "
          << " only the static tasking can be used for load-balancing" \
          << endl;
      _prc.abort();
      return mcPRL::EVAL_FAILED;
    }
    if(writeOpt == mcPRL::CENTDEL_WRITING ||
       writeOpt == mcPRL::PARADEL_WRITING) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: when data exchange is needed,"\
           << " deleting SubCellspaces during the writing is NOT allowed" \
           << endl;
      _prc.abort();
      return mcPRL::EVAL_FAILED;
    }
  }

  if(trans.needExchange()) {
    if(!_calcExchangeBRs(trans)) {
      _prc.abort();
      return mcPRL::EVAL_FAILED;
    }
  }

  if(_prc.isMaster() || _prc.isWriter()) {
    if(_prc.hasWriter()) {
      _nActiveWorkers = _prc.nProcesses() - 2; // exclude the master and writer processes
    }
    else {
      _nActiveWorkers = _prc.nProcesses() - 1; // exclude the master process
    }
  }

  mcPRL::EvaluateReturn done;
  if(_prc.isMaster()) { // master process
    done = _masterTF(trans, readOpt, writeOpt);
  }
  else if(_prc.isWriter()) { // writer process
    done = _writerTF(trans, writeOpt);
  }
  else { // worker processes
    done = _workerTF(evalType, trans, readOpt, writeOpt,  isGPUCompute,pf,ifInitNoData, pvGlbIdxs);
  }

  if(done != mcPRL::EVAL_FAILED) {
    if(trans.needExchange() || ifSyncOwnershipMap) {
      if(!syncOwnershipMap()) {
        done = mcPRL::EVAL_FAILED;
      }
      else if(trans.needExchange()) {
        if(!_calcExchangeRoutes(trans)) {
          done = mcPRL::EVAL_FAILED;
        }
        else if(!_makeCellStream(trans) ||
                !_iExchangeBegin() ||
                !_iExchangeEnd()) {
          done = mcPRL::EVAL_FAILED;
        }
      }
    }
  }

  return done;
}
mcPRL::EvaluateReturn mcPRL::DataManager::
_workerTF(mcPRL::EvaluateType evalType,
          mcPRL::Transition &trans,
          mcPRL::ReadingOption readOpt,
          mcPRL::WritingOption writeOpt,
		  bool  isGPUCompute,
		  mcPRL::pCuf pf,
          bool ifInitNoData,
          const mcPRL::LongVect *pvGlbIdxs) {
  if(evalType != mcPRL::EVAL_NONE &&
     evalType != mcPRL::EVAL_ALL &&
     evalType != mcPRL::EVAL_RANDOMLY &&
     evalType != mcPRL::EVAL_SELECTED) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid evaluation type (" \
         << evalType << ")" \
         << endl;
    _prc.abort();
    return mcPRL::EVAL_FAILED;
  }

  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  mcPRL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  int resp = 0;
  mcPRL::TaskSignal req = mcPRL::NEWTASK_SGNL;
  MPI_Status status;
  int mappedSubspcID = mcPRL::ERROR_ID;
  int submit2ID = _prc.hasWriter() ? 1 : 0; // if a writer exists, send to writer; otherwise master
  int readySubspcID = mcPRL::ERROR_ID;
  bool delSentSub = (writeOpt == mcPRL::CENTDEL_WRITING) ? true : false;

  while((mcPRL::TaskSignal)resp != mcPRL::QUIT_SGNL) { // keep running until receiving a Quitting instruction from master
    // if there is less than or equal to one SubCellspace to be evaluated in the queue
    if(_mOwnerships.subspcIDsOnPrc(_prc.id())->size() - _vEvaledSubspcIDs.size() <= 1) {
      req = mcPRL::NEWTASK_SGNL;
      MPI_Send(&req, 1, MPI_INT, 0, mcPRL::REQUEST_TAG, _prc.comm());
      MPI_Recv(&resp, 1, MPI_INT, 0, mcPRL::DSTRBT_TAG, _prc.comm(), &status);
      if((mcPRL::TaskSignal)resp != mcPRL::QUIT_SGNL) { // received SubCellspace ID from master
        mappedSubspcID = resp;
        _mOwnerships.mappingTo(mappedSubspcID, _prc.id());
        if(!_workerReadSubspcs(trans, mappedSubspcID, readOpt)) {
          _prc.abort();
          return mcPRL::EVAL_FAILED;
        }
      } // end -- if((mcPRL::TaskSignal)resp != mcPRL::QUIT_SGNL)
    } // end -- if(_mOwnerships.subspcIDsOnPrc(_prc.id())->size() - _vEvaledSubspcIDs.size() <= 1)

    // test pending transfers, only for centralized I/O
    if(!_vBcstSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltBcstIDs, mcPRL::BCST_TRANSF, delSentSub, false, false);
    }
    if(!_vSendSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSub, false, false);
    }
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSub, false, false);
    }

    // PROCESS A SUBSPACES THAT IS READY
    readySubspcID = _checkTransInDataReady(trans, readOpt);
    if(readySubspcID != mcPRL::ERROR_ID) {
      // when task farming, always evaluate the working rectangle.
      // the edge-first method won't be efficient, 'cuz all SubCellspaces may not
      // have been mapped yet.
      //cout << _prc.id() << ": processing SubCellspace " << readySubspcID << endl;
      switch(evalType) {
        case EVAL_NONE:
          if(_evalNone(trans, readySubspcID, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        case EVAL_ALL:
          if(_evalAll(trans, isGPUCompute,pf,readySubspcID, mcPRL::EVAL_WORKBR, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        case EVAL_RANDOMLY:
          if(_evalRandomly(trans, readySubspcID, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        case EVAL_SELECTED:
          if(pvGlbIdxs == NULL) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                 << " Error: NULL vector of cell indices to evaluate" \
                 << endl;
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          if(_evalSelected(trans, *pvGlbIdxs, readySubspcID, mcPRL::EVAL_WORKBR, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        default:
          break;
      }
      _vEvaledSubspcIDs.push_back(readySubspcID);

      if(!_workerWriteSubspcs(trans, readySubspcID, writeOpt)) {
        _prc.abort();
        return mcPRL::EVAL_FAILED;
      }
    } // end -- if(readySubspcID != ERROR_ID)

  } // end -- while((TaskSignal)resp != QUIT_SGNL)


  // COMPLETE ALL ASSIGNED SUBCELLSPACES
  const mcPRL::IntVect *pSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
  while(pSubspcIDs != NULL &&
      pSubspcIDs->size() - _vEvaledSubspcIDs.size() > 0) {
    // test pending transfers, only for centralized I/O
    if(!_vBcstSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltBcstIDs, mcPRL::BCST_TRANSF, delSentSub, false, false);
    }
    if(!_vSendSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSub, false, false);
    }
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSub, false, false);
    }

    // PROCESS A SUBSPACES THAT IS READY
    readySubspcID = _checkTransInDataReady(trans, readOpt);
    if(readySubspcID != mcPRL::ERROR_ID) {
      //cout << _prc.id() << ": processing SubCellspace " << readySubspcID << endl;
      switch(evalType) {
        case EVAL_NONE:
          if(_evalNone(trans, readySubspcID, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        case EVAL_ALL:
          if(_evalAll(trans, isGPUCompute,pf, readySubspcID, mcPRL::EVAL_WORKBR, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        case EVAL_RANDOMLY:
          if(_evalRandomly(trans, readySubspcID, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        case EVAL_SELECTED:
          if(pvGlbIdxs == NULL) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                 << " Error: NULL vector of cell indices to evaluate" \
                 << endl;
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          if(_evalSelected(trans, *pvGlbIdxs, readySubspcID, mcPRL::EVAL_WORKBR, ifInitNoData) == mcPRL::EVAL_FAILED) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
          break;
        default:
          break;
      }
      _vEvaledSubspcIDs.push_back(readySubspcID);

      if(!_workerWriteSubspcs(trans, readySubspcID, writeOpt)) {
        _prc.abort();
        return mcPRL::EVAL_FAILED;
      }
    } // end -- if(readySubspcID != ERROR_ID)
  }  // end -- while(_mOwnerships.subspcIDsOnPrc(_prc.id())->size() - _vEvaledSubspcIDs.size() > 0)

  // complete all pending transfers, only for centralized I/O
  while(!_vBcstSpcReqs.empty() && nCmpltBcsts != MPI_UNDEFINED) {
    nCmpltBcsts = _iTransfSpaceTest(vCmpltBcstIDs, mcPRL::BCST_TRANSF, delSentSub, false, false);
  }
  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSub, false, false);
  }
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSub, false, false);
  }

  req = mcPRL::QUIT_SGNL;
  MPI_Send(&req, 1, MPI_INT, 0, mcPRL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to master
  if(_prc.hasWriter()) {
    MPI_Send(&req, 1, MPI_INT, 1, mcPRL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to writer
  }

  _clearTransfSpaceReqs();

  return mcPRL::EVAL_SUCCEEDED;
}
mcPRL::EvaluateReturn mcPRL::DataManager::
_workerST(mcPRL::EvaluateType evalType,
          mcPRL::Transition &trans,
          mcPRL::WritingOption writeOpt,
		  bool isGPUCompute,
		  mcPRL::pCuf pf,
          bool ifInitNoData,
          const mcPRL::LongVect *pvGlbIdxs) {
  if(evalType != mcPRL::EVAL_NONE &&
     evalType != mcPRL::EVAL_ALL &&
     evalType != mcPRL::EVAL_RANDOMLY &&
     evalType != mcPRL::EVAL_SELECTED) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid evaluation type (" \
         << evalType << ")" \
         << endl;
    _prc.abort();
    return mcPRL::EVAL_FAILED;
  }

  _clearTransfSpaceReqs();

  const IntVect *pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
  if(pvSubspcIDs != NULL) {
    for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
      int subspcGlbID = pvSubspcIDs->at(iSubspc);
      if(!trans.needExchange() ||
         (trans.needExchange() && !trans.edgesFirst())) { // no exchange needed
        //cout << _prc.id() << ": processing SubCellspace " << subspcGlbID << endl;
        switch(evalType) {
          case EVAL_NONE:
            if(_evalNone(trans, subspcGlbID, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          case EVAL_ALL:
			  if(_evalAll(trans,isGPUCompute, pf,subspcGlbID, mcPRL::EVAL_WORKBR, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          case EVAL_RANDOMLY:
            if(_evalRandomly(trans, subspcGlbID, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          case EVAL_SELECTED:
            if(pvGlbIdxs == NULL) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                   << " Error: NULL vector of cell indices to evaluate" \
                   << endl;
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            if(_evalSelected(trans, *pvGlbIdxs, subspcGlbID, mcPRL::EVAL_WORKBR, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          default:
            break;
        }
        _vEvaledSubspcIDs.push_back(subspcGlbID);

        if(!_dymWriteSubspcs(trans, subspcGlbID, writeOpt)) {
          _prc.abort();
          return mcPRL::EVAL_FAILED;
        }
		trans.clearGPUMem();
      } // end -- if(!trans.needExchange())

      else { // exchange needed, process the edges first
        //cout << _prc.id() << ": processing SubCellspace edge " << subspcGlbID << endl;
        switch(evalType) {
          case EVAL_NONE:
            if(_evalNone(trans, subspcGlbID, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          case EVAL_ALL:
            if(_evalAll(trans, isGPUCompute,pf, subspcGlbID, mcPRL::EVAL_EDGES, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          case EVAL_RANDOMLY:
            if(_evalRandomly(trans, subspcGlbID, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          case EVAL_SELECTED:
            if(pvGlbIdxs == NULL) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                  << " Error: NULL vector of cell indices to evaluate" \
                  << endl;
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            if(_evalSelected(trans, *pvGlbIdxs, subspcGlbID, mcPRL::EVAL_EDGES, ifInitNoData) == mcPRL::EVAL_FAILED) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            break;
          default:
            break;
        }
      } // end -- exchange needed
    } // end -- for(iSubspc)

    if(trans.needExchange()) { // exchange, while processing the interiors
      if(!_makeCellStream(trans) || !_iExchangeBegin()) {
        _prc.abort();
        return mcPRL::EVAL_FAILED;
      }

      if(trans.edgesFirst()) {
        for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
          int subspcGlbID = pvSubspcIDs->at(iSubspc);
          //cout << _prc.id() << ": processing SubCellspace interior " << subspcGlbID << endl;
          switch(evalType) {
            case EVAL_NONE:
              break;
            case EVAL_ALL:
              if(_evalAll(trans, isGPUCompute,pf, subspcGlbID, mcPRL::EVAL_INTERIOR, ifInitNoData) == mcPRL::EVAL_FAILED) {
                _prc.abort();
                return mcPRL::EVAL_FAILED;
              }
              break;
            case EVAL_RANDOMLY:
              break;
            case EVAL_SELECTED:
              if(pvGlbIdxs == NULL) {
                cerr << __FILE__ << " function:" << __FUNCTION__ \
                    << " Error: NULL vector of cell indices to evaluate" \
                    << endl;
                _prc.abort();
                return mcPRL::EVAL_FAILED;
              }
              if(_evalSelected(trans, *pvGlbIdxs, subspcGlbID, mcPRL::EVAL_INTERIOR, ifInitNoData) == mcPRL::EVAL_FAILED) {
                _prc.abort();
                return mcPRL::EVAL_FAILED;
              }
              break;
            default:
              break;
          }
          _vEvaledSubspcIDs.push_back(subspcGlbID);
		   trans.clearGPUMem();
          if(!_dymWriteSubspcs(trans, subspcGlbID, writeOpt)) {
            _prc.abort();
            return mcPRL::EVAL_FAILED;
          }
        } // end -- for(iSubspc)

        if(!_iExchangeEnd()) {
          return mcPRL::EVAL_FAILED;
        }
      }
    } // end -- if(trans.needExchange())

    if(!_finalWriteSubspcs(trans, writeOpt)) {
      _prc.abort();
      return mcPRL::EVAL_FAILED;
    }

  } // end -- if(pvSubspcIDs != NULL)

  // complete all pending transfers, only for centralized I/O
  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  mcPRL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  bool delSentSub = (writeOpt == mcPRL::CENTDEL_WRITING) ? true : false;

  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSub, false, false);
  }

  // send Quitting signal
  mcPRL::TaskSignal req = mcPRL::QUIT_SGNL;
  if(_prc.hasWriter()) {
    MPI_Send(&req, 1, MPI_INT, 1, mcPRL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to writer
  }

  if(!_prc.isMaster()) {
    if(!_prc.hasWriter() && writeOpt != mcPRL::NO_WRITING) {
      MPI_Send(&req, 1, MPI_INT, 0, mcPRL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to master
      //cout << _prc.id() << " quitting" << endl;
    }
    _clearTransfSpaceReqs();
  }

  return mcPRL::EVAL_SUCCEEDED;
}
mcPRL::EvaluateReturn mcPRL::DataManager::
_evalAll(mcPRL::Transition &trans,bool isGPUCompute,
           pCuf pf,
         int subspcGlbID,
         mcPRL::EvaluateBR br2Eval,
         bool ifInitNoData) {
  if(br2Eval != mcPRL::EVAL_WORKBR) {
    if(subspcGlbID == mcPRL::ERROR_ID) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: cannot evaluate the Edges or Interior of a whole Cellspace" \
           << endl;
      return mcPRL::EVAL_FAILED;
    }
    if(trans.getOutLyrNames().empty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: no output Layers to evaluate Edges or Interior" \
           << endl;
      return mcPRL::EVAL_FAILED;
    }
    if(!trans.isOutLyr(trans.getPrimeLyrName())) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the primary Layer (" << trans.getPrimeLyrName() \
           << ") is NOT an output Layer" << endl;
      return mcPRL::EVAL_FAILED;
    }
  }

  if(!_initEvaluate(trans, subspcGlbID, br2Eval, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize the Transition" \
        << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::Layer *pPrmLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrmLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to find the primary Layer" \
         << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::Cellspace *pPrmCellspace = NULL;
  if(subspcGlbID == mcPRL::ERROR_ID) { // if evaluate the whole Cellspace
    pPrmCellspace = pPrmLyr->cellspace();
  }
  else { // if evaluate a SubCellspace
    pPrmCellspace = pPrmLyr->subCellspace_glbID(subspcGlbID, true);
  }
  if(pPrmCellspace == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to find the primary Cellspace" \
         << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::CoordBR workBR;
  const mcPRL::CoordBR *pWorkBR;
  mcPRL::EvaluateReturn done = mcPRL::EVAL_SUCCEEDED;
  switch(br2Eval) {
    case mcPRL::EVAL_WORKBR:
      if(subspcGlbID == mcPRL::ERROR_ID) { // if evaluate the whole Cellspace
        if(!pPrmCellspace->info()->calcWorkBR(&workBR, trans.getNbrhd(), true)) {
          return mcPRL::EVAL_FAILED;
        }
      }
      else { // if evaluate a SubCellspace
        workBR = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->workBR();
      }
      done = trans.evalBR(workBR,isGPUCompute,pf);
      break;
    case mcPRL::EVAL_EDGES:
      for(int iEdge = 0; iEdge < ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->nEdges(); iEdge++) {
        pWorkBR = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->edgeBR(iEdge);
        if(pWorkBR != NULL) {
          done = trans.evalBR(*pWorkBR,isGPUCompute, pf);
          if(done == mcPRL::EVAL_FAILED) {
            break;
          }
        }
      }
      break;
    case mcPRL::EVAL_INTERIOR:
      pWorkBR = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->interiorBR();
      if(pWorkBR != NULL) {
        done = trans.evalBR(*pWorkBR, isGPUCompute,pf);
      }
      break;
    default:
      break;
  }

  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcPRL::EVAL_FAILED;
  }

  return done;
}
bool mcPRL::DataManager::
_toMsgTag(int &msgTag,
          int lyrID,
          int subspcID,
          int maxSubspcID) {
  if(lyrID < 0) {
    cerr << __FILE__ << " function:" << __FUNCTION__
         << " Error: invalid Layer ID (" << lyrID \
         << ")" << endl;
    return false;
  }
  if((subspcID < 0 && maxSubspcID >= 0) ||
     (subspcID >= 0 && maxSubspcID < 0) ||
     (subspcID >= 0 && maxSubspcID >= 0 && subspcID > maxSubspcID)) {
    cerr << __FILE__ << " function:" << __FUNCTION__
         << " Error: invalid SubCellspace ID (" << subspcID \
         << ") and/or max SubCellspace ID (" << maxSubspcID \
         << ")" << endl;
    return false;
  }

  if(subspcID < 0 && maxSubspcID < 0) { // whole Cellspace
    msgTag = lyrID + 1; // to make sure msgTag is always greater than zero
  }
  else {
    stringstream ssMaxID;
    ssMaxID << maxSubspcID;
    string sMaxID = ssMaxID.str();
    int factor = std::pow(10.0, (int)(sMaxID.size())); // if maxSubspcID = 30, factor = 100;
    msgTag = (lyrID+1) * factor + subspcID;
  }

  return true;
}

bool mcPRL::DataManager::
_iBcastCellspaceBegin(const string &lyrName) {
  mcPRL::Layer *pLyr = getLayerByName(lyrName.c_str(), true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  }

  int msgTag;
  if(!_toMsgTag(msgTag, getLayerIdxByName(lyrName.c_str()),
                mcPRL::ERROR_ID, mcPRL::ERROR_ID)) {
    _prc.abort();
    return false;
  }

  if(_prc.isMaster()) { // Master process
    mcPRL::Cellspace *pSpc = pLyr->cellspace();
    if(pSpc == NULL || pSpc->isEmpty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL or empty Cellspace on Process ("
          << _prc.id() << ") to transfer" \
          << endl;
      _prc.abort();
      return false;
    }

    int msgSize = pSpc->info()->size() * pSpc->info()->dataSize();
    for(int toPrcID = 1; toPrcID < _prc.nProcesses(); toPrcID++) {
      _vBcstSpcReqs.resize(_vBcstSpcReqs.size() + 1);
      MPI_Isend(pSpc->_pData, msgSize, MPI_CHAR, toPrcID, msgTag, _prc.comm(), &(_vBcstSpcReqs.back()));
      _vBcstSpcInfos.push_back(mcPRL::TransferInfo(_prc.id(), toPrcID,
                                                  lyrName, mcPRL::ERROR_ID,
                                                  mcPRL::SPACE_TRANSFDATA));
    }
  } // end -- if(_prc.isMaster())

  else if(!_prc.isWriter()) { // Worker process
    mcPRL::Cellspace *pSpc = pLyr->cellspace();
    if(pSpc == NULL) {
      pLyr->createCellspace();
      pSpc = pLyr->cellspace();
    }
    if(pSpc == NULL || pSpc->isEmpty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL or empty Cellspace on Process ("
          << _prc.id() << ") to transfer" \
          << endl;
      _prc.abort();
      return false;
    }

    int msgSize = pSpc->info()->size() * pSpc->info()->dataSize();
    _vBcstSpcReqs.resize(_vBcstSpcReqs.size() + 1);
    MPI_Irecv(pSpc->_pData, msgSize, MPI_CHAR, 0, msgTag, _prc.comm(), &(_vRecvSpcReqs.back()));
    _vBcstSpcInfos.push_back(mcPRL::TransferInfo(0, _prc.id(),
                                                lyrName, mcPRL::ERROR_ID,
                                                mcPRL::SPACE_TRANSFDATA));
  } // end -- worker process

  return true;
}

bool mcPRL::DataManager::
_iTransfSubspaceBegin(int fromPrcID,
                      int toPrcID,
                      const string &lyrName,
                      int subspcGlbID) {
  mcPRL::Layer *pLyr = getLayerByName(lyrName.c_str(), true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  }

  int maxSubspcID = pLyr->maxSubCellspaceID();
  if(subspcGlbID < 0 || subspcGlbID > maxSubspcID) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: invalid SubCellspace ID (" << subspcGlbID \
        << ")" << endl;
    _prc.abort();
    return false;
  }

  int msgTag;
  if(!_toMsgTag(msgTag, getLayerIdxByName(lyrName.c_str()),
                subspcGlbID, maxSubspcID)) {
    _prc.abort();
    return false;
  }

  if(_prc.id() == fromPrcID) { // FROM process
    mcPRL::Cellspace *pSpc = pLyr->subCellspace_glbID(subspcGlbID, true);
    if(pSpc == NULL || pSpc->isEmpty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: NULL or empty SubCellspace on Process ("
           << _prc.id() << ") to transfer" \
           << endl;
      _prc.abort();
      return false;
    }

    int msgSize = pSpc->info()->size() * pSpc->info()->dataSize();
    _vSendSpcReqs.resize(_vSendSpcReqs.size() + 1);
    MPI_Isend(pSpc->_pData, msgSize, MPI_CHAR, toPrcID, msgTag, _prc.comm(), &(_vSendSpcReqs.back()));
    _vSendSpcInfos.push_back(mcPRL::TransferInfo(_prc.id(), toPrcID,
                             lyrName, subspcGlbID,
                             mcPRL::SPACE_TRANSFDATA));
  } // end -- if(_prc.id() == fromPrcID)

  else if(_prc.id() == toPrcID) { // TO process
    mcPRL::Cellspace *pSpc = pLyr->subCellspace_glbID(subspcGlbID);
    if(pSpc == NULL) {
      pSpc = pLyr->addSubCellspace(subspcGlbID);
    }
    if(pSpc == NULL || pSpc->isEmpty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: NULL or empty SubCellspace on Process ("
          << _prc.id() << ") to transfer" \
          << endl;
      _prc.abort();
      return false;
    }

    int msgSize = pSpc->info()->size() * pSpc->info()->dataSize();
    _vRecvSpcReqs.resize(_vRecvSpcReqs.size() + 1);
    MPI_Irecv(pSpc->_pData, msgSize, MPI_CHAR, fromPrcID, msgTag, _prc.comm(), &(_vRecvSpcReqs.back()));
    _vRecvSpcInfos.push_back(mcPRL::TransferInfo(fromPrcID, _prc.id(),
                                                lyrName, subspcGlbID,
                                                mcPRL::SPACE_TRANSFDATA));
  } // end -- if(_prc.id() == toPrcID)

  return true;
}

int mcPRL::DataManager::
_iTransfSpaceTest(mcPRL::IntVect &vCmpltIDs,
                  mcPRL::TransferType transfType,
                  bool delSentSpaces,
                  bool writeRecvSpaces,
                  bool delRecvSpaces) {
  int nCmplts = 0;

  vector<MPI_Request> *pvTransfReqs = NULL;
  mcPRL::TransferInfoVect *pvTransfInfos = NULL;
  switch(transfType) {
    case mcPRL::BCST_TRANSF:
      pvTransfReqs = &_vBcstSpcReqs;
      pvTransfInfos = &_vBcstSpcInfos;
      break;
    case mcPRL::SEND_TRANSF:
      pvTransfReqs = &_vSendSpcReqs;
      pvTransfInfos = &_vSendSpcInfos;
      break;
    case mcPRL::RECV_TRANSF:
      pvTransfReqs = &_vRecvSpcReqs;
      pvTransfInfos = &_vRecvSpcInfos;
      break;
    default:
      cerr << __FILE__ << " function:" << __FUNCTION__
           << " Error: invalid TransferType (" << transfType \
           << ")" << endl;
      return MPI_UNDEFINED;
  }

  if(pvTransfReqs->empty()) {
    return MPI_UNDEFINED;
  }

  vCmpltIDs.clear();
  vCmpltIDs.resize(pvTransfReqs->size());
  vector<MPI_Status> vCmpltStats(pvTransfReqs->size());

  MPI_Testsome(pvTransfReqs->size(), &(pvTransfReqs->at(0)),
               &nCmplts, &(vCmpltIDs[0]), &(vCmpltStats[0]));
  if(nCmplts >= 0) {
    vCmpltIDs.resize(nCmplts);
    vCmpltStats.resize(nCmplts);
  }

  for(int iCmplt = 0; iCmplt < nCmplts; iCmplt++) {
    mcPRL::TransferInfo &transfInfo = pvTransfInfos->at(vCmpltIDs[iCmplt]);
    transfInfo.complete();

    int fromPrcID = transfInfo.fromPrcID();
    int toPrcID = transfInfo.toPrcID();
    int subspcGlbID = transfInfo.subspcGlbID();

    const string &lyrName = transfInfo.lyrName();
    mcPRL::Layer *pLyr = getLayerByName(lyrName.c_str(), true);
    if(pLyr == NULL) {
      continue;
    }

    if(delSentSpaces &&
       _prc.id() == fromPrcID) {
      if(transfType == mcPRL::SEND_TRANSF) {
        if(!pLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
      else if(transfType == mcPRL::BCST_TRANSF) {
        if(pvTransfInfos->checkBcastCellspace(_prc.id(), lyrName)) {
          pLyr->delCellspace();
        }
      }
    } // end -- if(delSentSpaces)

    if(writeRecvSpaces &&
       _prc.id() == toPrcID &&
       transfType == mcPRL::RECV_TRANSF) {
      //cout << _prc.id() << ": writing SubCellspace " << subspcGlbID << endl;
      if(!pLyr->writeSubCellspaceByGDAL(subspcGlbID)) {
        cerr << __FILE__ << " function:" << __FUNCTION__
             << " Error: failed to write SubCellspace (" \
             << subspcGlbID << ")" << endl;
      }
    } // end -- if(writeRecvSpaces)

    if(delRecvSpaces &&
       _prc.id() == toPrcID &&
       transfType == mcPRL::RECV_TRANSF) {
      if(!pLyr->delSubCellspace_glbID(subspcGlbID)) {
        cerr << __FILE__ << " function:" << __FUNCTION__
             << " Error: unable to find SubCellspace (" \
             << subspcGlbID << ") to delete" << endl;
      }
    } // end -- if(delRecvSpaces)

  } // end -- for(iCmplt) loop

  return nCmplts;
}

void mcPRL::DataManager::
_clearTransfSpaceReqs() {
  _vBcstSpcReqs.clear();
  _vBcstSpcInfos.clear();
  _vSendSpcReqs.clear();
  _vSendSpcInfos.clear();
  _vRecvSpcReqs.clear();
  _vRecvSpcInfos.clear();
  _vEvaledSubspcIDs.clear();
}

int mcPRL::DataManager::
_checkTransInDataReady(mcPRL::Transition &trans,
                       mcPRL::ReadingOption readOpt) {
  int readySubspcID = mcPRL::ERROR_ID, subspcID2Check;

  const mcPRL::IntVect *pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
  if(pvSubspcIDs == NULL) {
    return mcPRL::ERROR_ID;
  }

  for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
    subspcID2Check = pvSubspcIDs->at(iSubspc);
    if(std::find(_vEvaledSubspcIDs.begin(), _vEvaledSubspcIDs.end(), subspcID2Check) != _vEvaledSubspcIDs.end()) {
      continue; // this SubCellspace has been evaluated, ignore the following steps, check the next SubCellspace
    }
    else { // if SubCellspace has NOT been evaluated
      if(readOpt == mcPRL::PARA_READING) { // if parallel reading, input data has been read
        readySubspcID = subspcID2Check;
        break; // break the for(iSubspc) loop
      }
      else if(readOpt == mcPRL::CENT_READING ||
              readOpt == mcPRL::CENTDEL_READING) { // if centralized reading, check if transfers have completed
        const vector<string> &vInLyrNames = trans.getInLyrNames();
        for(int iInLyr = 0; iInLyr < vInLyrNames.size(); iInLyr++) {
          mcPRL::Layer *pInLyr = getLayerByName(vInLyrNames[iInLyr].c_str(), true);
          if(pInLyr == NULL) {
            return mcPRL::ERROR_ID;
          }
          if(pInLyr->isDecomposed()) {
            const mcPRL::TransferInfo *pTransfInfo = _vRecvSpcInfos.findInfo(vInLyrNames[iInLyr], subspcID2Check);
            if(pTransfInfo == NULL || pTransfInfo->completed() == false) {
              subspcID2Check = mcPRL::ERROR_ID;
              break; // the transfer of this SubCellspace has NOT been completed, break the for(iInLyr) loop
            }
          }
          else {
            const mcPRL::TransferInfo *pTransfInfo = _vBcstSpcInfos.findInfo(vInLyrNames[iInLyr], mcPRL::ERROR_ID);
            if(pTransfInfo == NULL || pTransfInfo->completed() == false) {
              subspcID2Check = mcPRL::ERROR_ID;
              break; // the transfer (broadcast) of this Cellspace has NOT been completed, break the for(iInLyr) loop
            }
          }
        } // end -- for(iInLyr) loop

        if(subspcID2Check != mcPRL::ERROR_ID) {
          readySubspcID = subspcID2Check;
          break; // found a SubCellspace whose transfer has been completed, break the for(iSubspc) loop
        }
      } // end -- if(readOpt == mcPRL::CENT_READING)
      else if(readOpt == mcPRL::PGT_READING) {
        readySubspcID = subspcID2Check;
        break; // break the for(iSubspc) loop
      }
    } // end -- if SubCellspace has NOT been evaluated
  } // end -- for(iSubspc) loop

  return readySubspcID;
}

bool mcPRL::DataManager::
_initTransOutData(mcPRL::Transition &trans,
                  int subspcGlbID,
                  bool ifInitNoData) {
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    if(pOutLyr->cellspaceInfo() == NULL) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: Layer (" << vOutLyrNames[iOutLyr] \
           << ") does NOT have a global CellspaceInfo initialized" \
           << endl;
      return false;
    }

    double *pNoDataVal = NULL;
    double noDataVal;
    if(ifInitNoData) {
      noDataVal = pOutLyr->cellspaceInfo()->getNoDataValAs<double>();
      pNoDataVal = &noDataVal;
    }

    if(pOutLyr->isDecomposed()) { // decomposed Layer
      mcPRL::SubCellspace *pSubspc = pOutLyr->subCellspace_glbID(subspcGlbID, false);
      if(pSubspc == NULL) {
        pSubspc = pOutLyr->addSubCellspace(subspcGlbID, pNoDataVal);
      }
      if(pSubspc == NULL) {
        return false;
      }
    }
    else { // whole Cellspace on the Layer
      mcPRL::Cellspace *pSpc = pOutLyr->cellspace();
      if(pSpc == NULL) {
        pOutLyr->createCellspace(pNoDataVal);
        pSpc = pOutLyr->cellspace();
      }
      if(pSpc == NULL) {
        return false;
      }
    }
  } // end -- for(iOutLyr)

  return true;
}

bool mcPRL::DataManager::
_setTransCellspaces(mcPRL::Transition &trans,
                    int subspcGlbID) {
  trans.clearCellspaces();

  const vector<string> &vInLyrNames = trans.getInLyrNames();
  for(size_t iLyr = 0; iLyr < vInLyrNames.size(); iLyr++) {
    mcPRL::Layer *pLyr = getLayerByName(vInLyrNames[iLyr].c_str(), true);
    if(pLyr == NULL) {
      return false;
    }
    if(!pLyr->isDecomposed() &&
       pLyr->cellspace() != NULL &&
       !pLyr->cellspace()->isEmpty()) { // use whole Cellspace
      if(!trans.setCellspace(vInLyrNames[iLyr], pLyr->cellspace())) {
        return false;
      }
	  if(pLyr->cellspace()->_dpData==NULL)
	  {
		  pLyr->cellspace()->transDataToGPU();
	  }
    }
    else { // use SubCellspace
      mcPRL::SubCellspace *pSubSpc = pLyr->subCellspace_glbID(subspcGlbID, true);
      if(pSubSpc == NULL) {
        return false;
      }
      if(!trans.setCellspace(vInLyrNames[iLyr], pSubSpc)) {
        return false;
      }
	  if(pSubSpc->_dpData==NULL)
	  {
		pSubSpc->transDataToGPU();
	  }
    }
  } // end -- for(iLyr)

  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(size_t iLyr = 0; iLyr < vOutLyrNames.size(); iLyr++) {
    if(trans.getCellspaceByLyrName(vOutLyrNames[iLyr]) != NULL) {
      continue; // ignore the SubCellspaces that have been set as input data
    }
    mcPRL::Layer *pLyr = getLayerByName(vOutLyrNames[iLyr].c_str(), true);
    if(pLyr == NULL) {
      return false;
    }
    if(!pLyr->isDecomposed() &&
       pLyr->cellspace() != NULL &&
       !pLyr->cellspace()->isEmpty()) { // use whole Cellspace
      if(!trans.setCellspace(vOutLyrNames[iLyr], pLyr->cellspace())) {
        return false;
      }
	    if(pLyr->cellspace()->_dpData==NULL)
	  {
		  pLyr->cellspace()->getGPUMalloc();
	  }
    }
    else { // use SubCellspace
      SubCellspace *pSubSpc = pLyr->subCellspace_glbID(subspcGlbID, true);
      if(pSubSpc == NULL) {
        return false;
      }
      if(!trans.setCellspace(vOutLyrNames[iLyr], pSubSpc)) {
        return false;
      }
	    if(pSubSpc->_dpData==NULL)
	  {
		pSubSpc->getGPUMalloc();
	  }
    }
  } // end -- for(iLyr)

  if(!trans.afterSetCellspaces(subspcGlbID)) {
    return false;
  }

  return true;
}

bool mcPRL::DataManager::
_setTransNbrhd(mcPRL::Transition &trans) {
  trans.setNbrhd();

  if(!trans.getNbrhdName().empty()) {
    mcPRL::Neighborhood *pNbrhd = getNbrhdByName(trans.getNbrhdName().c_str(), true);
    if(pNbrhd != NULL) {
      trans.setNbrhd(pNbrhd);
      if(!trans.afterSetNbrhd()) {
        return false;
      }
    }
    else {
      return false;
    }
  }

  return true;
}

bool mcPRL::DataManager::
_initEvaluate(mcPRL::Transition &trans,
              int subspcGlbID,
              mcPRL::EvaluateBR br2Eval,
              bool ifInitNoData) {
  if(!_setTransNbrhd(trans)) {
    return false;
  }
  if(br2Eval != mcPRL::EVAL_INTERIOR) {
    if(!_initTransOutData(trans, subspcGlbID, ifInitNoData)) {
      return false;
    }
  }
  if(!_setTransCellspaces(trans, subspcGlbID)) {
    return false;
  }

  if(!trans.check()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: Transition check failed" << endl;
    return false;
  }

  if(br2Eval != mcPRL::EVAL_INTERIOR && trans.needExchange()) {
    trans.setUpdateTracking(true);
  }

  return true;
}

bool mcPRL::DataManager::
_fnlzEvaluate(mcPRL::Transition &trans,
              int subspcGlbID) {
  trans.setUpdateTracking(false);
  return true;
}

bool mcPRL::DataManager::
_workerReadSubspcs(mcPRL::Transition &trans,
                   int subspcGlbID,
                   mcPRL::ReadingOption readOpt) {
  const vector<string> &vInLyrNames = trans.getInLyrNames();
  for(int iLyr = 0; iLyr < vInLyrNames.size(); iLyr++) {
    const string &inLyrName = vInLyrNames[iLyr];
    mcPRL::Layer *pLyr = getLayerByName(inLyrName.c_str(), true);
    if(pLyr == NULL) {
      return false;
    }

    if(pLyr->isDecomposed()) { // decomposed Layer
      if(readOpt == mcPRL::PARA_READING) { // parallel reading
        if(!pLyr->loadSubCellspaceByGDAL(subspcGlbID)) {
          return false;
        }
      }
      else if(readOpt == mcPRL::CENT_READING ||
              readOpt == mcPRL::CENTDEL_READING) { // centralized reading
        if(!_iTransfSubspaceBegin(0, _prc.id(), inLyrName, subspcGlbID)) {
          return false;
        }
      }
      else if(readOpt == mcPRL::PGT_READING) {
        if(!pLyr->loadSubCellspaceByPGTIOL(subspcGlbID)) {
          return false;
        }
      }
    }
    else { // non-decomposed Layer
      cerr << __FILE__ << " function:" << __FUNCTION__
           << " Error: unable to read a SubCellspace within a non-decomposed Layer (" \
           << inLyrName << ")" \
           << endl;
      return false;
    }
  } // end -- for(iLyr)
  return true;
}

bool mcPRL::DataManager::
_workerWriteSubspcs(mcPRL::Transition &trans,
                    int subspcGlbID,
                    mcPRL::WritingOption writeOpt) {
  if(writeOpt == mcPRL::NO_WRITING) {
    return true;
  }

  mcPRL::TaskSignal req = mcPRL::SUBMIT_SGNL;
  int submit2ID = _prc.hasWriter() ? 1 : 0; // if a writer exists, send to writer; otherwise master
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    const string &outLyrName = vOutLyrNames[iOutLyr];
    mcPRL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    if(!pOutLyr->isDecomposed()) {
      cerr << __FILE__ << " function:" << __FUNCTION__
           << " Error: unable to write a SubCellspace within a non-decomposed Layer (" \
           << outLyrName << ")" \
           << endl;
      return false;
    }

    if(writeOpt == mcPRL::PARA_WRITING ||
       writeOpt == mcPRL::PARADEL_WRITING) {
      if(!pOutLyr->writeTmpSubCellspaceByGDAL(subspcGlbID, _prc.nProcesses())) {
        return false;
      }
      if(writeOpt == mcPRL::PARADEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
    }
    else if(writeOpt == mcPRL::CENT_WRITING ||
            writeOpt == mcPRL::CENTDEL_WRITING) {
      int lyrID = getLayerIdxByName(outLyrName.c_str(), true);
      MPI_Send(&req, 1, MPI_INT, submit2ID, mcPRL::REQUEST_TAG, _prc.comm());
      int sendSubspcInfo[2];
      sendSubspcInfo[0] = lyrID;
      sendSubspcInfo[1] = subspcGlbID;
      MPI_Send(sendSubspcInfo, 2, MPI_INT, submit2ID, mcPRL::SUBMIT_TAG, _prc.comm());
      if(!_iTransfSubspaceBegin(_prc.id(), submit2ID, outLyrName, subspcGlbID)) {
        return false;
      }
    }
    else if(writeOpt == mcPRL::PGT_WRITING ||
            writeOpt == mcPRL::PGTDEL_WRITING) {
      if(!pOutLyr->writeSubCellspaceByPGTIOL(subspcGlbID)) {
        return false;
      }

      if(writeOpt == mcPRL::PGTDEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
    }
  } // end -- for(iOutLyr)

  if(writeOpt == mcPRL::PARA_WRITING ||
     writeOpt == mcPRL::PARADEL_WRITING ||
     writeOpt == mcPRL::PGT_WRITING ||
     writeOpt == mcPRL::PGTDEL_WRITING) {
    MPI_Send(&req, 1, MPI_INT, submit2ID, mcPRL::REQUEST_TAG, _prc.comm());
    MPI_Send(&subspcGlbID, 1, MPI_INT, submit2ID, mcPRL::SUBMIT_TAG, _prc.comm());
  }

  return true;
}

bool mcPRL::DataManager::
_dymWriteSubspcs(mcPRL::Transition &trans,
                 int subspcGlbID,
                 mcPRL::WritingOption writeOpt) {
  if(writeOpt == mcPRL::NO_WRITING) {
    return true;
  }

  mcPRL::TaskSignal req = mcPRL::SUBMIT_SGNL;
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    const string &outLyrName = vOutLyrNames[iOutLyr];
    mcPRL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    if(!pOutLyr->isDecomposed()) {
      cerr << __FILE__ << " function:" << __FUNCTION__
          << " Error: unable to write a SubCellspace within a non-decomposed Layer (" \
          << outLyrName << ")" \
          << endl;
      return false;
    }

    if(writeOpt == mcPRL::PARA_WRITING ||
       writeOpt == mcPRL::PARADEL_WRITING) {
      if(!pOutLyr->writeTmpSubCellspaceByGDAL(subspcGlbID, _prc.nProcesses())) {
        return false;
      }
      if(writeOpt == mcPRL::PARADEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
    }
    else if((writeOpt == mcPRL::CENT_WRITING || writeOpt == mcPRL::CENTDEL_WRITING) &&
            _prc.hasWriter()) {
      int lyrID = getLayerIdxByName(outLyrName.c_str(), true);
      MPI_Send(&req, 1, MPI_INT, 1, mcPRL::REQUEST_TAG, _prc.comm());
      int sendSubspcInfo[2];
      sendSubspcInfo[0] = lyrID;
      sendSubspcInfo[1] = subspcGlbID;
      MPI_Send(sendSubspcInfo, 2, MPI_INT, 1, mcPRL::SUBMIT_TAG, _prc.comm());
      if(!_iTransfSubspaceBegin(_prc.id(), 1, outLyrName, subspcGlbID)) {
        return false;
      }
    }
    else if(writeOpt == mcPRL::PGT_WRITING ||
            writeOpt == mcPRL::PGTDEL_WRITING) {
      if(!pOutLyr->writeSubCellspaceByPGTIOL(subspcGlbID)) {
        return false;
      }
      if(writeOpt == mcPRL::PGTDEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)) {
          cerr << __FILE__ << " function:" << __FUNCTION__
              << " Error: unable to find SubCellspace (" \
              << subspcGlbID << ") to delete" << endl;
        }
      }
    }
  } // end -- for(iOutLyr)

  if((writeOpt == mcPRL::PARA_WRITING ||
      writeOpt == mcPRL::PARADEL_WRITING ||
      writeOpt == mcPRL::PGT_WRITING ||
      writeOpt == mcPRL::PGTDEL_WRITING) &&
     _prc.hasWriter()) {
    MPI_Send(&req, 1, MPI_INT, 1, mcPRL::REQUEST_TAG, _prc.comm());
    MPI_Send(&subspcGlbID, 1, MPI_INT, 1, mcPRL::SUBMIT_TAG, _prc.comm());
  }

  return true;
}

bool mcPRL::DataManager::
_finalWriteSubspcs(mcPRL::Transition &trans,
                   mcPRL::WritingOption writeOpt) {
  if(writeOpt == mcPRL::NO_WRITING) {
    return true;
  }

  if((writeOpt == mcPRL::CENT_WRITING || writeOpt == mcPRL::CENTDEL_WRITING) &&
     !_prc.hasWriter() &&
     !_prc.isMaster()) {
    mcPRL::TaskSignal req = mcPRL::SUBMIT_SGNL;
    const IntVect *pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
    if(pvSubspcIDs != NULL) {
      for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
        int subspcGlbID = pvSubspcIDs->at(iSubspc);

        const vector<string> &vOutLyrNames = trans.getOutLyrNames();
        for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
          const string &outLyrName = vOutLyrNames[iOutLyr];
          mcPRL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
          if(pOutLyr == NULL) {
            return false;
          }
          if(!pOutLyr->isDecomposed()) {
            cerr << __FILE__ << " function:" << __FUNCTION__
                << " Error: unable to write a SubCellspace within a non-decomposed Layer (" \
                << outLyrName << ")" \
                << endl;
            return false;
          }

          int lyrID = getLayerIdxByName(outLyrName.c_str(), true);
          MPI_Send(&req, 1, MPI_INT, 0, mcPRL::REQUEST_TAG, _prc.comm());
          int sendSubspcInfo[2];
          sendSubspcInfo[0] = lyrID;
          sendSubspcInfo[1] = subspcGlbID;
          MPI_Send(sendSubspcInfo, 2, MPI_INT, 0, mcPRL::SUBMIT_TAG, _prc.comm());
          if(!_iTransfSubspaceBegin(_prc.id(), 0, outLyrName, subspcGlbID)) {
            return false;
          }

        } // end -- for(iOutLyr)
      } // end -- for(iSubspc)
    } // end -- if(pvSubspcIDs != NULL)
  }

  return true;
}

mcPRL::EvaluateReturn mcPRL::DataManager::
_evalNone(mcPRL::Transition &trans,
          int subspcGlbID,
          bool ifInitNoData) {
  if(!_initEvaluate(trans, subspcGlbID, mcPRL::EVAL_WORKBR, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to initialize the Transition" \
         << endl;
    return mcPRL::EVAL_FAILED;
  }
  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcPRL::EVAL_FAILED;
  }

  return mcPRL::EVAL_SUCCEEDED;
}

mcPRL::EvaluateReturn mcPRL::DataManager::
_evalRandomly(mcPRL::Transition &trans,
              int subspcGlbID,
              bool ifInitNoData) {
  if(!_initEvaluate(trans, subspcGlbID, mcPRL::EVAL_WORKBR, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to initialize the Transition" \
         << endl;
    return mcPRL::EVAL_FAILED;
  }
  if(trans.needExchange()) {
    trans.setUpdateTracking(true);
  }

  mcPRL::Layer *pPrmLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrmLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to find the primary Layer" \
         << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::Cellspace *pPrmCellspace = NULL;
  if(subspcGlbID == mcPRL::ERROR_ID) { // if evaluate the whole Cellspace
    pPrmCellspace = pPrmLyr->cellspace();
  }
  else { // if evaluate a SubCellspace
    pPrmCellspace = pPrmLyr->subCellspace_glbID(subspcGlbID, true);
  }
  if(pPrmCellspace == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace" \
        << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::CoordBR workBR;
  if(subspcGlbID == mcPRL::ERROR_ID) { // if evaluate the whole Cellspace
    if(!pPrmCellspace->info()->calcWorkBR(&workBR, trans.getNbrhd(), true)) {
      return mcPRL::EVAL_FAILED;
    }
  }
  else { // if evaluate a SubCellspace
    workBR = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->workBR();
  }
  mcPRL::EvaluateReturn done = trans.evalRandomly(workBR);

  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcPRL::EVAL_FAILED;
  }

  return done;
}

mcPRL::EvaluateReturn mcPRL::DataManager::
_evalSelected(mcPRL::Transition &trans,
              const mcPRL::LongVect &vGlbIdxs,
              int subspcGlbID,
              mcPRL::EvaluateBR br2Eval,
              bool ifInitNoData) {
  if(br2Eval != mcPRL::EVAL_WORKBR) {
    if(subspcGlbID == mcPRL::ERROR_ID) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: cannot evaluate the Edges or Interior of a whole Cellspace" \
          << endl;
      return mcPRL::EVAL_FAILED;
    }
    if(trans.getOutLyrNames().empty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: no output Layers to evaluate Edges or Interior" \
           << endl;
      return mcPRL::EVAL_FAILED;
    }
    if(!trans.isOutLyr(trans.getPrimeLyrName())) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the primary Layer (" << trans.getPrimeLyrName() \
           << ") is NOT an output Layer" << endl;
      return mcPRL::EVAL_FAILED;
    }
  }

  if(!_initEvaluate(trans, subspcGlbID, br2Eval, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize the Transition" \
        << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::Layer *pPrmLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrmLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to find the primary Layer" \
        << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::Cellspace *pPrmCellspace = NULL;
  if(subspcGlbID == mcPRL::ERROR_ID) { // if evaluate the whole Cellspace
    pPrmCellspace = pPrmLyr->cellspace();
  }
  else { // if evaluate a SubCellspace
    pPrmCellspace = pPrmLyr->subCellspace_glbID(subspcGlbID, true);
  }
  if(pPrmCellspace == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace" \
        << endl;
    return mcPRL::EVAL_FAILED;
  }

  mcPRL::CoordBR workBR;
  const mcPRL::CoordBR *pWorkBR;
  mcPRL::EvaluateReturn done = mcPRL::EVAL_SUCCEEDED;
  switch(br2Eval) {
    case mcPRL::EVAL_WORKBR:
      if(subspcGlbID == mcPRL::ERROR_ID) { // if evaluate the whole Cellspace
        if(!pPrmCellspace->info()->calcWorkBR(&workBR, trans.getNbrhd(), true)) {
          return mcPRL::EVAL_FAILED;
        }
        done = trans.evalSelected(workBR, vGlbIdxs);
      }
      else { // if evaluate a SubCellspace
        pWorkBR = &(((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->workBR());
        for(int iIdx = 0; iIdx < vGlbIdxs.size(); iIdx++) {
          mcPRL::CellCoord lclCoord = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->glbIdx2lclCoord(vGlbIdxs[iIdx]);
          if(pWorkBR->ifContain(lclCoord)) {
            done = trans.evaluate(lclCoord);
            if(done == mcPRL::EVAL_FAILED ||
               done == mcPRL::EVAL_TERMINATED) {
              return done;
            }
          }
        }
      }
      break;
    case mcPRL::EVAL_EDGES:
      for(int iEdge = 0; iEdge < ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->nEdges(); iEdge++) {
        pWorkBR = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->edgeBR(iEdge);
        if(pWorkBR != NULL) {
          for(int iIdx = 0; iIdx < vGlbIdxs.size(); iIdx++) {
            mcPRL::CellCoord lclCoord = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->glbIdx2lclCoord(vGlbIdxs[iIdx]);
            if(pWorkBR->ifContain(lclCoord)) {
              done = trans.evaluate(lclCoord);
              if(done == mcPRL::EVAL_FAILED ||
                 done == mcPRL::EVAL_TERMINATED) {
                return done;
              }
            }
          } // end -- for(iIdx)
        }
      } // end -- for(iEdge)
      break;
    case mcPRL::EVAL_INTERIOR:
      pWorkBR = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->interiorBR();
      if(pWorkBR != NULL) {
        for(int iIdx = 0; iIdx < vGlbIdxs.size(); iIdx++) {
          mcPRL::CellCoord lclCoord = ((mcPRL::SubCellspace *)pPrmCellspace)->subInfo()->glbIdx2lclCoord(vGlbIdxs[iIdx]);
          if(pWorkBR->ifContain(lclCoord)) {
            done = trans.evaluate(lclCoord);
            if(done == mcPRL::EVAL_FAILED ||
               done == mcPRL::EVAL_TERMINATED) {
              return done;
            }
          }
        }
      }
      break;
    default:
      break;
  }

  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcPRL::EVAL_FAILED;
  }

  return done;
}

mcPRL::EvaluateReturn mcPRL::DataManager::
_masterTF(mcPRL::Transition &trans,
		  mcPRL::ReadingOption readOpt,
		  mcPRL::WritingOption writeOpt) {
  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  mcPRL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  mcPRL::TaskSignal req;
  MPI_Status status;

  bool delSentSpc = (readOpt == mcPRL::CENTDEL_READING) ? true : false;
  bool writeRecvSpc = (writeOpt != mcPRL::NO_WRITING) ? true : false;
  bool delRecvSpc = (writeOpt == mcPRL::CENTDEL_WRITING) ? true : false;

  while(_nActiveWorkers > 0) {
    MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcPRL::REQUEST_TAG, _prc.comm(), &status);
    int workerID = status.MPI_SOURCE;

    if((mcPRL::TaskSignal)req == mcPRL::NEWTASK_SGNL) { // worker requesting new task
      if(!_mOwnerships.allMapped()) { // there still are tasks
        int mappedSubspcID = _mOwnerships.mappingNextTo(workerID);
        if(mappedSubspcID == ERROR_ID) {
          _prc.abort();
          return mcPRL::EVAL_FAILED;
        }
        MPI_Send(&mappedSubspcID, 1, MPI_INT, workerID, mcPRL::DSTRBT_TAG, _prc.comm());

        if(readOpt == mcPRL::CENT_READING ||
           readOpt == mcPRL::CENTDEL_READING) { // centralized reading
          const vector<string> &vInLyrNames = trans.getInLyrNames();
          for(int iLyr = 0; iLyr < vInLyrNames.size(); iLyr++) {
            const string &inLyrName = vInLyrNames[iLyr];
            mcPRL::Layer *pLyr = getLayerByName(inLyrName.c_str(), true);
            if(pLyr == NULL) {
              _prc.abort();
              return mcPRL::EVAL_FAILED;
            }
            if(pLyr->isDecomposed()) { // decomposed Layer
              mcPRL::SubCellspace *pSubspc = pLyr->subCellspace_glbID(mappedSubspcID, false);
              if(pSubspc == NULL) {
                if(!pLyr->loadSubCellspaceByGDAL(mappedSubspcID)) {
                  _prc.abort();
                  return mcPRL::EVAL_FAILED;
                }
              }
              if(!_iTransfSubspaceBegin(_prc.id(), workerID, inLyrName, mappedSubspcID)) {
                _prc.abort();
                return mcPRL::EVAL_FAILED;
              }
            } // end -- if(pLyr->isDecomposed())
          } // end -- for(int iLyr = 0; iLyr < vLyrIDs.size(); iLyr++)
        } // end -- if(readOpt == mcPRL::CENT_READING || readOpt == mcPRL::CENTDEL_READING)
      } // end -- if(!_mOwnerships.allMapped())

      else { // all tasks are assigned
        mcPRL::TaskSignal resp = mcPRL::QUIT_SGNL;
        MPI_Send(&resp, 1, MPI_INT, workerID, mcPRL::DSTRBT_TAG, _prc.comm());
      }
    } // end -- if((TaskSignal)req == NEWTASK_SGNL)

    else if((mcPRL::TaskSignal)req == mcPRL::SUBMIT_SGNL) { // worker submitting result
      if(writeOpt == mcPRL::PARA_WRITING ||
         writeOpt == mcPRL::PARADEL_WRITING) { // parallel writing
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
        if(!mergeTmpSubspcByGDAL(trans, recvSubspcID)) {
          _prc.abort();
          return mcPRL::EVAL_FAILED;
        }
      }
      else if(writeOpt == mcPRL::CENT_WRITING ||
              writeOpt == mcPRL::CENTDEL_WRITING) { // centralized writing
        int recvSubspcInfo[2];
        MPI_Recv(recvSubspcInfo, 2, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
        int lyrID = recvSubspcInfo[0];
        int subspcGlbID = recvSubspcInfo[1];
        string lyrName = getLayerNameByIdx(lyrID);

        if(!_iTransfSubspaceBegin(workerID, _prc.id(), lyrName, subspcGlbID)) {
          _prc.abort();
          return EVAL_FAILED;
        }
      }
      else if(writeOpt == mcPRL::PGT_WRITING ||
              writeOpt == mcPRL::PGTDEL_WRITING) {
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
      }
    } // end -- if((TaskSignal)req == SUBMIT_SGNL)

    else if((mcPRL::TaskSignal)req == mcPRL::QUIT_SGNL) { // worker quitting
      //cout << "Process "<< workerID << " quit" << endl;
      _nActiveWorkers--;
    }

    // test pending transfers, only for centralized I/O
    if(!_vBcstSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltBcstIDs, mcPRL::BCST_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }
    if(!_vSendSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }

  } // end -- while(_nActiveWorkers > 0)

  // complete all pending transfers, only for centralized I/O
  while(!_vBcstSpcReqs.empty() && nCmpltBcsts != MPI_UNDEFINED) {
    nCmpltBcsts = _iTransfSpaceTest(vCmpltBcstIDs, mcPRL::BCST_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }
  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }

  _clearTransfSpaceReqs();

  return mcPRL::EVAL_SUCCEEDED;
}
mcPRL::EvaluateReturn mcPRL::DataManager::
_writerTF(mcPRL::Transition &trans,
          mcPRL::WritingOption writeOpt) {
  int nCmpltRecvs = 0;
  mcPRL::IntVect vCmpltRecvIDs;
  mcPRL::TaskSignal req;
  MPI_Status status;

  bool delSentSpc = false;
  bool writeRecvSpc = (writeOpt != mcPRL::NO_WRITING) ? true : false;
  bool delRecvSpc = (writeOpt == mcPRL::CENTDEL_WRITING) ? true : false;

  while(_nActiveWorkers > 0) {
    MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcPRL::REQUEST_TAG, _prc.comm(), &status);
    int workerID = status.MPI_SOURCE;

    if((mcPRL::TaskSignal)req == mcPRL::SUBMIT_SGNL) { // worker submitting result
      if(writeOpt == mcPRL::PARA_WRITING ||
         writeOpt == mcPRL::PARADEL_WRITING) { // parallel writing
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
        if(!mergeTmpSubspcByGDAL(trans, recvSubspcID)) {
          _prc.abort();
          return mcPRL::EVAL_FAILED;
        }
      }
      else if(writeOpt == mcPRL::CENT_WRITING ||
              writeOpt == mcPRL::CENTDEL_WRITING) { // centralized writing
        int recvSubspcInfo[2];
        MPI_Recv(recvSubspcInfo, 2, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
        int lyrID = recvSubspcInfo[0];
        int subspcGlbID = recvSubspcInfo[1];
        string lyrName = getLayerNameByIdx(lyrID);

        if(!_iTransfSubspaceBegin(workerID, _prc.id(), lyrName, subspcGlbID)) {
          _prc.abort();
          return EVAL_FAILED;
        }
      }
      else if(writeOpt == mcPRL::PGT_WRITING ||
              writeOpt == mcPRL::PGTDEL_WRITING) {
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
      }
    } // end -- if((TaskSignal)req == SUBMIT_SGNL)

    else if((mcPRL::TaskSignal)req == mcPRL::QUIT_SGNL) { // worker quitting
      //cout << "Process "<< workerID << " quit" << endl;
      _nActiveWorkers--;
    }

    // test pending transfers, only for centralized I/O
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }

  } // end -- while(_nActiveWorkers > 0)

  // complete all pending transfers, only for centralized I/O
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }

  _clearTransfSpaceReqs();

  return mcPRL::EVAL_SUCCEEDED;
}


mcPRL::EvaluateReturn mcPRL::DataManager::
_masterST(mcPRL::Transition &trans,
          mcPRL::WritingOption writeOpt) {
  mcPRL::TaskSignal req;
  MPI_Status status;

  if(!_prc.hasWriter()) {
    if(writeOpt == mcPRL::PARA_WRITING ||
       writeOpt == mcPRL::PARADEL_WRITING) { // parallel writing
      while(_nActiveWorkers > 0) {
        MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcPRL::REQUEST_TAG, _prc.comm(), &status);
        int workerID = status.MPI_SOURCE;
        if((mcPRL::TaskSignal)req == mcPRL::QUIT_SGNL) { // worker quitting
          //cout << "Process "<< workerID << " quit" << endl;
          _nActiveWorkers--;
        }
      } // end -- while(_nActiveWorkers > 0)
      if(!mergeAllTmpSubspcsByGDAL(trans)) {
        _prc.abort();
        return mcPRL::EVAL_FAILED;;
      }
    } // end -- parallel writing

    else if(writeOpt == mcPRL::CENT_WRITING ||
            writeOpt == mcPRL::CENTDEL_WRITING) { // centralized writing
      bool delRecvSpc = (writeOpt == mcPRL::CENTDEL_WRITING) ? true : false;
      int nCmpltRecvs = 0;
      mcPRL::IntVect vCmpltRecvIDs;

      while(_nActiveWorkers > 0) {
        MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcPRL::REQUEST_TAG, _prc.comm(), &status);
        int workerID = status.MPI_SOURCE;
        if((mcPRL::TaskSignal)req == mcPRL::SUBMIT_SGNL) { // worker submitting result
          int recvSubspcInfo[2];
          MPI_Recv(recvSubspcInfo, 2, MPI_INT, workerID, mcPRL::SUBMIT_TAG, _prc.comm(), &status);
          int lyrID = recvSubspcInfo[0];
          int subspcGlbID = recvSubspcInfo[1];
          string lyrName = getLayerNameByIdx(lyrID);

          if(!_iTransfSubspaceBegin(workerID, _prc.id(), lyrName, subspcGlbID)) {
            _prc.abort();
            return EVAL_FAILED;
          }
        } // end -- if((TaskSignal)req == SUBMIT_SGNL)

        else if((mcPRL::TaskSignal)req == mcPRL::QUIT_SGNL) { // worker quitting
          //cout << "Process "<< workerID << " quit" << endl;
          _nActiveWorkers--;
        }

        if(!_vRecvSpcReqs.empty()) {
          _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, true, true, delRecvSpc);
        }
      } // end -- while(_nActiveWorkers > 0)

      // write local SubCellspaces
      const IntVect *pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
      if(pvSubspcIDs != NULL) {
        for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
          int subspcGlbID = pvSubspcIDs->at(iSubspc);
          const vector<string> &vOutLyrNames = trans.getOutLyrNames();
          for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
            const string &outLyrName = vOutLyrNames[iOutLyr];
            mcPRL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
            if(pOutLyr == NULL) {
              _prc.abort();
              return EVAL_FAILED;
            }
            if(!pOutLyr->writeSubCellspaceByGDAL(subspcGlbID)) {
              _prc.abort();
              return EVAL_FAILED;
            }
          } // end -- for(iOutLyr)
        } // end -- for(iSubspc)
      }

      while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
        nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, true, true, delRecvSpc);
      }

    } // end -- centralized writing

    else if(writeOpt == mcPRL::PGT_WRITING ||
            writeOpt == mcPRL::PGTDEL_WRITING) { // parallel writing
      while(_nActiveWorkers > 0) {
        MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcPRL::REQUEST_TAG, _prc.comm(), &status);
        int workerID = status.MPI_SOURCE;
        if((mcPRL::TaskSignal)req == mcPRL::QUIT_SGNL) { // worker quitting
          //cout << "Process "<< workerID << " quit" << endl;
          _nActiveWorkers--;
        }
      } // end -- while(_nActiveWorkers > 0)
    } // end -- parallel writing
  }  // end -- if(!_prc.hasWriter())

  _clearTransfSpaceReqs();

  return mcPRL::EVAL_SUCCEEDED;
}

mcPRL::EvaluateReturn mcPRL::DataManager::
_writerST(mcPRL::Transition &trans,
          mcPRL::WritingOption writeOpt) {
  return _writerTF(trans, writeOpt);
}

bool mcPRL::DataManager::
_calcExchangeBRs(mcPRL::Transition &trans) {
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    if(!pOutLyr->calcAllBRs(trans.onlyUpdtCtrCell(), getNbrhdByName(trans.getNbrhdName().c_str()))) {
      return false;
    }
  }
  return true;
}

bool mcPRL::DataManager::
_calcExchangeRoutes(mcPRL::Transition &trans) {
  _mSendRoutes.clear();
  _mRecvRoutes.clear();

  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  const mcPRL::IntVect *pvAssignments = _mOwnerships.subspcIDsOnPrc(_prc.id());

  if(trans.needExchange() && !vOutLyrNames.empty() &&
     pvAssignments != NULL && !pvAssignments->empty()) {
    mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrNames[0].c_str(), true); // only one output Layer is needed for calculating the exchange map
    if(pOutLyr == NULL) {
      return false;
    }
    for(int iAssignment = 0; iAssignment < pvAssignments->size(); iAssignment++) {
      int subspcID = pvAssignments->at(iAssignment);
      const mcPRL::SubCellspaceInfo *pSubspcInfo = pOutLyr->subCellspaceInfo_glbID(subspcID, true);
      if(pSubspcInfo == NULL) {
        return false;
      }

      for(int iDir = 0; iDir < pSubspcInfo->nNbrDirs(); iDir++) {
        const mcPRL::IntVect &vNbrSpcIDs = pSubspcInfo->nbrSubSpcIDs(iDir);
        for(int iNbr = 0; iNbr < vNbrSpcIDs.size(); iNbr++) {
          int nbrSubspcID = vNbrSpcIDs[iNbr];
          int nbrPrc = _mOwnerships.findPrcBySubspcID(nbrSubspcID);

          const mcPRL::CoordBR *pSendBR = pSubspcInfo->sendBR(iDir, iNbr);
          if(pSendBR != NULL && nbrPrc >= 0) {
            if(!_mSendRoutes.add(subspcID, iDir, iNbr, nbrPrc)) {
              return false;
            }
          }

          const mcPRL::SubCellspaceInfo *pNbrSubspcInfo = pOutLyr->subCellspaceInfo_glbID(nbrSubspcID, true);
          if(pNbrSubspcInfo == NULL) {
            return false;
          }
          int iOppDir = pNbrSubspcInfo->oppositeDir(iDir);
          const mcPRL::IntVect &vOppNbrSpcIDs = pNbrSubspcInfo->nbrSubSpcIDs(iOppDir);
          mcPRL::IntVect::const_iterator itrOppNbrSpcID = std::find(vOppNbrSpcIDs.begin(), vOppNbrSpcIDs.end(), subspcID);
          if(itrOppNbrSpcID != vOppNbrSpcIDs.end()) {
            int iOppNbr = itrOppNbrSpcID - vOppNbrSpcIDs.begin();
            const mcPRL::CoordBR *pOppSendBR = pNbrSubspcInfo->sendBR(iOppDir, iOppNbr);
            if(pOppSendBR != NULL && nbrPrc >= 0) {
              if(!_mRecvRoutes.add(subspcID, iDir, iNbr, nbrPrc)) {
                return false;
              }
            }
          }
        } // end -- for(iNbr)
      } // end -- for(iDir)
    } // end -- for(iAssignment)
  }

  return true;
}

bool mcPRL::DataManager::
_loadCellStream(mcPRL::CellStream &cells2load) {
  long cellGlbIdx;
  void *aCellVal;
  const vector<pair<unsigned int, unsigned int> >& vLyrInfos = cells2load.getLayerInfos();
  for(int iLyr = 0; iLyr < vLyrInfos.size(); iLyr++) {
    unsigned int lyrID = vLyrInfos[iLyr].first;
    unsigned int dataSize = vLyrInfos[iLyr].second;
    int nCells = cells2load.getNumCellsOnLayer(lyrID);
    void* aCells = cells2load.getCellsOnLayer(lyrID);

    mcPRL::Layer *pLyr = getLayerByIdx(lyrID, true);
    if(pLyr == NULL) {
      return false;
    }
    if(!pLyr->loadCellStream(aCells, nCells)) {
      return false;
    }
  } // end --- for(iLyr)
  return true;
}

bool mcPRL::DataManager::
_makeCellStream(mcPRL::Transition &trans) {
  _vSendCellReqs.clear();
  _vRecvCellReqs.clear();
  _mSendCells.clear();
  _mRecvCells.clear();

  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  if(vOutLyrNames.empty()) {
    return true;
  }

  mcPRL::ExchangeMap::iterator itrSendRoute = _mSendRoutes.begin();
  while(itrSendRoute != _mSendRoutes.end()) {
    int toPrc = itrSendRoute->first;
    vector<mcPRL::ExchangeNode> &vSendNodes = itrSendRoute->second;
    for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
      mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
      if(pOutLyr == NULL) {
        return false;
      }
      int lyrID = getLayerIdxByName(vOutLyrNames[iOutLyr].c_str(), true);
      unsigned int lyrDataSize = pOutLyr->dataSize();
      if(!_mSendCells[toPrc].addLayer(lyrID, lyrDataSize)) {
        return false;
      }

      for(int iNode = 0; iNode < vSendNodes.size(); iNode++) {
        int subspcID = vSendNodes[iNode].subspcGlbID;
        int iDir = vSendNodes[iNode].iDirection;
        int iNbr = vSendNodes[iNode].iNeighbor;
        mcPRL::SubCellspace *pSubspc = pOutLyr->subCellspace_glbID(subspcID, true);
        if(pSubspc == NULL) {
          return false;
        }
        const mcPRL::CoordBR *pSendBR = pSubspc->subInfo()->sendBR(iDir, iNbr);
        const mcPRL::LongVect &vUpdtIdxs = pSubspc->getUpdatedIdxs();
        for(int iIdx = 0; iIdx < vUpdtIdxs.size(); iIdx++) {
          long lclIdx = vUpdtIdxs[iIdx];
          if(pSendBR->ifContain(lclIdx, pSubspc->info()->dims())) {
            long glbIdx = pSubspc->subInfo()->lclIdx2glbIdx(lclIdx);
            void *aCellVal = pSubspc->at(lclIdx);
            if(!_mSendCells[toPrc].addCell(glbIdx, aCellVal)) {
              return false;
            }
            //cout << toPrc << " " << pOutLyr->name() << " [" << glbIdx << ", " << pSubspc->atAs<unsigned short>(lclIdx) << "]" << endl;
          }
        } // end -- for(iIdx)
      } // end -- for(iNode)
    } // end -- for(iOutLyr)

    if(toPrc != _prc.id()) { // if toPrc is not myself
      _vSendCellReqs.resize(_vSendCellReqs.size() + 1);
      MPI_Isend(&(_mSendCells[toPrc].getCellCounts().at(0)),
                _mSendCells[toPrc].getNumLayers(),
                MPI_INT, toPrc, _prc.id(), _prc.comm(),
                &(_vSendCellReqs.back()));
    }

    itrSendRoute++;
  } // end -- while(itrSendRoute != _mSendRoutes.end())

  // clear all update tracks
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    pOutLyr->clearUpdateTracks();
  }

  mcPRL::ExchangeMap::iterator itrRecvRoute = _mRecvRoutes.begin();
  while(itrRecvRoute != _mRecvRoutes.end()) {
    int fromPrc = itrRecvRoute->first;
    vector<mcPRL::ExchangeNode> &vRecvNodes = itrRecvRoute->second;
    for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
      mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
      if(pOutLyr == NULL) {
        return false;
      }
      int lyrID = getLayerIdxByName(vOutLyrNames[iOutLyr].c_str(), true);
      unsigned int lyrDataSize = pOutLyr->dataSize();
      if(!_mRecvCells[fromPrc].addLayer(lyrID, lyrDataSize)) {
        return false;
      }
    } // end -- for(iOutLyr)

    if(fromPrc != _prc.id()) {
      _vRecvCellReqs.resize(_vRecvCellReqs.size() + 1);
      MPI_Irecv(&(_mRecvCells[fromPrc].getCellCounts().at(0)),
                _mRecvCells[fromPrc].getNumLayers(),
                MPI_INT, fromPrc, fromPrc, _prc.comm(),
                &(_vRecvCellReqs.back()));
    }

    itrRecvRoute++;
  } // end -- while(itrRecvRoute != _mRecvRoutes.end())

  return true;
}

bool mcPRL::DataManager::
_iExchangeBegin() {
  vector<MPI_Status> vSendStats(_vSendCellReqs.size());
  MPI_Waitall(_vSendCellReqs.size(), &(_vSendCellReqs[0]), &(vSendStats[0]));
  _vSendCellReqs.clear();

  mcPRL::ExchangeMap::iterator itrSendRoute = _mSendRoutes.begin();
  while(itrSendRoute != _mSendRoutes.end()) {
    int toPrc = itrSendRoute->first;
    if(toPrc != _prc.id()) {
      _vSendCellReqs.resize(_vSendCellReqs.size() + 1);
      MPI_Isend(_mSendCells[toPrc].getStream(),
                _mSendCells[toPrc].size(),
                MPI_CHAR, toPrc, _prc.id(), _prc.comm(),
                &(_vSendCellReqs.back()));
    }

    itrSendRoute++;
  } // end -- while(itrSendRoute != _mSendRoutes.end())

  vector<MPI_Status> vRecvStats(_vRecvCellReqs.size());
  MPI_Waitall(_vRecvCellReqs.size(), &(_vRecvCellReqs[0]), &(vRecvStats[0]));
  _vRecvCellReqs.clear();

  mcPRL::ExchangeMap::iterator itrRecvRoute = _mRecvRoutes.begin();
  while(itrRecvRoute != _mRecvRoutes.end()) {
    int fromPrc = itrRecvRoute->first;
    if(fromPrc != _prc.id()) {
      if(!_mRecvCells[fromPrc].resize()) {
        return false;
      }
      _vRecvCellReqs.resize(_vRecvCellReqs.size() + 1);
      MPI_Irecv(_mRecvCells[fromPrc].getStream(),
                _mRecvCells[fromPrc].size(),
                MPI_CHAR, fromPrc, fromPrc, _prc.comm(),
                &(_vRecvCellReqs.back()));
    }
    else {
      _mRecvCells[_prc.id()] = _mSendCells[_prc.id()];
    }

    itrRecvRoute++;
  } // end --- while(itrRecvRoute != _mRecvRoutes.end())

  return true;
}

bool mcPRL::DataManager::
_iExchangeEnd() {
  vector<MPI_Status> vSendStats(_vSendCellReqs.size());
  MPI_Waitall(_vSendCellReqs.size(), &(_vSendCellReqs[0]), &(vSendStats[0]));
  _vSendCellReqs.clear();
  _mSendCells.clear();

  vector<MPI_Status> vRecvStats(_vRecvCellReqs.size());
  MPI_Waitall(_vRecvCellReqs.size(), &(_vRecvCellReqs[0]), &(vRecvStats[0]));
  _vRecvCellReqs.clear();
  map<int, mcPRL::CellStream>::iterator itrRecvCells = _mRecvCells.begin();
  while(itrRecvCells != _mRecvCells.end()) {
    mcPRL::CellStream &recvCells = itrRecvCells->second;
    if(!_loadCellStream(recvCells)) {
      return false;
    }
    itrRecvCells++;
  }
  _mRecvCells.clear();

  return true;
}

/****************************************************
*                 Public Methods                    *
****************************************************/
mcPRL::DataManager::
DataManager()
  :_nActiveWorkers(0) {
  GDALAllRegister();
}

mcPRL::DataManager::
~DataManager() {
  clearLayers();
  clearNbrhds();
}

bool mcPRL::DataManager::
initMPI(MPI_Comm comm,
        bool hasWriter) {
  return _prc.set(comm, hasWriter);
}

void mcPRL::DataManager::
finalizeMPI() {
  _prc.finalize();
}

mcPRL::Process& mcPRL::DataManager::
mpiPrc() {
  return _prc;
}

int mcPRL::DataManager::
nLayers() const {
  return _lLayers.size();
}

mcPRL::Layer* mcPRL::DataManager::
addLayer(const char *aLyrName) {
  mcPRL::Layer *pLyr = getLayerByName(aLyrName, false);
  if(pLyr != NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__
         << " Error: Layer name (" << aLyrName << ") already exists in DataManager. " \
         << "Failed to add new Layer."<< endl;
  }
  else {
    
    _lLayers.push_back(mcPRL::Layer(aLyrName));
    pLyr = &(_lLayers.back());
   
  }
  return pLyr;
}


/*---------------------------GDAL----------------------------------------------------*/
mcPRL::Layer* mcPRL::DataManager::
addLayerByGDAL(const char *aLyrName,
               const char *aGdalFileName,
               int iBand,
               bool pReading) {
  mcPRL::Layer *pLyr = addLayer(aLyrName);

  if(pReading) { // parallel reading: all processes read the file in parallel
    if(!pLyr->openGdalDS(aGdalFileName, GA_ReadOnly)) {
      _prc.abort();
      return pLyr;
    }
    pLyr->dsBand(iBand);

    if(!pLyr->initCellspaceInfoByGDAL()) {
      _prc.abort();
      return pLyr;
    }
  }
  else { // master process reads the file and distribute info to other processes
    vector<char> vBuf;
    int bufSize;

    if(_prc.isMaster()) {
      if(!pLyr->openGdalDS(aGdalFileName, GA_ReadOnly)) {
        _prc.abort();
        return pLyr;
      }
      pLyr->dsBand(iBand);

      if(!pLyr->initCellspaceInfoByGDAL()) {
        _prc.abort();
        return pLyr;
      }

      pLyr->cellspaceInfo()->add2Buf(vBuf);
      bufSize = vBuf.size();
    }
    else {
      pLyr->initCellspaceInfo();
      pLyr->dsBand(iBand); // worker will know this Layer has a GDAL dataset
    }

    if(MPI_Bcast(&bufSize, 1, MPI_INT, 0, _prc.comm()) != MPI_SUCCESS) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: failed to broadcast CellspaceInfo buffer size" \
          << endl;
      _prc.abort();
      return pLyr;
    }

    if(!_prc.isMaster()) {
      vBuf.resize(bufSize);
    }

    if(MPI_Bcast(&(vBuf[0]), bufSize, MPI_CHAR, 0, _prc.comm()) != MPI_SUCCESS) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: failed to broadcast CellspaceInfo" \
          << endl;
      _prc.abort();
      return pLyr;
    }

    if(!_prc.isMaster()) {
      if(!pLyr->cellspaceInfo()->initFromBuf(vBuf)) {
        _prc.abort();
        return pLyr;
      }
    }
  }
  return pLyr;
}

bool mcPRL::DataManager::
createLayerGDAL(const char *aLyrName,
                const char *aGdalFileName,
                const char *aGdalFormat,
                char **aGdalOptions) {
  mcPRL::Layer *pLyr = getLayerByName(aLyrName, true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  }

  if(_prc.hasWriter()) { // if a writer Process exists
    if(_prc.isWriter()) {
      if(!pLyr->createGdalDS(aGdalFileName, aGdalFormat, aGdalOptions)) {
        _prc.abort();
        return false;
      }
    }
    _prc.sync();
    if(!_prc.isWriter()) {
      pLyr->openGdalDS(aGdalFileName, GA_ReadOnly); // worker processes cannot write to the dataset
      pLyr->dsBand(1); // all Processes will know this layer has a GDAL dataset
    }
  }
  else { // if no writer Process exists
    if(_prc.isMaster()) {
      if(!pLyr->createGdalDS(aGdalFileName, aGdalFormat, aGdalOptions)) {
        _prc.abort();
        return false;
      }
    }
    _prc.sync();
    if(!_prc.isMaster()) {
      pLyr->openGdalDS(aGdalFileName, GA_ReadOnly); // worker processes cannot write to the dataset
      pLyr->dsBand(1); // all Processes will know this layer has a GDAL dataset
    }
  }

  return true;
}

void mcPRL::DataManager::
closeGDAL() {
  if(!_lLayers.empty()) {
    list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
    while(itrLyr != _lLayers.end()) {
      itrLyr->closeGdalDS();
      itrLyr++;
    }
  }
}

/*---------------------------hsj-------------PGIOL--------------------*/
mcPRL::Layer* mcPRL::DataManager::
addLayerByPGTIOL(const char *aLyrName,
                 const char *aPgtiolFileName,
                 int iBand,
                 bool pReading) {
  mcPRL::Layer *pLyr = addLayer(aLyrName);

  if(pReading) { // parallel reading: all processes read the file in parallel
    int prc = _prc.hasWriter()?1:0;
    if(!pLyr->openPgtiolDS(aPgtiolFileName, prc, PG_ReadOnly)) {
      _prc.abort();
      return pLyr;
    }
    pLyr->dsBand(iBand);

    if(!pLyr->initCellspaceInfoByPGTIOL()) {
      _prc.abort();
      return pLyr;
    }
  }
  else { // master process reads the file and distribute info to other processes
    vector<char> vBuf;
    int bufSize;

    if(_prc.isMaster()) {
      if(!pLyr->openPgtiolDS(aPgtiolFileName,_prc.id())) {
        _prc.abort();
        return pLyr;
      }
      pLyr->dsBand(iBand);

      if(!pLyr->initCellspaceInfoByPGTIOL()) {
        _prc.abort();
        return pLyr;
      }

      pLyr->cellspaceInfo()->add2Buf(vBuf);
      bufSize = vBuf.size();
    }
    else {
      pLyr->initCellspaceInfo();
      pLyr->dsBand(iBand); // worker will know this Layer has a GDAL dataset
    }

    if(MPI_Bcast(&bufSize, 1, MPI_INT, 0, _prc.comm()) != MPI_SUCCESS) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: failed to broadcast CellspaceInfo buffer size" \
          << endl;
      _prc.abort();
      return pLyr;
    }

    if(!_prc.isMaster()) {
      vBuf.resize(bufSize);
    }

    if(MPI_Bcast(&(vBuf[0]), bufSize, MPI_CHAR, 0, _prc.comm()) != MPI_SUCCESS) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: failed to broadcast CellspaceInfo" \
          << endl;
      _prc.abort();
      return pLyr;
    }

    if(!_prc.isMaster()) {
      if(!pLyr->cellspaceInfo()->initFromBuf(vBuf)) {
        _prc.abort();
        return pLyr;
      }
    }
  }
  return pLyr;
}

bool mcPRL::DataManager::
createLayerPGTIOL(const char *aLyrName,
                  const char *aPgtiolFileName,
                  char **aPgtiolOptions) {
  mcPRL::Layer *pLyr = getLayerByName(aLyrName, true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  } 
  if(_prc.hasWriter()) { // if a writer Process exists
      if(_prc.isWriter()) {
      if(!pLyr->createPgtiolDS(aPgtiolFileName,1,aPgtiolOptions)) {
        _prc.abort();
        return false;
      }
    }
    _prc.sync();
    if(!_prc.isWriter()) {
      pLyr->openPgtiolDS(aPgtiolFileName,1,PG_Update);
      pLyr->dsBand(1); // all Processes will know this layer has a pGTIOL dataset
    }
  }
  else { // if no writer Process exists
    
    if(_prc.isMaster()) {
      if(!pLyr->createPgtiolDS(aPgtiolFileName,0,aPgtiolOptions)) {
        _prc.abort();
        return false;
      }
    }
	 
    _prc.sync();
    if(!_prc.isMaster()) {
      pLyr->openPgtiolDS(aPgtiolFileName,0,PG_Update);
      pLyr->dsBand(1); // all Processes will know this layer has a pGTIOL dataset
    }
  }
  return true;
}

void mcPRL::DataManager::
closePGTIOL() {
  if(!_lLayers.empty()) {
    list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
    while(itrLyr != _lLayers.end()) {
      itrLyr->closePgtiolDS();
      itrLyr++;
    }
  }
}

void mcPRL::DataManager::
closeDatasets() {
  if(!_lLayers.empty()) {
    list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
    while(itrLyr != _lLayers.end()) {
      itrLyr->closeGdalDS();
      itrLyr->closePgtiolDS();
      itrLyr++;
    }
  }
}

bool mcPRL::DataManager::
rmvLayerByIdx(int lyrID,
              bool warning) {
  bool done = true;
  if(lyrID < 0 || lyrID >= _lLayers.size()) {
    done = false;
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: invalid Layer index (" << lyrID \
           << ")" << endl;
    }
  }
  else {
    list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
    advance(itrLyr, lyrID);
    _lLayers.erase(itrLyr);
  }
  return done;
}

bool mcPRL::DataManager::
rmvLayerByName(const char *aLyrName,
               bool warning) {
  bool done = false;
  list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
  while(itrLyr != _lLayers.end()) {
    if(strcmp(aLyrName, itrLyr->name()) == 0) {
      _lLayers.erase(itrLyr);
      done = true;
      break;
    }
    itrLyr++;
  }
  if(!done && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: invalid Layer name (" << aLyrName \
        << ")" << endl;
  }
  return done;
}

void mcPRL::DataManager::
clearLayers() {
  _lLayers.clear();
}

mcPRL::Layer* mcPRL::DataManager::
getLayerByIdx(int lyrID,
              bool warning) {
  mcPRL::Layer *pLayer = NULL;
  if(lyrID >= 0 && lyrID < _lLayers.size()) {
    list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
    advance(itrLyr, lyrID);
    pLayer = &(*itrLyr);
  }
  else {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Warning: NO Layer with index (" << lyrID \
           << ") was found" << endl;
    }
  }
  return pLayer;
}

const mcPRL::Layer* mcPRL::DataManager::
getLayerByIdx(int lyrID,
              bool warning) const {
  const mcPRL::Layer *pLayer = NULL;
  if(lyrID >= 0 && lyrID < _lLayers.size()) {
    list<mcPRL::Layer>::const_iterator itrLyr = _lLayers.begin();
    advance(itrLyr, lyrID);
    pLayer = &(*itrLyr);
  }
  else {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Warning: NO Layer with index (" << lyrID \
           << ") was found" << endl;
    }
  }
  return pLayer;
}

int mcPRL::DataManager::
getLayerIdxByName(const char *aLyrName,
                  bool warning) const {
  int lyrIdx = ERROR_ID, lyrCount = 0;
  bool found = false;
  list<mcPRL::Layer>::const_iterator itrLyr = _lLayers.begin();
  while(itrLyr != _lLayers.end()) {
    if(strcmp(aLyrName, itrLyr->name()) == 0) {
      lyrIdx = lyrCount;
      found = true;
      break;
    }
    lyrCount++;
    itrLyr++;
  }
  if(!found && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: NO Layer with name (" << aLyrName \
         << ") was found" << endl;
  }
  return lyrIdx;
}

mcPRL::Layer* mcPRL::DataManager::
getLayerByName(const char *aLyrName,
               bool warning) {
  mcPRL::Layer* pLyr = NULL;
  list<mcPRL::Layer>::iterator itrLyr = _lLayers.begin();
  while(itrLyr != _lLayers.end()) {
    if(strcmp(aLyrName, itrLyr->name()) == 0) {
      pLyr = &(*itrLyr);
      break;
    }
    itrLyr++;
  }
  if(pLyr == NULL && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Warning: NO Layer with name (" << aLyrName \
        << ") was found" << endl;
  }
  return pLyr;
}

const mcPRL::Layer* mcPRL::DataManager::
getLayerByName(const char *aLyrName,
               bool warning) const {
  const mcPRL::Layer* pLyr = NULL;
  list<mcPRL::Layer>::const_iterator itrLyr = _lLayers.begin();
  while(itrLyr != _lLayers.end()) {
    if(strcmp(aLyrName, itrLyr->name()) == 0) {
      pLyr = &(*itrLyr);
      break;
    }
    itrLyr++;
  }
  if(pLyr == NULL && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Warning: NO Layer with name (" << aLyrName \
        << ") was found" << endl;
  }
  return pLyr;
}

const char* mcPRL::DataManager::
getLayerNameByIdx(int lyrID,
                  bool warning) const {
  const char *aLyrName = NULL;
  const mcPRL::Layer *pLyr = getLayerByIdx(lyrID, warning);
  if(pLyr != NULL) {
    aLyrName = pLyr->name();
  }
  return aLyrName;
}
bool mcPRL::DataManager::
beginGetRand(const char *aLyrName)
{
	mcPRL::Layer *pLyr = getLayerByName(aLyrName, true);
	if(pLyr == NULL) {
		_prc.abort();
		return false;
	}
	   const mcPRL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
	   if(pvSubspcIDs != NULL) {
        for(int iSub = 0; iSub < pvSubspcIDs->size(); iSub++) {
          int mappedSubspcID = pvSubspcIDs->at(iSub);
          if(!pLyr->loadSubCellspaceByRAND(mappedSubspcID)) {
            return false;
          }
        }
      }
	   return true;
}
bool mcPRL::DataManager::
beginReadingLayer(const char *aLyrName,
                  mcPRL::ReadingOption readOpt) {
  mcPRL::Layer *pLyr = getLayerByName(aLyrName, true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  }

  if(!pLyr->isDecomposed()) { // non-decomposed Layer
    if(readOpt == mcPRL::PARA_READING) {
      if(!_prc.isMaster() && !_prc.isWriter()) {
        if(!pLyr->loadCellspaceByGDAL()) {
          return false;
        }
      }
    }
    else if(readOpt == mcPRL::CENT_READING ||
            readOpt == mcPRL::CENTDEL_READING) {
      if(_prc.isMaster()) {
        if(!pLyr->loadCellspaceByGDAL()) {
          _prc.abort();
          return false;
        }
      }
      if(!_iBcastCellspaceBegin(aLyrName)) {
        return false;
      }
    }
    else if(readOpt == mcPRL::PGT_READING) {
      if(!_prc.isMaster() && !_prc.isWriter()) {
        if(!pLyr->loadCellspaceByPGTIOL()) {
          return false;
        }
      }
    }
  } // end -- non-decomposed Layer

  else { // decomposed Layer
    if(readOpt == mcPRL::PARA_READING) { // parallel reading
      const mcPRL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
      if(pvSubspcIDs != NULL) {
        for(int iSub = 0; iSub < pvSubspcIDs->size(); iSub++) {
          int mappedSubspcID = pvSubspcIDs->at(iSub);
          if(!pLyr->loadSubCellspaceByGDAL(mappedSubspcID)) {
            return false;
          }
        }
      }
    } // end -- parallel reading

    else if(readOpt == mcPRL::CENT_READING ||
            readOpt == mcPRL::CENTDEL_READING) { // centralized reading
      if(_prc.isMaster()) { // master process
        mcPRL::IntVect vMappedSubspcIDs = _mOwnerships.mappedSubspcIDs();
        for(int iSub = 0; iSub < vMappedSubspcIDs.size(); iSub++) {
          int mappedSubspcID = vMappedSubspcIDs[iSub];
          int prcID = _mOwnerships.findPrcBySubspcID(mappedSubspcID);
          if(mappedSubspcID >= 0 && prcID >= 0) {
            if(!pLyr->loadSubCellspaceByGDAL(mappedSubspcID)) {
              return false;
            }
            if(prcID != _prc.id()) { // transfer the SubCellspace if it belongs to a worker
              if(!_iTransfSubspaceBegin(_prc.id(), prcID, aLyrName, mappedSubspcID)) {
                return false;
              }
            }
          }
        } // end -- for(iSub)
      } // end -- master process

      else if(!_prc.isWriter()){ // worker process
        const mcPRL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
        if(pvSubspcIDs != NULL) {
          for(int iSub = 0; iSub < pvSubspcIDs->size(); iSub++) {
            int mappedSubspcID = pvSubspcIDs->at(iSub);
            if(!_iTransfSubspaceBegin(0, _prc.id(), aLyrName, mappedSubspcID)) {
              return false;
            }
          }
        }
      } // end -- worker process
    } // end -- centralized reading

    else if(readOpt == mcPRL::PGT_READING) {
      const mcPRL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
      if(pvSubspcIDs != NULL) {
        for(int iSub = 0; iSub < pvSubspcIDs->size(); iSub++) {
          int mappedSubspcID = pvSubspcIDs->at(iSub);
          if(!pLyr->loadSubCellspaceByPGTIOL(mappedSubspcID)) {
            return false;
          }
        }
      }
    } // end -- pGTIOL reading

  } // end -- decomposed Layer
  return true;
}

void mcPRL::DataManager::
finishReadingLayers(mcPRL::ReadingOption readOpt) {
  // Complete all pending transfers, only for centralized I/O
  mcPRL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  bool delSentSpc = (readOpt == mcPRL::CENTDEL_READING) ? true : false;
  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  while(!_vBcstSpcReqs.empty() && nCmpltBcsts != MPI_UNDEFINED) {
    nCmpltBcsts = _iTransfSpaceTest(vCmpltBcstIDs, mcPRL::BCST_TRANSF, delSentSpc, false, false);
  }
  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcPRL::SEND_TRANSF, delSentSpc, false, false);                                   //´«Êý¾Ý
  }
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcPRL::RECV_TRANSF, delSentSpc, false, false);
  }

  _clearTransfSpaceReqs();
}

int mcPRL::DataManager::
nNbrhds() const {
  return _lNbrhds.size();
}

mcPRL::Neighborhood* mcPRL::DataManager::
addNbrhd(const char *aNbrhdName) {
  mcPRL::Neighborhood *pNbrhd = getNbrhdByName(aNbrhdName, false);
  if(pNbrhd != NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: Neighborhood name (" << aNbrhdName \
         << ") already exists in DataManager. Failed to add new Neighborhood" \
         << endl;
  }
  else {
    _lNbrhds.push_back(mcPRL::Neighborhood(aNbrhdName));
    pNbrhd = &(_lNbrhds.back());
  }
  return pNbrhd;
}

mcPRL::Neighborhood* mcPRL::DataManager::
addNbrhd(const mcPRL::Neighborhood &nbrhd) {
  const char *aNbrhdName = nbrhd.name();
  mcPRL::Neighborhood *pNbrhd = getNbrhdByName(aNbrhdName, false);
  if(pNbrhd != NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: Neighborhood name (" << aNbrhdName \
         << ") already exists in DataManager. Failed to add new Neighborhood" \
        << endl;
  }
  else {
    _lNbrhds.push_back(mcPRL::Neighborhood(nbrhd));
    pNbrhd = &(_lNbrhds.back());
  }
  return pNbrhd;
}

bool mcPRL::DataManager::
rmvNbrhdByIdx(int nbrhdID,
              bool warning) {
  bool done = true;
  if(nbrhdID < 0 || nbrhdID >= _lNbrhds.size()) {
    done = false;
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: invalid Neighborhood index (" << nbrhdID \
           << ")" << endl;
    }
  }
  else {
    list<mcPRL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
    advance(itrNbrhd, nbrhdID);
    _lNbrhds.erase(itrNbrhd);
  }
  return done;
}

bool mcPRL::DataManager::
rmvNbrhdByName(const char *aNbrhdName,
               bool warning) {
  bool done = false;
  list<mcPRL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
  while(itrNbrhd != _lNbrhds.end()) {
    if(strcmp(aNbrhdName, itrNbrhd->name()) == 0) {
      _lNbrhds.erase(itrNbrhd);
      done = true;
      break;
    }
    itrNbrhd++;
  }
  if(!done && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: invalid Neighborhood name (" << aNbrhdName \
        << ")" << endl;
  }
  return done;
}

void mcPRL::DataManager::
clearNbrhds() {
  _lNbrhds.clear();
}

mcPRL::Neighborhood* mcPRL::DataManager::
getNbrhdByIdx(int nbrhdID,
              bool warning) {
  mcPRL::Neighborhood *pNbrhd = NULL;
  if(nbrhdID >= 0 && nbrhdID < _lNbrhds.size()) {
    list<mcPRL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
    advance(itrNbrhd, nbrhdID);
    pNbrhd = &(*itrNbrhd);
  }
  else {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Warning: NO Neighborhood with index (" << nbrhdID \
           << ") was found" << endl;
    }
  }
  return pNbrhd;
}

const mcPRL::Neighborhood* mcPRL::DataManager::
getNbrhdByIdx(int nbrhdID,
              bool warning) const {
  const mcPRL::Neighborhood *pNbrhd = NULL;
  if(nbrhdID >= 0 && nbrhdID < _lNbrhds.size()) {
    list<mcPRL::Neighborhood>::const_iterator itrNbrhd = _lNbrhds.begin();
    advance(itrNbrhd, nbrhdID);
    pNbrhd = &(*itrNbrhd);
  }
  else {
    if(warning) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Warning: NO Neighborhood with index (" << nbrhdID \
           << ") was found" << endl;
    }
  }
  return pNbrhd;
}

int mcPRL::DataManager::
getNbrhdIdxByName(const char *aNbrhdName,
                  bool warning) {
  int nbrhdIdx = ERROR_ID, nbrhdCount = 0;
  bool found = false;
  list<mcPRL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
  while(itrNbrhd != _lNbrhds.end()) {
    if(strcmp(aNbrhdName, itrNbrhd->name()) == 0) {
      nbrhdIdx = nbrhdCount;
      found = true;
      break;
    }
    nbrhdCount++;
    itrNbrhd++;
  }
  if(!found && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: NO Neighborhood with name (" \
         << aNbrhdName << ") was found" << endl;
  }
  return nbrhdIdx;
}

mcPRL::Neighborhood* mcPRL::DataManager::
getNbrhdByName(const char *aNbrhdName,
               bool warning) {
  mcPRL::Neighborhood* pNbrhd = NULL;
  list<mcPRL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
  while(itrNbrhd != _lNbrhds.end()) {
    if(strcmp(aNbrhdName, itrNbrhd->name()) == 0) {
      pNbrhd = &(*itrNbrhd);
      break;
    }
    itrNbrhd++;
  }
  if(pNbrhd == NULL && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: NO Neighborhood with name (" \
         << aNbrhdName << ") was found" << endl;
  }
  return pNbrhd;
}

const mcPRL::Neighborhood* mcPRL::DataManager::
getNbrhdByName(const char *aNbrhdName,
               bool warning) const {
  const mcPRL::Neighborhood* pNbrhd = NULL;
  list<mcPRL::Neighborhood>::const_iterator itrNbrhd = _lNbrhds.begin();
  while(itrNbrhd != _lNbrhds.end()) {
    if(strcmp(aNbrhdName, itrNbrhd->name()) == 0) {
      pNbrhd = &(*itrNbrhd);
      break;
    }
    itrNbrhd++;
  }
  if(pNbrhd == NULL && warning) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: NO Neighborhood with name (" \
        << aNbrhdName << ") was found" << endl;
  }
  return pNbrhd;
}

bool mcPRL::DataManager::
dcmpLayer(int lyrID,
          int nbrhdID,
          int nRowSubspcs,
          int nColSubspcs) {
  mcPRL::Layer *pLyr = getLayerByIdx(lyrID, true);
  if(pLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to apply decomposition on a NULL Layer" \
         << endl;
    return false;
  }

  mcPRL::Neighborhood *pNbrhd = nbrhdID < 0 ? NULL : getNbrhdByIdx(nbrhdID, true);

  return pLyr->decompose(nRowSubspcs, nColSubspcs, pNbrhd);
}

bool mcPRL::DataManager::
dcmpLayer(const char *aLyrName,
          const char *aNbrhdName,
          int nRowSubspcs,
          int nColSubspcs) {
  mcPRL::Layer *pLyr = getLayerByName(aLyrName, true);
  if(pLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to apply decomposition on a NULL Layer" \
         << endl;
    return false;
  }

  mcPRL::Neighborhood *pNbrhd = aNbrhdName == NULL ? NULL : getNbrhdByName(aNbrhdName, true);

  return pLyr->decompose(nRowSubspcs, nColSubspcs, pNbrhd);
}

bool mcPRL::DataManager::
propagateDcmp(int fromLyrID,
              int toLyrID) {
  mcPRL::Layer *pFromLyr = getLayerByIdx(fromLyrID, true);
  if(pFromLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to copy decomposition from a NULL Layer" \
         << endl;
    return false;
  }

  mcPRL::Layer *pToLyr = getLayerByIdx(toLyrID, true);
  if(pToLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to copy decomposition to a NULL Layer" \
        << endl;
    return false;
  }

  if(pFromLyr == pToLyr) {
    return true;
  }

  return pToLyr->copyDcmp(pFromLyr);
}

bool mcPRL::DataManager::
propagateDcmp(int fromLyrID,
              const IntVect &vToLyrIDs) {
  for(int iToLyrIdx = 0; iToLyrIdx < vToLyrIDs.size(); iToLyrIdx++) {
    int toLyrID = vToLyrIDs[iToLyrIdx];
    if(!propagateDcmp(fromLyrID, toLyrID)) {
      return false;
    }
  }
  return true;
}

bool mcPRL::DataManager::
propagateDcmp(const string &fromLyrName,
              const string &toLyrName) {
  mcPRL::Layer *pFromLyr = getLayerByName(fromLyrName.c_str(), true);
  if(pFromLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to copy decomposition from a NULL Layer" \
         << endl;
    return false;
  }

  mcPRL::Layer *pToLyr = getLayerByName(toLyrName.c_str(), true);
  if(pToLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to copy decomposition from a NULL Layer" \
        << endl;
    return false;
  }

  if(pFromLyr == pToLyr) {
    return true;
  }

  return pToLyr->copyDcmp(pFromLyr);
}

bool mcPRL::DataManager::
propagateDcmp(const string &fromLyrName,
              const vector<string> &vToLyrNames) {
  for(int iToLyrName = 0; iToLyrName < vToLyrNames.size(); iToLyrName++) {
    const string &toLyrName = vToLyrNames[iToLyrName];
    if(!propagateDcmp(fromLyrName, toLyrName)) {
      return false;
    }
  }
  return true;
}

bool mcPRL::DataManager::
dcmpLayers(const IntVect &vLyrIDs,
           int nbrhdID,
           int nRowSubspcs,
           int nColSubspcs) {
  if(!dcmpLayer(vLyrIDs[0], nbrhdID,
                nRowSubspcs, nColSubspcs)) {
    return false;
  }
  if(!propagateDcmp(vLyrIDs[0], vLyrIDs)) {
    return false;
  }
  return true;
}

bool mcPRL::DataManager::
dcmpLayers(const vector<string> &vLyrNames,
           const char *aNbrhdName,
           int nRowSubspcs,
           int nColSubspcs) {
  if(!dcmpLayer(vLyrNames[0].c_str(), aNbrhdName,
                nRowSubspcs, nColSubspcs)) {
    return false;
  }
  if(!propagateDcmp(vLyrNames[0], vLyrNames)) {
    return false;
  }
  return true;
}

bool mcPRL::DataManager::
dcmpLayers(mcPRL::Transition &trans,
           int nRowSubspcs,
           int nColSubspcs) {
  const char *aNbrhdName = trans.getNbrhdName().empty() ? NULL : trans.getNbrhdName().c_str();
  if(!dcmpLayer(trans.getPrimeLyrName().c_str(),
                aNbrhdName,
                nRowSubspcs, nColSubspcs)) {
    return false;
  }

  if(!propagateDcmp(trans.getPrimeLyrName(), trans.getInLyrNames()) ||
     !propagateDcmp(trans.getPrimeLyrName(), trans.getOutLyrNames())) {
    return false;
  }
  return true;
}

bool mcPRL::DataManager::
dcmpAllLayers(const char *aNbrhdName,
              int nRowSubspcs,
              int nColSubspcs) {
  list<mcPRL::Layer>::iterator iLyr = _lLayers.begin();
  mcPRL::Layer *p1stLyr = &(*iLyr);
  mcPRL::Neighborhood *pNbrhd = aNbrhdName == NULL ? NULL : getNbrhdByName(aNbrhdName, true);
  if(!p1stLyr->decompose(nRowSubspcs, nColSubspcs, pNbrhd)) {
    return false;
  }
  iLyr++;

  while(iLyr != _lLayers.end()) {
    if(!(*iLyr).copyDcmp(p1stLyr)) {
      return false;
    }
    iLyr++;
  }

  return true;
}

const mcPRL::OwnershipMap& mcPRL::DataManager::
ownershipMap() const {
  return _mOwnerships;
}

bool mcPRL::DataManager::
syncOwnershipMap() {
  vector<char> buf;
  int bufSize = 0;
  if(_prc.isMaster()) {
    if(!_mOwnerships.map2buf(buf)) {
      _prc.abort();
      return false;
    }
    bufSize = buf.size();
  }

  MPI_Bcast(&bufSize, 1, MPI_INT, 0, _prc.comm());

  if(!_prc.isMaster()) {
    buf.resize(bufSize);
  }

  MPI_Bcast(&(buf[0]), bufSize, MPI_CHAR, 0, _prc.comm());

  if(!_prc.isMaster()) {
    if(!_mOwnerships.buf2map(buf)) {
      _prc.abort();
      return false;
    }
  }

  return true;
}

bool mcPRL::DataManager::
initTaskFarm(mcPRL::Transition &trans,
             mcPRL::MappingMethod mapMethod,
             int nSubspcs2Map,
             mcPRL::ReadingOption readOpt) {
  if(_prc.nProcesses() < 2) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to initialize task farming when there are " \
         << _prc.nProcesses() << " Processes" \
         << endl;
    _prc.abort();
    return false;
  }
  else if(_prc.nProcesses() <= 2 && _prc.hasWriter()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize task farming when there are " \
        << _prc.nProcesses() << " Processes that include a writer Process" \
        << endl;
    _prc.abort();
    return false;
  }

  mcPRL::Layer *pPrimeLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrimeLyr == NULL) {
    _prc.abort();
    return false;
  }
  if(!pPrimeLyr->isDecomposed()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: the primary Layer (" << trans.getPrimeLyrName() \
         << ") is NOT decomposed for task farming" \
         << endl;
    _prc.abort();
    return false;
  }

  mcPRL::IntVect vSubspcIDs = pPrimeLyr->allSubCellspaceIDs();
  mcPRL::IntVect vPrcIDs = _prc.allPrcIDs(false, false); // exclude the master and writer processes
  _mOwnerships.init(vSubspcIDs, vPrcIDs);
  _mOwnerships.mapping(mapMethod, nSubspcs2Map); // initial mapping

  _clearTransfSpaceReqs();

  if(!_initReadInData(trans, readOpt)) {
    _prc.abort();
    return false;
  }

  return true;
}

bool mcPRL::DataManager::
initStaticTask(mcPRL::Transition &trans,
               mcPRL::MappingMethod mapMethod,
               mcPRL::ReadingOption readOpt) {
  if(_prc.nProcesses() <= 1 && _prc.hasWriter()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize static tasking when there are " \
        << _prc.nProcesses() << " Processes that include a writer Process" \
        << endl;
    _prc.abort();
    return false;
  }

  mcPRL::Layer *pPrimeLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrimeLyr == NULL) {
    _prc.abort();
    return false;
  }

  mcPRL::IntVect vSubspcIDs = pPrimeLyr->allSubCellspaceIDs();
  mcPRL::IntVect vPrcIDs = _prc.allPrcIDs(true, false); // exclude the writer processes
  _mOwnerships.init(vSubspcIDs, vPrcIDs);
  _mOwnerships.mapping(mapMethod); // map all SubCellspaces

  _clearTransfSpaceReqs();

  if(!_initReadInData(trans, readOpt)) {
    _prc.abort();
    return false;
  }

  // Complete all pending transfers, only for centralized I/O
  finishReadingLayers(readOpt);

  return true;
}


bool mcPRL::DataManager::
mergeTmpSubspcByGDAL(mcPRL::Transition &trans,
                     int subspcGlbID) {
  if((_prc.hasWriter() && _prc.isWriter()) ||
     (!_prc.hasWriter() && _prc.isMaster())) {
    const vector<string> &vOutLyrs = trans.getOutLyrNames();
    for(int iOutLyr = 0; iOutLyr < vOutLyrs.size(); iOutLyr++) {
      mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrs[iOutLyr].c_str(), true);
      if(pOutLyr == NULL) {
        _prc.abort();
        return false;
      }
      if(!pOutLyr->mergeSubCellspaceByGDAL(subspcGlbID, _prc.nProcesses())) {
        _prc.abort();
        return false;
      }
      if(!_mOwnerships.findSubspcIDOnPrc(subspcGlbID, _prc.id())) {
        pOutLyr->delSubCellspace_glbID(subspcGlbID);
      }
    }
  }
  return true;
}

bool mcPRL::DataManager::
mergeAllTmpSubspcsByGDAL(mcPRL::Transition &trans) {
  if((_prc.hasWriter() && _prc.isWriter()) ||
     (!_prc.hasWriter() && _prc.isMaster())) {
    const vector<string> &vOutLyrs = trans.getOutLyrNames();
    for(int iOutLyr = 0; iOutLyr < vOutLyrs.size(); iOutLyr++) {
      mcPRL::Layer *pOutLyr = getLayerByName(vOutLyrs[iOutLyr].c_str(), true);
      if(pOutLyr == NULL) {
        _prc.abort();
        return false;
      }
      list<mcPRL::SubCellspaceInfo>::const_iterator itrSubInfo = pOutLyr->allSubCellspaceInfos()->begin();
      while(itrSubInfo != pOutLyr->allSubCellspaceInfos()->end()) {
        int glbID = itrSubInfo->id();
        if(!pOutLyr->mergeSubCellspaceByGDAL(glbID, _prc.nProcesses())) {
          return false;
        }
        if(!_mOwnerships.findSubspcIDOnPrc(glbID, _prc.id())) {
          pOutLyr->delSubCellspace_glbID(glbID);
        }
        itrSubInfo++;
      }
    } // end of iOutLyr loop
  }
  return true;
}
