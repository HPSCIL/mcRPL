#include "mcrpl-dataManager.h"

/****************************************************
*                 Protected Methods                 *
****************************************************/
bool mcRPL::DataManager::
_initReadInData(mcRPL::Transition &trans,
                mcRPL::ReadingOption readOpt) {
  const vector<string> &vInLyrNames = trans.getInLyrNames();
  for(int iInLyr = 0; iInLyr < vInLyrNames.size(); iInLyr++) {
    const string &inLyrName = vInLyrNames.at(iInLyr);
    if(!beginReadingLayer(inLyrName.c_str(), readOpt)) {
      return false;
    }
  } // end -- for(iInLyr) loop

  return true;
}
mcRPL::EvaluateReturn mcRPL::DataManager::
evaluate_ST(mcRPL::EvaluateType evalType,
            mcRPL::Transition &trans,
            mcRPL::WritingOption writeOpt,
			bool isGPUCompute,
			mcRPL::pCuf pf,
            bool ifInitNoData,
            const mcRPL::LongVect *pvGlbIdxs) {
				if(evalType!=EVAL_ALL&&isGPUCompute==true)
	{
		 cerr << __FILE__ << " function:" << __FUNCTION__ \
			  << " Error: when GPU is to be used," \
			  << " only EVAL_ALL  be used" \
			   << endl;
    _prc.abort();
    return mcRPL::EVAL_FAILED;
	}
  if(!trans.onlyUpdtCtrCell() &&
     trans.needExchange() &&
     writeOpt != mcRPL::NO_WRITING &&
     (_prc.hasWriter() ||
      (writeOpt != mcRPL::CENT_WRITING || writeOpt != mcRPL::CENTDEL_WRITING))) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: when a non-centralized neighborhood-scope Transition is to be used," \
        << " and data exchange and writing are required, " \
        << " only the centralized writing without a writer can be used" \
        << endl;
    _prc.abort();
    return mcRPL::EVAL_FAILED;
  }
  int nSMcount=_prc.getDeive().getDeviceinfo()->smCount();
  trans.setSMcount(nSMcount);
  if(trans.needExchange()) {
    if(!_calcExchangeBRs(trans) ||
       !_calcExchangeRoutes(trans)) {
      _prc.abort();
      return mcRPL::EVAL_FAILED;
    }
	
    /*
      for(int prcID = 0; prcID < _prc.nProcesses(); prcID++) {
        if(_prc.id() == prcID) {
          cout << "Send Route on process " << prcID << endl;
          mcRPL::ExchangeMap::iterator itrSendRoute = _mSendRoutes.begin();
          while(itrSendRoute != _mSendRoutes.end()) {
            cout << itrSendRoute->first << ": ";
            vector<mcRPL::ExchangeNode> &vSendNodes = itrSendRoute->second;
            for(int iSend = 0; iSend < vSendNodes.size(); iSend++) {
              mcRPL::ExchangeNode &sendNode = vSendNodes[iSend];
              cout << "(" << sendNode.subspcGlbID << ", " \
                  << sendNode.iDirection << ", "
                  << sendNode.iNeighbor << ")";
            }
            cout << endl;
            itrSendRoute++;
          }


          cout << "Recv Route on process " << prcID << endl;
          mcRPL::ExchangeMap::iterator itrRecvRoute = _mRecvRoutes.begin();
          while(itrRecvRoute != _mRecvRoutes.end()) {
            cout << itrRecvRoute->first << ": ";
            vector<mcRPL::ExchangeNode> &vRecvNodes = itrRecvRoute->second;
            for(int iRecv = 0; iRecv < vRecvNodes.size(); iRecv++) {
              mcRPL::ExchangeNode &recvNode = vRecvNodes[iRecv];
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
  mcRPL::EvaluateReturn done;
  if(_prc.isWriter()) { // writer process
    done = _writerST(trans, writeOpt);
  }
  else { // worker processes
	  done = _workerST(evalType, trans, writeOpt, isGPUCompute,pf,ifInitNoData, pvGlbIdxs);
    if(_prc.isMaster() && done != mcRPL::EVAL_FAILED) { // master process
      done = _masterST(trans, writeOpt);
    }
  }
  return done;
}
mcRPL::EvaluateReturn mcRPL::DataManager::
evaluate_TF(mcRPL::EvaluateType evalType,
            mcRPL::Transition &trans,
            mcRPL::ReadingOption readOpt,
            mcRPL::WritingOption writeOpt,
			bool  isGPUCompute,
			mcRPL::pCuf pf,
            bool ifInitNoData,
            bool ifSyncOwnershipMap,
            const mcRPL::LongVect *pvGlbIdxs) {
				if(evalType!=EVAL_ALL&&isGPUCompute==true)
	{
		 cerr << __FILE__ << " function:" << __FUNCTION__ \
			  << " Error: when GPU is to be used," \
			  << " only EVAL_ALL  be used" \
			   << endl;
    _prc.abort();
    return mcRPL::EVAL_FAILED;
	}
  if(trans.needExchange()) {
    if(!trans.onlyUpdtCtrCell() &&
        writeOpt != mcRPL::NO_WRITING) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: when a non-centralized neighborhood-scope Transition is to be used," \
          << " and data exchange and writing are required, "
          << " only the static tasking can be used for load-balancing" \
          << endl;
      _prc.abort();
      return mcRPL::EVAL_FAILED;
    }
    if(writeOpt == mcRPL::CENTDEL_WRITING ||
       writeOpt == mcRPL::PARADEL_WRITING) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: when data exchange is needed,"\
           << " deleting SubCellspaces during the writing is NOT allowed" \
           << endl;
      _prc.abort();
      return mcRPL::EVAL_FAILED;
    }
  }

  if(trans.needExchange()) {
    if(!_calcExchangeBRs(trans)) {
      _prc.abort();
      return mcRPL::EVAL_FAILED;
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

  mcRPL::EvaluateReturn done;
  if(_prc.isMaster()) { // master process
    done = _masterTF(trans, readOpt, writeOpt);
  }
  else if(_prc.isWriter()) { // writer process
    done = _writerTF(trans, writeOpt);
  }
  else { // worker processes
    done = _workerTF(evalType, trans, readOpt, writeOpt,  isGPUCompute,pf,ifInitNoData, pvGlbIdxs);
  }

  if(done != mcRPL::EVAL_FAILED) {
    if(trans.needExchange() || ifSyncOwnershipMap) {
      if(!syncOwnershipMap()) {
        done = mcRPL::EVAL_FAILED;
      }
      else if(trans.needExchange()) {
        if(!_calcExchangeRoutes(trans)) {
          done = mcRPL::EVAL_FAILED;
        }
        else if(!_makeCellStream(trans) ||
                !_iExchangeBegin() ||
                !_iExchangeEnd()) {
          done = mcRPL::EVAL_FAILED;
        }
      }
    }
  }

  return done;
}
mcRPL::EvaluateReturn mcRPL::DataManager::
_workerTF(mcRPL::EvaluateType evalType,
          mcRPL::Transition &trans,
          mcRPL::ReadingOption readOpt,
          mcRPL::WritingOption writeOpt,
		  bool  isGPUCompute,
		  mcRPL::pCuf pf,
          bool ifInitNoData,
          const mcRPL::LongVect *pvGlbIdxs) {
  if(evalType != mcRPL::EVAL_NONE &&
     evalType != mcRPL::EVAL_ALL &&
     evalType != mcRPL::EVAL_RANDOMLY &&
     evalType != mcRPL::EVAL_SELECTED) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid evaluation type (" \
         << evalType << ")" \
         << endl;
    _prc.abort();
    return mcRPL::EVAL_FAILED;
  }

  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  mcRPL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  int resp = 0;
  mcRPL::TaskSignal req = mcRPL::NEWTASK_SGNL;
  MPI_Status status;
  int mappedSubspcID = mcRPL::ERROR_ID;
  int submit2ID = _prc.hasWriter() ? 1 : 0; // if a writer exists, send to writer; otherwise master
  int readySubspcID = mcRPL::ERROR_ID;
  bool delSentSub = (writeOpt == mcRPL::CENTDEL_WRITING) ? true : false;

  while((mcRPL::TaskSignal)resp != mcRPL::QUIT_SGNL) { // keep running until receiving a Quitting instruction from master
    // if there is less than or equal to one SubCellspace to be evaluated in the queue
    if(_mOwnerships.subspcIDsOnPrc(_prc.id())->size() - _vEvaledSubspcIDs.size() <= 1) {
      req = mcRPL::NEWTASK_SGNL;
      MPI_Send(&req, 1, MPI_INT, 0, mcRPL::REQUEST_TAG, _prc.comm());
      MPI_Recv(&resp, 1, MPI_INT, 0, mcRPL::DSTRBT_TAG, _prc.comm(), &status);
      if((mcRPL::TaskSignal)resp != mcRPL::QUIT_SGNL) { // received SubCellspace ID from master
        mappedSubspcID = resp;
        _mOwnerships.mappingTo(mappedSubspcID, _prc.id());
        if(!_workerReadSubspcs(trans, mappedSubspcID, readOpt)) {
          _prc.abort();
          return mcRPL::EVAL_FAILED;
        }
      } // end -- if((mcRPL::TaskSignal)resp != mcRPL::QUIT_SGNL)
    } // end -- if(_mOwnerships.subspcIDsOnPrc(_prc.id())->size() - _vEvaledSubspcIDs.size() <= 1)

    // test pending transfers, only for centralized I/O
    if(!_vBcstSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltBcstIDs, mcRPL::BCST_TRANSF, delSentSub, false, false);
    }
    if(!_vSendSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSub, false, false);
    }
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSub, false, false);
    }

    // PROCESS A SUBSPACES THAT IS READY
    readySubspcID = _checkTransInDataReady(trans, readOpt);
    if(readySubspcID != mcRPL::ERROR_ID) {
      // when task farming, always evaluate the working rectangle.
      // the edge-first method won't be efficient, 'cuz all SubCellspaces may not
      // have been mapped yet.
      //cout << _prc.id() << ": processing SubCellspace " << readySubspcID << endl;
      switch(evalType) {
        case EVAL_NONE:
          if(_evalNone(trans, readySubspcID, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        case EVAL_ALL:
          if(_evalAll(trans, isGPUCompute,pf,readySubspcID, mcRPL::EVAL_WORKBR, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        case EVAL_RANDOMLY:
          if(_evalRandomly(trans, readySubspcID, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        case EVAL_SELECTED:
          if(pvGlbIdxs == NULL) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                 << " Error: NULL vector of cell indices to evaluate" \
                 << endl;
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          if(_evalSelected(trans, *pvGlbIdxs, readySubspcID, mcRPL::EVAL_WORKBR, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        default:
          break;
      }
      _vEvaledSubspcIDs.push_back(readySubspcID);

      if(!_workerWriteSubspcs(trans, readySubspcID, writeOpt)) {
        _prc.abort();
        return mcRPL::EVAL_FAILED;
      }
    } // end -- if(readySubspcID != ERROR_ID)

  } // end -- while((TaskSignal)resp != QUIT_SGNL)


  // COMPLETE ALL ASSIGNED SUBCELLSPACES
  const mcRPL::IntVect *pSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
  while(pSubspcIDs != NULL &&
      pSubspcIDs->size() - _vEvaledSubspcIDs.size() > 0) {
    // test pending transfers, only for centralized I/O
    if(!_vBcstSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltBcstIDs, mcRPL::BCST_TRANSF, delSentSub, false, false);
    }
    if(!_vSendSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSub, false, false);
    }
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSub, false, false);
    }

    // PROCESS A SUBSPACES THAT IS READY
    readySubspcID = _checkTransInDataReady(trans, readOpt);
    if(readySubspcID != mcRPL::ERROR_ID) {
      //cout << _prc.id() << ": processing SubCellspace " << readySubspcID << endl;
      switch(evalType) {
        case EVAL_NONE:
          if(_evalNone(trans, readySubspcID, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        case EVAL_ALL:
          if(_evalAll(trans, isGPUCompute,pf, readySubspcID, mcRPL::EVAL_WORKBR, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        case EVAL_RANDOMLY:
          if(_evalRandomly(trans, readySubspcID, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        case EVAL_SELECTED:
          if(pvGlbIdxs == NULL) {
            cerr << __FILE__ << " function:" << __FUNCTION__ \
                 << " Error: NULL vector of cell indices to evaluate" \
                 << endl;
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          if(_evalSelected(trans, *pvGlbIdxs, readySubspcID, mcRPL::EVAL_WORKBR, ifInitNoData) == mcRPL::EVAL_FAILED) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
          break;
        default:
          break;
      }
      _vEvaledSubspcIDs.push_back(readySubspcID);

      if(!_workerWriteSubspcs(trans, readySubspcID, writeOpt)) {
        _prc.abort();
        return mcRPL::EVAL_FAILED;
      }
    } // end -- if(readySubspcID != ERROR_ID)
  }  // end -- while(_mOwnerships.subspcIDsOnPrc(_prc.id())->size() - _vEvaledSubspcIDs.size() > 0)

  // complete all pending transfers, only for centralized I/O
  while(!_vBcstSpcReqs.empty() && nCmpltBcsts != MPI_UNDEFINED) {
    nCmpltBcsts = _iTransfSpaceTest(vCmpltBcstIDs, mcRPL::BCST_TRANSF, delSentSub, false, false);
  }
  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSub, false, false);
  }
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSub, false, false);
  }

  req = mcRPL::QUIT_SGNL;
  MPI_Send(&req, 1, MPI_INT, 0, mcRPL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to master
  if(_prc.hasWriter()) {
    MPI_Send(&req, 1, MPI_INT, 1, mcRPL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to writer
  }

  _clearTransfSpaceReqs();

  return mcRPL::EVAL_SUCCEEDED;
}
mcRPL::EvaluateReturn mcRPL::DataManager::
_workerST(mcRPL::EvaluateType evalType,
          mcRPL::Transition &trans,
          mcRPL::WritingOption writeOpt,
		  bool isGPUCompute,
		  mcRPL::pCuf pf,
          bool ifInitNoData,
          const mcRPL::LongVect *pvGlbIdxs) {
  if(evalType != mcRPL::EVAL_NONE &&
     evalType != mcRPL::EVAL_ALL &&
     evalType != mcRPL::EVAL_RANDOMLY &&
     evalType != mcRPL::EVAL_SELECTED) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: invalid evaluation type (" \
         << evalType << ")" \
         << endl;
    _prc.abort();
    return mcRPL::EVAL_FAILED;
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
            if(_evalNone(trans, subspcGlbID, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          case EVAL_ALL:
			  if(_evalAll(trans,isGPUCompute, pf,subspcGlbID, mcRPL::EVAL_WORKBR, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          case EVAL_RANDOMLY:
            if(_evalRandomly(trans, subspcGlbID, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          case EVAL_SELECTED:
            if(pvGlbIdxs == NULL) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                   << " Error: NULL vector of cell indices to evaluate" \
                   << endl;
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            if(_evalSelected(trans, *pvGlbIdxs, subspcGlbID, mcRPL::EVAL_WORKBR, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          default:
            break;
        }
        _vEvaledSubspcIDs.push_back(subspcGlbID);

        if(!_dymWriteSubspcs(trans, subspcGlbID, writeOpt)) {
          _prc.abort();
          return mcRPL::EVAL_FAILED;
        }
		trans.clearGPUMem();
      } // end -- if(!trans.needExchange())

      else { // exchange needed, process the edges first
        //cout << _prc.id() << ": processing SubCellspace edge " << subspcGlbID << endl;
        switch(evalType) {
          case EVAL_NONE:
            if(_evalNone(trans, subspcGlbID, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          case EVAL_ALL:
            if(_evalAll(trans, isGPUCompute,pf, subspcGlbID, mcRPL::EVAL_EDGES, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          case EVAL_RANDOMLY:
            if(_evalRandomly(trans, subspcGlbID, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            break;
          case EVAL_SELECTED:
            if(pvGlbIdxs == NULL) {
              cerr << __FILE__ << " function:" << __FUNCTION__ \
                  << " Error: NULL vector of cell indices to evaluate" \
                  << endl;
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            if(_evalSelected(trans, *pvGlbIdxs, subspcGlbID, mcRPL::EVAL_EDGES, ifInitNoData) == mcRPL::EVAL_FAILED) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
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
        return mcRPL::EVAL_FAILED;
      }

      if(trans.edgesFirst()) {
        for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
          int subspcGlbID = pvSubspcIDs->at(iSubspc);
          //cout << _prc.id() << ": processing SubCellspace interior " << subspcGlbID << endl;
          switch(evalType) {
            case EVAL_NONE:
              break;
            case EVAL_ALL:
              if(_evalAll(trans, isGPUCompute,pf, subspcGlbID, mcRPL::EVAL_INTERIOR, ifInitNoData) == mcRPL::EVAL_FAILED) {
                _prc.abort();
                return mcRPL::EVAL_FAILED;
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
                return mcRPL::EVAL_FAILED;
              }
              if(_evalSelected(trans, *pvGlbIdxs, subspcGlbID, mcRPL::EVAL_INTERIOR, ifInitNoData) == mcRPL::EVAL_FAILED) {
                _prc.abort();
                return mcRPL::EVAL_FAILED;
              }
              break;
            default:
              break;
          }
          _vEvaledSubspcIDs.push_back(subspcGlbID);
		   trans.clearGPUMem();
          if(!_dymWriteSubspcs(trans, subspcGlbID, writeOpt)) {
            _prc.abort();
            return mcRPL::EVAL_FAILED;
          }
        } // end -- for(iSubspc)

        if(!_iExchangeEnd()) {
          return mcRPL::EVAL_FAILED;
        }
      }
    } // end -- if(trans.needExchange())

    if(!_finalWriteSubspcs(trans, writeOpt)) {
      _prc.abort();
      return mcRPL::EVAL_FAILED;
    }

  } // end -- if(pvSubspcIDs != NULL)

  // complete all pending transfers, only for centralized I/O
  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  mcRPL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  bool delSentSub = (writeOpt == mcRPL::CENTDEL_WRITING) ? true : false;

  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSub, false, false);
  }

  // send Quitting signal
  mcRPL::TaskSignal req = mcRPL::QUIT_SGNL;
  if(_prc.hasWriter()) {
    MPI_Send(&req, 1, MPI_INT, 1, mcRPL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to writer
  }

  if(!_prc.isMaster()) {
    if(!_prc.hasWriter() && writeOpt != mcRPL::NO_WRITING) {
      MPI_Send(&req, 1, MPI_INT, 0, mcRPL::REQUEST_TAG, _prc.comm()); // send a Quitting signal to master
      //cout << _prc.id() << " quitting" << endl;
    }
    _clearTransfSpaceReqs();
  }

  return mcRPL::EVAL_SUCCEEDED;
}
mcRPL::EvaluateReturn mcRPL::DataManager::
_evalAll(mcRPL::Transition &trans,bool isGPUCompute,
           pCuf pf,
         int subspcGlbID,
         mcRPL::EvaluateBR br2Eval,
         bool ifInitNoData) {
  if(br2Eval != mcRPL::EVAL_WORKBR) {
    if(subspcGlbID == mcRPL::ERROR_ID) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: cannot evaluate the Edges or Interior of a whole Cellspace" \
           << endl;
      return mcRPL::EVAL_FAILED;
    }
    if(trans.getOutLyrNames().empty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: no output Layers to evaluate Edges or Interior" \
           << endl;
      return mcRPL::EVAL_FAILED;
    }
    if(!trans.isOutLyr(trans.getPrimeLyrName())) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the primary Layer (" << trans.getPrimeLyrName() \
           << ") is NOT an output Layer" << endl;
      return mcRPL::EVAL_FAILED;
    }
  }

  if(!_initEvaluate(trans, subspcGlbID, br2Eval, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize the Transition" \
        << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::Layer *pPrmLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrmLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to find the primary Layer" \
         << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::Cellspace *pPrmCellspace = NULL;
  if(subspcGlbID == mcRPL::ERROR_ID) { // if evaluate the whole Cellspace
    pPrmCellspace = pPrmLyr->cellspace();
  }
  else { // if evaluate a SubCellspace
    pPrmCellspace = pPrmLyr->subCellspace_glbID(subspcGlbID, true);
  }
  if(pPrmCellspace == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to find the primary Cellspace" \
         << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::CoordBR workBR;
  const mcRPL::CoordBR *pWorkBR;
  mcRPL::EvaluateReturn done = mcRPL::EVAL_SUCCEEDED;
  switch(br2Eval) {
    case mcRPL::EVAL_WORKBR:
      if(subspcGlbID == mcRPL::ERROR_ID) { // if evaluate the whole Cellspace
        if(!pPrmCellspace->info()->calcWorkBR(&workBR, trans.getNbrhd(), true)) {
          return mcRPL::EVAL_FAILED;
        }
      }
      else { // if evaluate a SubCellspace
        workBR = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->workBR();
      }
      done = trans.evalBR(workBR,isGPUCompute,pf);
      break;
    case mcRPL::EVAL_EDGES:
      for(int iEdge = 0; iEdge < ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->nEdges(); iEdge++) {
        pWorkBR = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->edgeBR(iEdge);
        if(pWorkBR != NULL) {
          done = trans.evalBR(*pWorkBR,isGPUCompute, pf);
          if(done == mcRPL::EVAL_FAILED) {
            break;
          }
        }
      }
      break;
    case mcRPL::EVAL_INTERIOR:
      pWorkBR = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->interiorBR();
      if(pWorkBR != NULL) {
        done = trans.evalBR(*pWorkBR, isGPUCompute,pf);
      }
      break;
    default:
      break;
  }

  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcRPL::EVAL_FAILED;
  }

  return done;
}
bool mcRPL::DataManager::
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

bool mcRPL::DataManager::
_iBcastCellspaceBegin(const string &lyrName) {
  mcRPL::Layer *pLyr = getLayerByName(lyrName.c_str(), true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  }

  int msgTag;
  if(!_toMsgTag(msgTag, getLayerIdxByName(lyrName.c_str()),
                mcRPL::ERROR_ID, mcRPL::ERROR_ID)) {
    _prc.abort();
    return false;
  }

  if(_prc.isMaster()) { // Master process
    mcRPL::Cellspace *pSpc = pLyr->cellspace();
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
      _vBcstSpcInfos.push_back(mcRPL::TransferInfo(_prc.id(), toPrcID,
                                                  lyrName, mcRPL::ERROR_ID,
                                                  mcRPL::SPACE_TRANSFDATA));
    }
  } // end -- if(_prc.isMaster())

  else if(!_prc.isWriter()) { // Worker process
    mcRPL::Cellspace *pSpc = pLyr->cellspace();
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
    _vBcstSpcInfos.push_back(mcRPL::TransferInfo(0, _prc.id(),
                                                lyrName, mcRPL::ERROR_ID,
                                                mcRPL::SPACE_TRANSFDATA));
  } // end -- worker process

  return true;
}

bool mcRPL::DataManager::
_iTransfSubspaceBegin(int fromPrcID,
                      int toPrcID,
                      const string &lyrName,
                      int subspcGlbID) {
  mcRPL::Layer *pLyr = getLayerByName(lyrName.c_str(), true);
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
    mcRPL::Cellspace *pSpc = pLyr->subCellspace_glbID(subspcGlbID, true);
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
    _vSendSpcInfos.push_back(mcRPL::TransferInfo(_prc.id(), toPrcID,
                             lyrName, subspcGlbID,
                             mcRPL::SPACE_TRANSFDATA));
  } // end -- if(_prc.id() == fromPrcID)

  else if(_prc.id() == toPrcID) { // TO process
    mcRPL::Cellspace *pSpc = pLyr->subCellspace_glbID(subspcGlbID);
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
    _vRecvSpcInfos.push_back(mcRPL::TransferInfo(fromPrcID, _prc.id(),
                                                lyrName, subspcGlbID,
                                                mcRPL::SPACE_TRANSFDATA));
  } // end -- if(_prc.id() == toPrcID)

  return true;
}

int mcRPL::DataManager::
_iTransfSpaceTest(mcRPL::IntVect &vCmpltIDs,
                  mcRPL::TransferType transfType,
                  bool delSentSpaces,
                  bool writeRecvSpaces,
                  bool delRecvSpaces) {
  int nCmplts = 0;

  vector<MPI_Request> *pvTransfReqs = NULL;
  mcRPL::TransferInfoVect *pvTransfInfos = NULL;
  switch(transfType) {
    case mcRPL::BCST_TRANSF:
      pvTransfReqs = &_vBcstSpcReqs;
      pvTransfInfos = &_vBcstSpcInfos;
      break;
    case mcRPL::SEND_TRANSF:
      pvTransfReqs = &_vSendSpcReqs;
      pvTransfInfos = &_vSendSpcInfos;
      break;
    case mcRPL::RECV_TRANSF:
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
    mcRPL::TransferInfo &transfInfo = pvTransfInfos->at(vCmpltIDs[iCmplt]);
    transfInfo.complete();

    int fromPrcID = transfInfo.fromPrcID();
    int toPrcID = transfInfo.toPrcID();
    int subspcGlbID = transfInfo.subspcGlbID();

    const string &lyrName = transfInfo.lyrName();
    mcRPL::Layer *pLyr = getLayerByName(lyrName.c_str(), true);
    if(pLyr == NULL) {
      continue;
    }

    if(delSentSpaces &&
       _prc.id() == fromPrcID) {
      if(transfType == mcRPL::SEND_TRANSF) {
        if(!pLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
      else if(transfType == mcRPL::BCST_TRANSF) {
        if(pvTransfInfos->checkBcastCellspace(_prc.id(), lyrName)) {
          pLyr->delCellspace();
        }
      }
    } // end -- if(delSentSpaces)

    if(writeRecvSpaces &&
       _prc.id() == toPrcID &&
       transfType == mcRPL::RECV_TRANSF) {
      //cout << _prc.id() << ": writing SubCellspace " << subspcGlbID << endl;
      if(!pLyr->writeSubCellspaceByGDAL(subspcGlbID)) {
        cerr << __FILE__ << " function:" << __FUNCTION__
             << " Error: failed to write SubCellspace (" \
             << subspcGlbID << ")" << endl;
      }
    } // end -- if(writeRecvSpaces)

    if(delRecvSpaces &&
       _prc.id() == toPrcID &&
       transfType == mcRPL::RECV_TRANSF) {
      if(!pLyr->delSubCellspace_glbID(subspcGlbID)) {
        cerr << __FILE__ << " function:" << __FUNCTION__
             << " Error: unable to find SubCellspace (" \
             << subspcGlbID << ") to delete" << endl;
      }
    } // end -- if(delRecvSpaces)

  } // end -- for(iCmplt) loop

  return nCmplts;
}

void mcRPL::DataManager::
_clearTransfSpaceReqs() {
  _vBcstSpcReqs.clear();
  _vBcstSpcInfos.clear();
  _vSendSpcReqs.clear();
  _vSendSpcInfos.clear();
  _vRecvSpcReqs.clear();
  _vRecvSpcInfos.clear();
  _vEvaledSubspcIDs.clear();
}

int mcRPL::DataManager::
_checkTransInDataReady(mcRPL::Transition &trans,
                       mcRPL::ReadingOption readOpt) {
  int readySubspcID = mcRPL::ERROR_ID, subspcID2Check;

  const mcRPL::IntVect *pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
  if(pvSubspcIDs == NULL) {
    return mcRPL::ERROR_ID;
  }

  for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
    subspcID2Check = pvSubspcIDs->at(iSubspc);
    if(std::find(_vEvaledSubspcIDs.begin(), _vEvaledSubspcIDs.end(), subspcID2Check) != _vEvaledSubspcIDs.end()) {
      continue; // this SubCellspace has been evaluated, ignore the following steps, check the next SubCellspace
    }
    else { // if SubCellspace has NOT been evaluated
      if(readOpt == mcRPL::PARA_READING) { // if parallel reading, input data has been read
        readySubspcID = subspcID2Check;
        break; // break the for(iSubspc) loop
      }
      else if(readOpt == mcRPL::CENT_READING ||
              readOpt == mcRPL::CENTDEL_READING) { // if centralized reading, check if transfers have completed
        const vector<string> &vInLyrNames = trans.getInLyrNames();
        for(int iInLyr = 0; iInLyr < vInLyrNames.size(); iInLyr++) {
          mcRPL::Layer *pInLyr = getLayerByName(vInLyrNames[iInLyr].c_str(), true);
          if(pInLyr == NULL) {
            return mcRPL::ERROR_ID;
          }
          if(pInLyr->isDecomposed()) {
            const mcRPL::TransferInfo *pTransfInfo = _vRecvSpcInfos.findInfo(vInLyrNames[iInLyr], subspcID2Check);
            if(pTransfInfo == NULL || pTransfInfo->completed() == false) {
              subspcID2Check = mcRPL::ERROR_ID;
              break; // the transfer of this SubCellspace has NOT been completed, break the for(iInLyr) loop
            }
          }
          else {
            const mcRPL::TransferInfo *pTransfInfo = _vBcstSpcInfos.findInfo(vInLyrNames[iInLyr], mcRPL::ERROR_ID);
            if(pTransfInfo == NULL || pTransfInfo->completed() == false) {
              subspcID2Check = mcRPL::ERROR_ID;
              break; // the transfer (broadcast) of this Cellspace has NOT been completed, break the for(iInLyr) loop
            }
          }
        } // end -- for(iInLyr) loop

        if(subspcID2Check != mcRPL::ERROR_ID) {
          readySubspcID = subspcID2Check;
          break; // found a SubCellspace whose transfer has been completed, break the for(iSubspc) loop
        }
      } // end -- if(readOpt == mcRPL::CENT_READING)
      else if(readOpt == mcRPL::PGT_READING) {
        readySubspcID = subspcID2Check;
        break; // break the for(iSubspc) loop
      }
    } // end -- if SubCellspace has NOT been evaluated
  } // end -- for(iSubspc) loop

  return readySubspcID;
}

bool mcRPL::DataManager::
_initTransOutData(mcRPL::Transition &trans,
                  int subspcGlbID,
                  bool ifInitNoData) {
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
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
      mcRPL::SubCellspace *pSubspc = pOutLyr->subCellspace_glbID(subspcGlbID, false);
      if(pSubspc == NULL) {
        pSubspc = pOutLyr->addSubCellspace(subspcGlbID, pNoDataVal);
      }
      if(pSubspc == NULL) {
        return false;
      }
    }
    else { // whole Cellspace on the Layer
      mcRPL::Cellspace *pSpc = pOutLyr->cellspace();
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

bool mcRPL::DataManager::
_setTransCellspaces(mcRPL::Transition &trans,
                    int subspcGlbID) {
  trans.clearCellspaces();

  const vector<string> &vInLyrNames = trans.getInLyrNames();
  for(size_t iLyr = 0; iLyr < vInLyrNames.size(); iLyr++) {
    mcRPL::Layer *pLyr = getLayerByName(vInLyrNames[iLyr].c_str(), true);
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
      mcRPL::SubCellspace *pSubSpc = pLyr->subCellspace_glbID(subspcGlbID, true);
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
    mcRPL::Layer *pLyr = getLayerByName(vOutLyrNames[iLyr].c_str(), true);
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

bool mcRPL::DataManager::
_setTransNbrhd(mcRPL::Transition &trans) {
  trans.setNbrhd();

  if(!trans.getNbrhdName().empty()) {
    mcRPL::Neighborhood *pNbrhd = getNbrhdByName(trans.getNbrhdName().c_str(), true);
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

bool mcRPL::DataManager::
_initEvaluate(mcRPL::Transition &trans,
              int subspcGlbID,
              mcRPL::EvaluateBR br2Eval,
              bool ifInitNoData) {
  if(!_setTransNbrhd(trans)) {
    return false;
  }
  if(br2Eval != mcRPL::EVAL_INTERIOR) {
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

  if(br2Eval != mcRPL::EVAL_INTERIOR && trans.needExchange()) {
    trans.setUpdateTracking(true);
  }

  return true;
}

bool mcRPL::DataManager::
_fnlzEvaluate(mcRPL::Transition &trans,
              int subspcGlbID) {
  trans.setUpdateTracking(false);
  return true;
}

bool mcRPL::DataManager::
_workerReadSubspcs(mcRPL::Transition &trans,
                   int subspcGlbID,
                   mcRPL::ReadingOption readOpt) {
  const vector<string> &vInLyrNames = trans.getInLyrNames();
  for(int iLyr = 0; iLyr < vInLyrNames.size(); iLyr++) {
    const string &inLyrName = vInLyrNames[iLyr];
    mcRPL::Layer *pLyr = getLayerByName(inLyrName.c_str(), true);
    if(pLyr == NULL) {
      return false;
    }

    if(pLyr->isDecomposed()) { // decomposed Layer
      if(readOpt == mcRPL::PARA_READING) { // parallel reading
        if(!pLyr->loadSubCellspaceByGDAL(subspcGlbID)) {
          return false;
        }
      }
      else if(readOpt == mcRPL::CENT_READING ||
              readOpt == mcRPL::CENTDEL_READING) { // centralized reading
        if(!_iTransfSubspaceBegin(0, _prc.id(), inLyrName, subspcGlbID)) {
          return false;
        }
      }
      else if(readOpt == mcRPL::PGT_READING) {
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

bool mcRPL::DataManager::
_workerWriteSubspcs(mcRPL::Transition &trans,
                    int subspcGlbID,
                    mcRPL::WritingOption writeOpt) {
  if(writeOpt == mcRPL::NO_WRITING) {
    return true;
  }

  mcRPL::TaskSignal req = mcRPL::SUBMIT_SGNL;
  int submit2ID = _prc.hasWriter() ? 1 : 0; // if a writer exists, send to writer; otherwise master
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    const string &outLyrName = vOutLyrNames[iOutLyr];
    mcRPL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
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

    if(writeOpt == mcRPL::PARA_WRITING ||
       writeOpt == mcRPL::PARADEL_WRITING) {
      if(!pOutLyr->writeTmpSubCellspaceByGDAL(subspcGlbID, _prc.nProcesses())) {
        return false;
      }
      if(writeOpt == mcRPL::PARADEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
    }
    else if(writeOpt == mcRPL::CENT_WRITING ||
            writeOpt == mcRPL::CENTDEL_WRITING) {
      int lyrID = getLayerIdxByName(outLyrName.c_str(), true);
      MPI_Send(&req, 1, MPI_INT, submit2ID, mcRPL::REQUEST_TAG, _prc.comm());
      int sendSubspcInfo[2];
      sendSubspcInfo[0] = lyrID;
      sendSubspcInfo[1] = subspcGlbID;
      MPI_Send(sendSubspcInfo, 2, MPI_INT, submit2ID, mcRPL::SUBMIT_TAG, _prc.comm());
      if(!_iTransfSubspaceBegin(_prc.id(), submit2ID, outLyrName, subspcGlbID)) {
        return false;
      }
    }
    else if(writeOpt == mcRPL::PGT_WRITING ||
            writeOpt == mcRPL::PGTDEL_WRITING) {
      if(!pOutLyr->writeSubCellspaceByPGTIOL(subspcGlbID)) {
        return false;
      }

      if(writeOpt == mcRPL::PGTDEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
    }
  } // end -- for(iOutLyr)

  if(writeOpt == mcRPL::PARA_WRITING ||
     writeOpt == mcRPL::PARADEL_WRITING ||
     writeOpt == mcRPL::PGT_WRITING ||
     writeOpt == mcRPL::PGTDEL_WRITING) {
    MPI_Send(&req, 1, MPI_INT, submit2ID, mcRPL::REQUEST_TAG, _prc.comm());
    MPI_Send(&subspcGlbID, 1, MPI_INT, submit2ID, mcRPL::SUBMIT_TAG, _prc.comm());
  }

  return true;
}

bool mcRPL::DataManager::
_dymWriteSubspcs(mcRPL::Transition &trans,
                 int subspcGlbID,
                 mcRPL::WritingOption writeOpt) {
  if(writeOpt == mcRPL::NO_WRITING) {
    return true;
  }

  mcRPL::TaskSignal req = mcRPL::SUBMIT_SGNL;
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    const string &outLyrName = vOutLyrNames[iOutLyr];
    mcRPL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
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

    if(writeOpt == mcRPL::PARA_WRITING ||
       writeOpt == mcRPL::PARADEL_WRITING) {
      if(!pOutLyr->writeTmpSubCellspaceByGDAL(subspcGlbID, _prc.nProcesses())) {
        return false;
      }
      if(writeOpt == mcRPL::PARADEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)){
          cerr << __FILE__ << " function:" << __FUNCTION__
               << " Error: unable to find SubCellspace (" \
               << subspcGlbID << ") to delete" << endl;
        }
      }
    }
    else if((writeOpt == mcRPL::CENT_WRITING || writeOpt == mcRPL::CENTDEL_WRITING) &&
            _prc.hasWriter()) {
      int lyrID = getLayerIdxByName(outLyrName.c_str(), true);
      MPI_Send(&req, 1, MPI_INT, 1, mcRPL::REQUEST_TAG, _prc.comm());
      int sendSubspcInfo[2];
      sendSubspcInfo[0] = lyrID;
      sendSubspcInfo[1] = subspcGlbID;
      MPI_Send(sendSubspcInfo, 2, MPI_INT, 1, mcRPL::SUBMIT_TAG, _prc.comm());
      if(!_iTransfSubspaceBegin(_prc.id(), 1, outLyrName, subspcGlbID)) {
        return false;
      }
    }
    else if(writeOpt == mcRPL::PGT_WRITING ||
            writeOpt == mcRPL::PGTDEL_WRITING) {
      if(!pOutLyr->writeSubCellspaceByPGTIOL(subspcGlbID)) {
        return false;
      }
      if(writeOpt == mcRPL::PGTDEL_WRITING) {
        if(!pOutLyr->delSubCellspace_glbID(subspcGlbID)) {
          cerr << __FILE__ << " function:" << __FUNCTION__
              << " Error: unable to find SubCellspace (" \
              << subspcGlbID << ") to delete" << endl;
        }
      }
    }
  } // end -- for(iOutLyr)

  if((writeOpt == mcRPL::PARA_WRITING ||
      writeOpt == mcRPL::PARADEL_WRITING ||
      writeOpt == mcRPL::PGT_WRITING ||
      writeOpt == mcRPL::PGTDEL_WRITING) &&
     _prc.hasWriter()) {
    MPI_Send(&req, 1, MPI_INT, 1, mcRPL::REQUEST_TAG, _prc.comm());
    MPI_Send(&subspcGlbID, 1, MPI_INT, 1, mcRPL::SUBMIT_TAG, _prc.comm());
  }

  return true;
}

bool mcRPL::DataManager::
_finalWriteSubspcs(mcRPL::Transition &trans,
                   mcRPL::WritingOption writeOpt) {
  if(writeOpt == mcRPL::NO_WRITING) {
    return true;
  }

  if((writeOpt == mcRPL::CENT_WRITING || writeOpt == mcRPL::CENTDEL_WRITING) &&
     !_prc.hasWriter() &&
     !_prc.isMaster()) {
    mcRPL::TaskSignal req = mcRPL::SUBMIT_SGNL;
    const IntVect *pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
    if(pvSubspcIDs != NULL) {
      for(int iSubspc = 0; iSubspc < pvSubspcIDs->size(); iSubspc++) {
        int subspcGlbID = pvSubspcIDs->at(iSubspc);

        const vector<string> &vOutLyrNames = trans.getOutLyrNames();
        for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
          const string &outLyrName = vOutLyrNames[iOutLyr];
          mcRPL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
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
          MPI_Send(&req, 1, MPI_INT, 0, mcRPL::REQUEST_TAG, _prc.comm());
          int sendSubspcInfo[2];
          sendSubspcInfo[0] = lyrID;
          sendSubspcInfo[1] = subspcGlbID;
          MPI_Send(sendSubspcInfo, 2, MPI_INT, 0, mcRPL::SUBMIT_TAG, _prc.comm());
          if(!_iTransfSubspaceBegin(_prc.id(), 0, outLyrName, subspcGlbID)) {
            return false;
          }

        } // end -- for(iOutLyr)
      } // end -- for(iSubspc)
    } // end -- if(pvSubspcIDs != NULL)
  }

  return true;
}

mcRPL::EvaluateReturn mcRPL::DataManager::
_evalNone(mcRPL::Transition &trans,
          int subspcGlbID,
          bool ifInitNoData) {
  if(!_initEvaluate(trans, subspcGlbID, mcRPL::EVAL_WORKBR, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to initialize the Transition" \
         << endl;
    return mcRPL::EVAL_FAILED;
  }
  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcRPL::EVAL_FAILED;
  }

  return mcRPL::EVAL_SUCCEEDED;
}

mcRPL::EvaluateReturn mcRPL::DataManager::
_evalRandomly(mcRPL::Transition &trans,
              int subspcGlbID,
              bool ifInitNoData) {
  if(!_initEvaluate(trans, subspcGlbID, mcRPL::EVAL_WORKBR, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to initialize the Transition" \
         << endl;
    return mcRPL::EVAL_FAILED;
  }
  if(trans.needExchange()) {
    trans.setUpdateTracking(true);
  }

  mcRPL::Layer *pPrmLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrmLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to find the primary Layer" \
         << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::Cellspace *pPrmCellspace = NULL;
  if(subspcGlbID == mcRPL::ERROR_ID) { // if evaluate the whole Cellspace
    pPrmCellspace = pPrmLyr->cellspace();
  }
  else { // if evaluate a SubCellspace
    pPrmCellspace = pPrmLyr->subCellspace_glbID(subspcGlbID, true);
  }
  if(pPrmCellspace == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace" \
        << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::CoordBR workBR;
  if(subspcGlbID == mcRPL::ERROR_ID) { // if evaluate the whole Cellspace
    if(!pPrmCellspace->info()->calcWorkBR(&workBR, trans.getNbrhd(), true)) {
      return mcRPL::EVAL_FAILED;
    }
  }
  else { // if evaluate a SubCellspace
    workBR = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->workBR();
  }
  mcRPL::EvaluateReturn done = trans.evalRandomly(workBR);

  if(!_fnlzEvaluate(trans, subspcGlbID)) {
    return mcRPL::EVAL_FAILED;
  }

  return done;
}

mcRPL::EvaluateReturn mcRPL::DataManager::
_evalSelected(mcRPL::Transition &trans,
              const mcRPL::LongVect &vGlbIdxs,
              int subspcGlbID,
              mcRPL::EvaluateBR br2Eval,
              bool ifInitNoData) {
  if(br2Eval != mcRPL::EVAL_WORKBR) {
    if(subspcGlbID == mcRPL::ERROR_ID) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
          << " Error: cannot evaluate the Edges or Interior of a whole Cellspace" \
          << endl;
      return mcRPL::EVAL_FAILED;
    }
    if(trans.getOutLyrNames().empty()) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: no output Layers to evaluate Edges or Interior" \
           << endl;
      return mcRPL::EVAL_FAILED;
    }
    if(!trans.isOutLyr(trans.getPrimeLyrName())) {
      cerr << __FILE__ << " function:" << __FUNCTION__ \
           << " Error: the primary Layer (" << trans.getPrimeLyrName() \
           << ") is NOT an output Layer" << endl;
      return mcRPL::EVAL_FAILED;
    }
  }

  if(!_initEvaluate(trans, subspcGlbID, br2Eval, ifInitNoData)) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize the Transition" \
        << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::Layer *pPrmLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrmLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to find the primary Layer" \
        << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::Cellspace *pPrmCellspace = NULL;
  if(subspcGlbID == mcRPL::ERROR_ID) { // if evaluate the whole Cellspace
    pPrmCellspace = pPrmLyr->cellspace();
  }
  else { // if evaluate a SubCellspace
    pPrmCellspace = pPrmLyr->subCellspace_glbID(subspcGlbID, true);
  }
  if(pPrmCellspace == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to find the primary Cellspace" \
        << endl;
    return mcRPL::EVAL_FAILED;
  }

  mcRPL::CoordBR workBR;
  const mcRPL::CoordBR *pWorkBR;
  mcRPL::EvaluateReturn done = mcRPL::EVAL_SUCCEEDED;
  switch(br2Eval) {
    case mcRPL::EVAL_WORKBR:
      if(subspcGlbID == mcRPL::ERROR_ID) { // if evaluate the whole Cellspace
        if(!pPrmCellspace->info()->calcWorkBR(&workBR, trans.getNbrhd(), true)) {
          return mcRPL::EVAL_FAILED;
        }
        done = trans.evalSelected(workBR, vGlbIdxs);
      }
      else { // if evaluate a SubCellspace
        pWorkBR = &(((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->workBR());
        for(int iIdx = 0; iIdx < vGlbIdxs.size(); iIdx++) {
          mcRPL::CellCoord lclCoord = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->glbIdx2lclCoord(vGlbIdxs[iIdx]);
          if(pWorkBR->ifContain(lclCoord)) {
            done = trans.evaluate(lclCoord);
            if(done == mcRPL::EVAL_FAILED ||
               done == mcRPL::EVAL_TERMINATED) {
              return done;
            }
          }
        }
      }
      break;
    case mcRPL::EVAL_EDGES:
      for(int iEdge = 0; iEdge < ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->nEdges(); iEdge++) {
        pWorkBR = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->edgeBR(iEdge);
        if(pWorkBR != NULL) {
          for(int iIdx = 0; iIdx < vGlbIdxs.size(); iIdx++) {
            mcRPL::CellCoord lclCoord = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->glbIdx2lclCoord(vGlbIdxs[iIdx]);
            if(pWorkBR->ifContain(lclCoord)) {
              done = trans.evaluate(lclCoord);
              if(done == mcRPL::EVAL_FAILED ||
                 done == mcRPL::EVAL_TERMINATED) {
                return done;
              }
            }
          } // end -- for(iIdx)
        }
      } // end -- for(iEdge)
      break;
    case mcRPL::EVAL_INTERIOR:
      pWorkBR = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->interiorBR();
      if(pWorkBR != NULL) {
        for(int iIdx = 0; iIdx < vGlbIdxs.size(); iIdx++) {
          mcRPL::CellCoord lclCoord = ((mcRPL::SubCellspace *)pPrmCellspace)->subInfo()->glbIdx2lclCoord(vGlbIdxs[iIdx]);
          if(pWorkBR->ifContain(lclCoord)) {
            done = trans.evaluate(lclCoord);
            if(done == mcRPL::EVAL_FAILED ||
               done == mcRPL::EVAL_TERMINATED) {
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
    return mcRPL::EVAL_FAILED;
  }

  return done;
}

mcRPL::EvaluateReturn mcRPL::DataManager::
_masterTF(mcRPL::Transition &trans,
		  mcRPL::ReadingOption readOpt,
		  mcRPL::WritingOption writeOpt) {
  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  mcRPL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  mcRPL::TaskSignal req;
  MPI_Status status;

  bool delSentSpc = (readOpt == mcRPL::CENTDEL_READING) ? true : false;
  bool writeRecvSpc = (writeOpt != mcRPL::NO_WRITING) ? true : false;
  bool delRecvSpc = (writeOpt == mcRPL::CENTDEL_WRITING) ? true : false;

  while(_nActiveWorkers > 0) {
    MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcRPL::REQUEST_TAG, _prc.comm(), &status);
    int workerID = status.MPI_SOURCE;

    if((mcRPL::TaskSignal)req == mcRPL::NEWTASK_SGNL) { // worker requesting new task
      if(!_mOwnerships.allMapped()) { // there still are tasks
        int mappedSubspcID = _mOwnerships.mappingNextTo(workerID);
        if(mappedSubspcID == ERROR_ID) {
          _prc.abort();
          return mcRPL::EVAL_FAILED;
        }
        MPI_Send(&mappedSubspcID, 1, MPI_INT, workerID, mcRPL::DSTRBT_TAG, _prc.comm());

        if(readOpt == mcRPL::CENT_READING ||
           readOpt == mcRPL::CENTDEL_READING) { // centralized reading
          const vector<string> &vInLyrNames = trans.getInLyrNames();
          for(int iLyr = 0; iLyr < vInLyrNames.size(); iLyr++) {
            const string &inLyrName = vInLyrNames[iLyr];
            mcRPL::Layer *pLyr = getLayerByName(inLyrName.c_str(), true);
            if(pLyr == NULL) {
              _prc.abort();
              return mcRPL::EVAL_FAILED;
            }
            if(pLyr->isDecomposed()) { // decomposed Layer
              mcRPL::SubCellspace *pSubspc = pLyr->subCellspace_glbID(mappedSubspcID, false);
              if(pSubspc == NULL) {
                if(!pLyr->loadSubCellspaceByGDAL(mappedSubspcID)) {
                  _prc.abort();
                  return mcRPL::EVAL_FAILED;
                }
              }
              if(!_iTransfSubspaceBegin(_prc.id(), workerID, inLyrName, mappedSubspcID)) {
                _prc.abort();
                return mcRPL::EVAL_FAILED;
              }
            } // end -- if(pLyr->isDecomposed())
          } // end -- for(int iLyr = 0; iLyr < vLyrIDs.size(); iLyr++)
        } // end -- if(readOpt == mcRPL::CENT_READING || readOpt == mcRPL::CENTDEL_READING)
      } // end -- if(!_mOwnerships.allMapped())

      else { // all tasks are assigned
        mcRPL::TaskSignal resp = mcRPL::QUIT_SGNL;
        MPI_Send(&resp, 1, MPI_INT, workerID, mcRPL::DSTRBT_TAG, _prc.comm());
      }
    } // end -- if((TaskSignal)req == NEWTASK_SGNL)

    else if((mcRPL::TaskSignal)req == mcRPL::SUBMIT_SGNL) { // worker submitting result
      if(writeOpt == mcRPL::PARA_WRITING ||
         writeOpt == mcRPL::PARADEL_WRITING) { // parallel writing
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
        if(!mergeTmpSubspcByGDAL(trans, recvSubspcID)) {
          _prc.abort();
          return mcRPL::EVAL_FAILED;
        }
      }
      else if(writeOpt == mcRPL::CENT_WRITING ||
              writeOpt == mcRPL::CENTDEL_WRITING) { // centralized writing
        int recvSubspcInfo[2];
        MPI_Recv(recvSubspcInfo, 2, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
        int lyrID = recvSubspcInfo[0];
        int subspcGlbID = recvSubspcInfo[1];
        string lyrName = getLayerNameByIdx(lyrID);

        if(!_iTransfSubspaceBegin(workerID, _prc.id(), lyrName, subspcGlbID)) {
          _prc.abort();
          return EVAL_FAILED;
        }
      }
      else if(writeOpt == mcRPL::PGT_WRITING ||
              writeOpt == mcRPL::PGTDEL_WRITING) {
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
      }
    } // end -- if((TaskSignal)req == SUBMIT_SGNL)

    else if((mcRPL::TaskSignal)req == mcRPL::QUIT_SGNL) { // worker quitting
      //cout << "Process "<< workerID << " quit" << endl;
      _nActiveWorkers--;
    }

    // test pending transfers, only for centralized I/O
    if(!_vBcstSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltBcstIDs, mcRPL::BCST_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }
    if(!_vSendSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }

  } // end -- while(_nActiveWorkers > 0)

  // complete all pending transfers, only for centralized I/O
  while(!_vBcstSpcReqs.empty() && nCmpltBcsts != MPI_UNDEFINED) {
    nCmpltBcsts = _iTransfSpaceTest(vCmpltBcstIDs, mcRPL::BCST_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }
  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }

  _clearTransfSpaceReqs();

  return mcRPL::EVAL_SUCCEEDED;
}
mcRPL::EvaluateReturn mcRPL::DataManager::
_writerTF(mcRPL::Transition &trans,
          mcRPL::WritingOption writeOpt) {
  int nCmpltRecvs = 0;
  mcRPL::IntVect vCmpltRecvIDs;
  mcRPL::TaskSignal req;
  MPI_Status status;

  bool delSentSpc = false;
  bool writeRecvSpc = (writeOpt != mcRPL::NO_WRITING) ? true : false;
  bool delRecvSpc = (writeOpt == mcRPL::CENTDEL_WRITING) ? true : false;

  while(_nActiveWorkers > 0) {
    MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcRPL::REQUEST_TAG, _prc.comm(), &status);
    int workerID = status.MPI_SOURCE;

    if((mcRPL::TaskSignal)req == mcRPL::SUBMIT_SGNL) { // worker submitting result
      if(writeOpt == mcRPL::PARA_WRITING ||
         writeOpt == mcRPL::PARADEL_WRITING) { // parallel writing
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
        if(!mergeTmpSubspcByGDAL(trans, recvSubspcID)) {
          _prc.abort();
          return mcRPL::EVAL_FAILED;
        }
      }
      else if(writeOpt == mcRPL::CENT_WRITING ||
              writeOpt == mcRPL::CENTDEL_WRITING) { // centralized writing
        int recvSubspcInfo[2];
        MPI_Recv(recvSubspcInfo, 2, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
        int lyrID = recvSubspcInfo[0];
        int subspcGlbID = recvSubspcInfo[1];
        string lyrName = getLayerNameByIdx(lyrID);

        if(!_iTransfSubspaceBegin(workerID, _prc.id(), lyrName, subspcGlbID)) {
          _prc.abort();
          return EVAL_FAILED;
        }
      }
      else if(writeOpt == mcRPL::PGT_WRITING ||
              writeOpt == mcRPL::PGTDEL_WRITING) {
        int recvSubspcID;
        MPI_Recv(&recvSubspcID, 1, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
      }
    } // end -- if((TaskSignal)req == SUBMIT_SGNL)

    else if((mcRPL::TaskSignal)req == mcRPL::QUIT_SGNL) { // worker quitting
      //cout << "Process "<< workerID << " quit" << endl;
      _nActiveWorkers--;
    }

    // test pending transfers, only for centralized I/O
    if(!_vRecvSpcReqs.empty()) {
      _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
    }

  } // end -- while(_nActiveWorkers > 0)

  // complete all pending transfers, only for centralized I/O
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSpc, writeRecvSpc, delRecvSpc);
  }

  _clearTransfSpaceReqs();

  return mcRPL::EVAL_SUCCEEDED;
}


mcRPL::EvaluateReturn mcRPL::DataManager::
_masterST(mcRPL::Transition &trans,
          mcRPL::WritingOption writeOpt) {
  mcRPL::TaskSignal req;
  MPI_Status status;

  if(!_prc.hasWriter()) {
    if(writeOpt == mcRPL::PARA_WRITING ||
       writeOpt == mcRPL::PARADEL_WRITING) { // parallel writing
      while(_nActiveWorkers > 0) {
        MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcRPL::REQUEST_TAG, _prc.comm(), &status);
        int workerID = status.MPI_SOURCE;
        if((mcRPL::TaskSignal)req == mcRPL::QUIT_SGNL) { // worker quitting
          //cout << "Process "<< workerID << " quit" << endl;
          _nActiveWorkers--;
        }
      } // end -- while(_nActiveWorkers > 0)
      if(!mergeAllTmpSubspcsByGDAL(trans)) {
        _prc.abort();
        return mcRPL::EVAL_FAILED;;
      }
    } // end -- parallel writing

    else if(writeOpt == mcRPL::CENT_WRITING ||
            writeOpt == mcRPL::CENTDEL_WRITING) { // centralized writing
      bool delRecvSpc = (writeOpt == mcRPL::CENTDEL_WRITING) ? true : false;
      int nCmpltRecvs = 0;
      mcRPL::IntVect vCmpltRecvIDs;

      while(_nActiveWorkers > 0) {
        MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcRPL::REQUEST_TAG, _prc.comm(), &status);
        int workerID = status.MPI_SOURCE;
        if((mcRPL::TaskSignal)req == mcRPL::SUBMIT_SGNL) { // worker submitting result
          int recvSubspcInfo[2];
          MPI_Recv(recvSubspcInfo, 2, MPI_INT, workerID, mcRPL::SUBMIT_TAG, _prc.comm(), &status);
          int lyrID = recvSubspcInfo[0];
          int subspcGlbID = recvSubspcInfo[1];
          string lyrName = getLayerNameByIdx(lyrID);

          if(!_iTransfSubspaceBegin(workerID, _prc.id(), lyrName, subspcGlbID)) {
            _prc.abort();
            return EVAL_FAILED;
          }
        } // end -- if((TaskSignal)req == SUBMIT_SGNL)

        else if((mcRPL::TaskSignal)req == mcRPL::QUIT_SGNL) { // worker quitting
          //cout << "Process "<< workerID << " quit" << endl;
          _nActiveWorkers--;
        }

        if(!_vRecvSpcReqs.empty()) {
          _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, true, true, delRecvSpc);
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
            mcRPL::Layer *pOutLyr = getLayerByName(outLyrName.c_str(), true);
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
        nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, true, true, delRecvSpc);
      }

    } // end -- centralized writing

    else if(writeOpt == mcRPL::PGT_WRITING ||
            writeOpt == mcRPL::PGTDEL_WRITING) { // parallel writing
      while(_nActiveWorkers > 0) {
        MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, mcRPL::REQUEST_TAG, _prc.comm(), &status);
        int workerID = status.MPI_SOURCE;
        if((mcRPL::TaskSignal)req == mcRPL::QUIT_SGNL) { // worker quitting
          //cout << "Process "<< workerID << " quit" << endl;
          _nActiveWorkers--;
        }
      } // end -- while(_nActiveWorkers > 0)
    } // end -- parallel writing
  }  // end -- if(!_prc.hasWriter())

  _clearTransfSpaceReqs();

  return mcRPL::EVAL_SUCCEEDED;
}

mcRPL::EvaluateReturn mcRPL::DataManager::
_writerST(mcRPL::Transition &trans,
          mcRPL::WritingOption writeOpt) {
  return _writerTF(trans, writeOpt);
}

bool mcRPL::DataManager::
_calcExchangeBRs(mcRPL::Transition &trans) {
  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
    mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    if(!pOutLyr->calcAllBRs(trans.onlyUpdtCtrCell(), getNbrhdByName(trans.getNbrhdName().c_str()))) {
      return false;
    }
  }
  return true;
}

bool mcRPL::DataManager::
_calcExchangeRoutes(mcRPL::Transition &trans) {
  _mSendRoutes.clear();
  _mRecvRoutes.clear();

  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  const mcRPL::IntVect *pvAssignments = _mOwnerships.subspcIDsOnPrc(_prc.id());

  if(trans.needExchange() && !vOutLyrNames.empty() &&
     pvAssignments != NULL && !pvAssignments->empty()) {
    mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrNames[0].c_str(), true); // only one output Layer is needed for calculating the exchange map
    if(pOutLyr == NULL) {
      return false;
    }
    for(int iAssignment = 0; iAssignment < pvAssignments->size(); iAssignment++) {
      int subspcID = pvAssignments->at(iAssignment);
      const mcRPL::SubCellspaceInfo *pSubspcInfo = pOutLyr->subCellspaceInfo_glbID(subspcID, true);
      if(pSubspcInfo == NULL) {
        return false;
      }

      for(int iDir = 0; iDir < pSubspcInfo->nNbrDirs(); iDir++) {
        const mcRPL::IntVect &vNbrSpcIDs = pSubspcInfo->nbrSubSpcIDs(iDir);
        for(int iNbr = 0; iNbr < vNbrSpcIDs.size(); iNbr++) {
          int nbrSubspcID = vNbrSpcIDs[iNbr];
          int nbrPrc = _mOwnerships.findPrcBySubspcID(nbrSubspcID);

          const mcRPL::CoordBR *pSendBR = pSubspcInfo->sendBR(iDir, iNbr);
          if(pSendBR != NULL && nbrPrc >= 0) {
            if(!_mSendRoutes.add(subspcID, iDir, iNbr, nbrPrc)) {
              return false;
            }
          }

          const mcRPL::SubCellspaceInfo *pNbrSubspcInfo = pOutLyr->subCellspaceInfo_glbID(nbrSubspcID, true);
          if(pNbrSubspcInfo == NULL) {
            return false;
          }
          int iOppDir = pNbrSubspcInfo->oppositeDir(iDir);
          const mcRPL::IntVect &vOppNbrSpcIDs = pNbrSubspcInfo->nbrSubSpcIDs(iOppDir);
          mcRPL::IntVect::const_iterator itrOppNbrSpcID = std::find(vOppNbrSpcIDs.begin(), vOppNbrSpcIDs.end(), subspcID);
          if(itrOppNbrSpcID != vOppNbrSpcIDs.end()) {
            int iOppNbr = itrOppNbrSpcID - vOppNbrSpcIDs.begin();
            const mcRPL::CoordBR *pOppSendBR = pNbrSubspcInfo->sendBR(iOppDir, iOppNbr);
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

bool mcRPL::DataManager::
_loadCellStream(mcRPL::CellStream &cells2load) {
  long cellGlbIdx;
  void *aCellVal;
  const vector<pair<unsigned int, unsigned int> >& vLyrInfos = cells2load.getLayerInfos();
  for(int iLyr = 0; iLyr < vLyrInfos.size(); iLyr++) {
    unsigned int lyrID = vLyrInfos[iLyr].first;
    unsigned int dataSize = vLyrInfos[iLyr].second;
    int nCells = cells2load.getNumCellsOnLayer(lyrID);
    void* aCells = cells2load.getCellsOnLayer(lyrID);

    mcRPL::Layer *pLyr = getLayerByIdx(lyrID, true);
    if(pLyr == NULL) {
      return false;
    }
    if(!pLyr->loadCellStream(aCells, nCells)) {
      return false;
    }
  } // end --- for(iLyr)
  return true;
}

bool mcRPL::DataManager::
_makeCellStream(mcRPL::Transition &trans) {
  _vSendCellReqs.clear();
  _vRecvCellReqs.clear();
  _mSendCells.clear();
  _mRecvCells.clear();

  const vector<string> &vOutLyrNames = trans.getOutLyrNames();
  if(vOutLyrNames.empty()) {
    return true;
  }

  mcRPL::ExchangeMap::iterator itrSendRoute = _mSendRoutes.begin();
  while(itrSendRoute != _mSendRoutes.end()) {
    int toPrc = itrSendRoute->first;
    vector<mcRPL::ExchangeNode> &vSendNodes = itrSendRoute->second;
    for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
      mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
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
        mcRPL::SubCellspace *pSubspc = pOutLyr->subCellspace_glbID(subspcID, true);
        if(pSubspc == NULL) {
          return false;
        }
        const mcRPL::CoordBR *pSendBR = pSubspc->subInfo()->sendBR(iDir, iNbr);
        const mcRPL::LongVect &vUpdtIdxs = pSubspc->getUpdatedIdxs();
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
    mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
    if(pOutLyr == NULL) {
      return false;
    }
    pOutLyr->clearUpdateTracks();
  }

  mcRPL::ExchangeMap::iterator itrRecvRoute = _mRecvRoutes.begin();
  while(itrRecvRoute != _mRecvRoutes.end()) {
    int fromPrc = itrRecvRoute->first;
    vector<mcRPL::ExchangeNode> &vRecvNodes = itrRecvRoute->second;
    for(int iOutLyr = 0; iOutLyr < vOutLyrNames.size(); iOutLyr++) {
      mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrNames[iOutLyr].c_str(), true);
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

bool mcRPL::DataManager::
_iExchangeBegin() {
  vector<MPI_Status> vSendStats(_vSendCellReqs.size());
  MPI_Waitall(_vSendCellReqs.size(), &(_vSendCellReqs[0]), &(vSendStats[0]));
  _vSendCellReqs.clear();

  mcRPL::ExchangeMap::iterator itrSendRoute = _mSendRoutes.begin();
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

  mcRPL::ExchangeMap::iterator itrRecvRoute = _mRecvRoutes.begin();
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

bool mcRPL::DataManager::
_iExchangeEnd() {
  vector<MPI_Status> vSendStats(_vSendCellReqs.size());
  MPI_Waitall(_vSendCellReqs.size(), &(_vSendCellReqs[0]), &(vSendStats[0]));
  _vSendCellReqs.clear();
  _mSendCells.clear();

  vector<MPI_Status> vRecvStats(_vRecvCellReqs.size());
  MPI_Waitall(_vRecvCellReqs.size(), &(_vRecvCellReqs[0]), &(vRecvStats[0]));
  _vRecvCellReqs.clear();
  map<int, mcRPL::CellStream>::iterator itrRecvCells = _mRecvCells.begin();
  while(itrRecvCells != _mRecvCells.end()) {
    mcRPL::CellStream &recvCells = itrRecvCells->second;
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
mcRPL::DataManager::
DataManager()
  :_nActiveWorkers(0) {
  GDALAllRegister();
}

mcRPL::DataManager::
~DataManager() {
  clearLayers();
  clearNbrhds();
}

bool mcRPL::DataManager::
initMPI(MPI_Comm comm,
        bool hasWriter) {
  return _prc.set(comm, hasWriter);
}

void mcRPL::DataManager::
finalizeMPI() {
  _prc.finalize();
}

mcRPL::Process& mcRPL::DataManager::
mpiPrc() {
  return _prc;
}

int mcRPL::DataManager::
nLayers() const {
  return _lLayers.size();
}

mcRPL::Layer* mcRPL::DataManager::
addLayer(const char *aLyrName) {
  mcRPL::Layer *pLyr = getLayerByName(aLyrName, false);
  if(pLyr != NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__
         << " Error: Layer name (" << aLyrName << ") already exists in DataManager. " \
         << "Failed to add new Layer."<< endl;
  }
  else {
    
    _lLayers.push_back(mcRPL::Layer(aLyrName));
    pLyr = &(_lLayers.back());
   
  }
  return pLyr;
}


/*---------------------------GDAL----------------------------------------------------*/
mcRPL::Layer* mcRPL::DataManager::
addLayerByGDAL(const char *aLyrName,
               const char *aGdalFileName,
               int iBand,
               bool pReading) {
  mcRPL::Layer *pLyr = addLayer(aLyrName);

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

bool mcRPL::DataManager::
createLayerGDAL(const char *aLyrName,
                const char *aGdalFileName,
                const char *aGdalFormat,
                char **aGdalOptions) {
  mcRPL::Layer *pLyr = getLayerByName(aLyrName, true);
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

void mcRPL::DataManager::
closeGDAL() {
  if(!_lLayers.empty()) {
    list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
    while(itrLyr != _lLayers.end()) {
      itrLyr->closeGdalDS();
      itrLyr++;
    }
  }
}

/*---------------------------hsj-------------PGIOL--------------------*/
mcRPL::Layer* mcRPL::DataManager::
addLayerByPGTIOL(const char *aLyrName,
                 const char *aPgtiolFileName,
                 int iBand,
                 bool pReading) {
  mcRPL::Layer *pLyr = addLayer(aLyrName);

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

bool mcRPL::DataManager::
createLayerPGTIOL(const char *aLyrName,
                  const char *aPgtiolFileName,
                  char **aPgtiolOptions) {
  mcRPL::Layer *pLyr = getLayerByName(aLyrName, true);
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

void mcRPL::DataManager::
closePGTIOL() {
  if(!_lLayers.empty()) {
    list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
    while(itrLyr != _lLayers.end()) {
      itrLyr->closePgtiolDS();
      itrLyr++;
    }
  }
}

void mcRPL::DataManager::
closeDatasets() {
  if(!_lLayers.empty()) {
    list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
    while(itrLyr != _lLayers.end()) {
      itrLyr->closeGdalDS();
      itrLyr->closePgtiolDS();
      itrLyr++;
    }
  }
}

bool mcRPL::DataManager::
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
    list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
    advance(itrLyr, lyrID);
    _lLayers.erase(itrLyr);
  }
  return done;
}

bool mcRPL::DataManager::
rmvLayerByName(const char *aLyrName,
               bool warning) {
  bool done = false;
  list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
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

void mcRPL::DataManager::
clearLayers() {
  _lLayers.clear();
}

mcRPL::Layer* mcRPL::DataManager::
getLayerByIdx(int lyrID,
              bool warning) {
  mcRPL::Layer *pLayer = NULL;
  if(lyrID >= 0 && lyrID < _lLayers.size()) {
    list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
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

const mcRPL::Layer* mcRPL::DataManager::
getLayerByIdx(int lyrID,
              bool warning) const {
  const mcRPL::Layer *pLayer = NULL;
  if(lyrID >= 0 && lyrID < _lLayers.size()) {
    list<mcRPL::Layer>::const_iterator itrLyr = _lLayers.begin();
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

int mcRPL::DataManager::
getLayerIdxByName(const char *aLyrName,
                  bool warning) const {
  int lyrIdx = ERROR_ID, lyrCount = 0;
  bool found = false;
  list<mcRPL::Layer>::const_iterator itrLyr = _lLayers.begin();
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

mcRPL::Layer* mcRPL::DataManager::
getLayerByName(const char *aLyrName,
               bool warning) {
  mcRPL::Layer* pLyr = NULL;
  list<mcRPL::Layer>::iterator itrLyr = _lLayers.begin();
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

const mcRPL::Layer* mcRPL::DataManager::
getLayerByName(const char *aLyrName,
               bool warning) const {
  const mcRPL::Layer* pLyr = NULL;
  list<mcRPL::Layer>::const_iterator itrLyr = _lLayers.begin();
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

const char* mcRPL::DataManager::
getLayerNameByIdx(int lyrID,
                  bool warning) const {
  const char *aLyrName = NULL;
  const mcRPL::Layer *pLyr = getLayerByIdx(lyrID, warning);
  if(pLyr != NULL) {
    aLyrName = pLyr->name();
  }
  return aLyrName;
}
bool mcRPL::DataManager::
beginGetRand(const char *aLyrName)
{
	mcRPL::Layer *pLyr = getLayerByName(aLyrName, true);
	if(pLyr == NULL) {
		_prc.abort();
		return false;
	}
	   const mcRPL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
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
bool mcRPL::DataManager::
beginReadingLayer(const char *aLyrName,
                  mcRPL::ReadingOption readOpt) {
  mcRPL::Layer *pLyr = getLayerByName(aLyrName, true);
  if(pLyr == NULL) {
    _prc.abort();
    return false;
  }

  if(!pLyr->isDecomposed()) { // non-decomposed Layer
    if(readOpt == mcRPL::PARA_READING) {
      if(!_prc.isMaster() && !_prc.isWriter()) {
        if(!pLyr->loadCellspaceByGDAL()) {
          return false;
        }
      }
    }
    else if(readOpt == mcRPL::CENT_READING ||
            readOpt == mcRPL::CENTDEL_READING) {
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
    else if(readOpt == mcRPL::PGT_READING) {
      if(!_prc.isMaster() && !_prc.isWriter()) {
        if(!pLyr->loadCellspaceByPGTIOL()) {
          return false;
        }
      }
    }
  } // end -- non-decomposed Layer

  else { // decomposed Layer
    if(readOpt == mcRPL::PARA_READING) { // parallel reading
      const mcRPL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
      if(pvSubspcIDs != NULL) {
        for(int iSub = 0; iSub < pvSubspcIDs->size(); iSub++) {
          int mappedSubspcID = pvSubspcIDs->at(iSub);
          if(!pLyr->loadSubCellspaceByGDAL(mappedSubspcID)) {
            return false;
          }
        }
      }
    } // end -- parallel reading

    else if(readOpt == mcRPL::CENT_READING ||
            readOpt == mcRPL::CENTDEL_READING) { // centralized reading
      if(_prc.isMaster()) { // master process
        mcRPL::IntVect vMappedSubspcIDs = _mOwnerships.mappedSubspcIDs();
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
        const mcRPL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
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

    else if(readOpt == mcRPL::PGT_READING) {
      const mcRPL::IntVect* pvSubspcIDs = _mOwnerships.subspcIDsOnPrc(_prc.id());
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

void mcRPL::DataManager::
finishReadingLayers(mcRPL::ReadingOption readOpt) {
  // Complete all pending transfers, only for centralized I/O
  mcRPL::IntVect vCmpltBcstIDs, vCmpltSendIDs, vCmpltRecvIDs;
  bool delSentSpc = (readOpt == mcRPL::CENTDEL_READING) ? true : false;
  int nCmpltBcsts = 0, nCmpltSends = 0, nCmpltRecvs = 0;
  while(!_vBcstSpcReqs.empty() && nCmpltBcsts != MPI_UNDEFINED) {
    nCmpltBcsts = _iTransfSpaceTest(vCmpltBcstIDs, mcRPL::BCST_TRANSF, delSentSpc, false, false);
  }
  while(!_vSendSpcReqs.empty() && nCmpltSends != MPI_UNDEFINED) {
    nCmpltSends = _iTransfSpaceTest(vCmpltSendIDs, mcRPL::SEND_TRANSF, delSentSpc, false, false);                                   //´«Êý¾Ý
  }
  while(!_vRecvSpcReqs.empty() && nCmpltRecvs != MPI_UNDEFINED) {
    nCmpltRecvs = _iTransfSpaceTest(vCmpltRecvIDs, mcRPL::RECV_TRANSF, delSentSpc, false, false);
  }

  _clearTransfSpaceReqs();
}

int mcRPL::DataManager::
nNbrhds() const {
  return _lNbrhds.size();
}

mcRPL::Neighborhood* mcRPL::DataManager::
addNbrhd(const char *aNbrhdName) {
  mcRPL::Neighborhood *pNbrhd = getNbrhdByName(aNbrhdName, false);
  if(pNbrhd != NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: Neighborhood name (" << aNbrhdName \
         << ") already exists in DataManager. Failed to add new Neighborhood" \
         << endl;
  }
  else {
    _lNbrhds.push_back(mcRPL::Neighborhood(aNbrhdName));
    pNbrhd = &(_lNbrhds.back());
  }
  return pNbrhd;
}

mcRPL::Neighborhood* mcRPL::DataManager::
addNbrhd(const mcRPL::Neighborhood &nbrhd) {
  const char *aNbrhdName = nbrhd.name();
  mcRPL::Neighborhood *pNbrhd = getNbrhdByName(aNbrhdName, false);
  if(pNbrhd != NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Warning: Neighborhood name (" << aNbrhdName \
         << ") already exists in DataManager. Failed to add new Neighborhood" \
        << endl;
  }
  else {
    _lNbrhds.push_back(mcRPL::Neighborhood(nbrhd));
    pNbrhd = &(_lNbrhds.back());
  }
  return pNbrhd;
}

bool mcRPL::DataManager::
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
    list<mcRPL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
    advance(itrNbrhd, nbrhdID);
    _lNbrhds.erase(itrNbrhd);
  }
  return done;
}

bool mcRPL::DataManager::
rmvNbrhdByName(const char *aNbrhdName,
               bool warning) {
  bool done = false;
  list<mcRPL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
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

void mcRPL::DataManager::
clearNbrhds() {
  _lNbrhds.clear();
}

mcRPL::Neighborhood* mcRPL::DataManager::
getNbrhdByIdx(int nbrhdID,
              bool warning) {
  mcRPL::Neighborhood *pNbrhd = NULL;
  if(nbrhdID >= 0 && nbrhdID < _lNbrhds.size()) {
    list<mcRPL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
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

const mcRPL::Neighborhood* mcRPL::DataManager::
getNbrhdByIdx(int nbrhdID,
              bool warning) const {
  const mcRPL::Neighborhood *pNbrhd = NULL;
  if(nbrhdID >= 0 && nbrhdID < _lNbrhds.size()) {
    list<mcRPL::Neighborhood>::const_iterator itrNbrhd = _lNbrhds.begin();
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

int mcRPL::DataManager::
getNbrhdIdxByName(const char *aNbrhdName,
                  bool warning) {
  int nbrhdIdx = ERROR_ID, nbrhdCount = 0;
  bool found = false;
  list<mcRPL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
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

mcRPL::Neighborhood* mcRPL::DataManager::
getNbrhdByName(const char *aNbrhdName,
               bool warning) {
  mcRPL::Neighborhood* pNbrhd = NULL;
  list<mcRPL::Neighborhood>::iterator itrNbrhd = _lNbrhds.begin();
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

const mcRPL::Neighborhood* mcRPL::DataManager::
getNbrhdByName(const char *aNbrhdName,
               bool warning) const {
  const mcRPL::Neighborhood* pNbrhd = NULL;
  list<mcRPL::Neighborhood>::const_iterator itrNbrhd = _lNbrhds.begin();
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

bool mcRPL::DataManager::
dcmpLayer(int lyrID,
          int nbrhdID,
          int nRowSubspcs,
          int nColSubspcs) {
  mcRPL::Layer *pLyr = getLayerByIdx(lyrID, true);
  if(pLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to apply decomposition on a NULL Layer" \
         << endl;
    return false;
  }

  mcRPL::Neighborhood *pNbrhd = nbrhdID < 0 ? NULL : getNbrhdByIdx(nbrhdID, true);

  return pLyr->decompose(nRowSubspcs, nColSubspcs, pNbrhd);
}

bool mcRPL::DataManager::
dcmpLayer(const char *aLyrName,
          const char *aNbrhdName,
          int nRowSubspcs,
          int nColSubspcs) {
  mcRPL::Layer *pLyr = getLayerByName(aLyrName, true);
  if(pLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to apply decomposition on a NULL Layer" \
         << endl;
    return false;
  }

  mcRPL::Neighborhood *pNbrhd = aNbrhdName == NULL ? NULL : getNbrhdByName(aNbrhdName, true);

  return pLyr->decompose(nRowSubspcs, nColSubspcs, pNbrhd);
}

bool mcRPL::DataManager::
propagateDcmp(int fromLyrID,
              int toLyrID) {
  mcRPL::Layer *pFromLyr = getLayerByIdx(fromLyrID, true);
  if(pFromLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to copy decomposition from a NULL Layer" \
         << endl;
    return false;
  }

  mcRPL::Layer *pToLyr = getLayerByIdx(toLyrID, true);
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

bool mcRPL::DataManager::
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

bool mcRPL::DataManager::
propagateDcmp(const string &fromLyrName,
              const string &toLyrName) {
  mcRPL::Layer *pFromLyr = getLayerByName(fromLyrName.c_str(), true);
  if(pFromLyr == NULL) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
         << " Error: unable to copy decomposition from a NULL Layer" \
         << endl;
    return false;
  }

  mcRPL::Layer *pToLyr = getLayerByName(toLyrName.c_str(), true);
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

bool mcRPL::DataManager::
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

bool mcRPL::DataManager::
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

bool mcRPL::DataManager::
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

bool mcRPL::DataManager::
dcmpLayers(mcRPL::Transition &trans,
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

bool mcRPL::DataManager::
dcmpAllLayers(const char *aNbrhdName,
              int nRowSubspcs,
              int nColSubspcs) {
  list<mcRPL::Layer>::iterator iLyr = _lLayers.begin();
  mcRPL::Layer *p1stLyr = &(*iLyr);
  mcRPL::Neighborhood *pNbrhd = aNbrhdName == NULL ? NULL : getNbrhdByName(aNbrhdName, true);
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

const mcRPL::OwnershipMap& mcRPL::DataManager::
ownershipMap() const {
  return _mOwnerships;
}

bool mcRPL::DataManager::
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

bool mcRPL::DataManager::
initTaskFarm(mcRPL::Transition &trans,
             mcRPL::MappingMethod mapMethod,
             int nSubspcs2Map,
             mcRPL::ReadingOption readOpt) {
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

  mcRPL::Layer *pPrimeLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
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

  mcRPL::IntVect vSubspcIDs = pPrimeLyr->allSubCellspaceIDs();
  mcRPL::IntVect vPrcIDs = _prc.allPrcIDs(false, false); // exclude the master and writer processes
  _mOwnerships.init(vSubspcIDs, vPrcIDs);
  _mOwnerships.mapping(mapMethod, nSubspcs2Map); // initial mapping

  _clearTransfSpaceReqs();

  if(!_initReadInData(trans, readOpt)) {
    _prc.abort();
    return false;
  }

  return true;
}

bool mcRPL::DataManager::
initStaticTask(mcRPL::Transition &trans,
               mcRPL::MappingMethod mapMethod,
               mcRPL::ReadingOption readOpt) {
  if(_prc.nProcesses() <= 1 && _prc.hasWriter()) {
    cerr << __FILE__ << " function:" << __FUNCTION__ \
        << " Error: unable to initialize static tasking when there are " \
        << _prc.nProcesses() << " Processes that include a writer Process" \
        << endl;
    _prc.abort();
    return false;
  }

  mcRPL::Layer *pPrimeLyr = getLayerByName(trans.getPrimeLyrName().c_str(), true);
  if(pPrimeLyr == NULL) {
    _prc.abort();
    return false;
  }

  mcRPL::IntVect vSubspcIDs = pPrimeLyr->allSubCellspaceIDs();
  mcRPL::IntVect vPrcIDs = _prc.allPrcIDs(true, false); // exclude the writer processes
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


bool mcRPL::DataManager::
mergeTmpSubspcByGDAL(mcRPL::Transition &trans,
                     int subspcGlbID) {
  if((_prc.hasWriter() && _prc.isWriter()) ||
     (!_prc.hasWriter() && _prc.isMaster())) {
    const vector<string> &vOutLyrs = trans.getOutLyrNames();
    for(int iOutLyr = 0; iOutLyr < vOutLyrs.size(); iOutLyr++) {
      mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrs[iOutLyr].c_str(), true);
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

bool mcRPL::DataManager::
mergeAllTmpSubspcsByGDAL(mcRPL::Transition &trans) {
  if((_prc.hasWriter() && _prc.isWriter()) ||
     (!_prc.hasWriter() && _prc.isMaster())) {
    const vector<string> &vOutLyrs = trans.getOutLyrNames();
    for(int iOutLyr = 0; iOutLyr < vOutLyrs.size(); iOutLyr++) {
      mcRPL::Layer *pOutLyr = getLayerByName(vOutLyrs[iOutLyr].c_str(), true);
      if(pOutLyr == NULL) {
        _prc.abort();
        return false;
      }
      list<mcRPL::SubCellspaceInfo>::const_iterator itrSubInfo = pOutLyr->allSubCellspaceInfos()->begin();
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
