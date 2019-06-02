#include "mcrpl-dataManager.h"
#include <iostream>
#include <fstream>
#include"mcrpl-cuda.h"
#include "mcrpl-OperatorDevice.h"
using namespace std;
int main(int argc, char *argv[]) {
  const string usage("usage: pCopy workspace input-Filename num-row-subspaces num-col-subspaces task-farming(1/0) io-option(0/1/2/3/4) with-writer(1/0)");

  // Declare a DataManager and initialize MPI
 // bool withWriter = (bool)atoi(argv[argc-1]);
  bool withWriter = 0;
  mcRPL::DataManager copyDM;
  if(!copyDM.initMPI(MPI_COMM_WORLD, withWriter)) {
    cerr << "Error: unable to initialize MPI" << endl;
    return -1;
  }

  // Handle arguments
  //if(argc != 8) {
  //  if(copyDM.mpiPrc().isMaster()) {
  //    cout << usage << endl;
  //  }
  //  copyDM.finalizeMPI();
  //  return -1;
  //}

  string workspace, inFilename, outFilename;
 // workspace.assign(argv[1]);
//  inFilename.assign(workspace + argv[2]);
 // int nRowSubspcs = atoi(argv[3]);
 // int nColSubspcs = atoi(argv[4]);
 // bool taskFarming = (bool)atoi(argv[5]);
  //int ioOption = atoi(argv[6]);
  mcRPL::ReadingOption readOpt;
  mcRPL::WritingOption writeOpt;
  string sReadOpt, sWriteOpt;
   workspace.assign("D:\\data\\newdata\\");
 inFilename.assign("D:\\data\\newdata\\2009-329-flaash.dat"); 
  int nRowSubspcs =3;
  int nColSubspcs = 1;
  bool taskFarming =0;
  int ioOption =3;
  switch(ioOption) {
    case 0:
      readOpt = mcRPL::CENTDEL_READING;
      writeOpt = mcRPL::NO_WRITING;
      sReadOpt = "CENTDEL_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 1:
      readOpt = mcRPL::PARA_READING;
      writeOpt = mcRPL::NO_WRITING;
      sReadOpt = "PARA_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 2:
      readOpt = mcRPL::PGT_READING;
      writeOpt = mcRPL::NO_WRITING;
      sReadOpt = "PGT_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 3:
      readOpt = mcRPL::CENTDEL_READING;
      writeOpt = mcRPL::CENTDEL_WRITING;
      sReadOpt = "CENTDEL_READING";
      sWriteOpt = "CENTDEL_WRITING";
      break;
    case 4:
      readOpt = mcRPL::PARA_READING;
      writeOpt = mcRPL::PARADEL_WRITING;
      sReadOpt = "PARA_READING";
      sWriteOpt = "PARADEL_WRITING";
      break;
    case 5:
      readOpt = mcRPL::PGT_READING;
      writeOpt = mcRPL::PGTDEL_WRITING;
      sReadOpt = "PGT_READING";
      sWriteOpt = "PGTDEL_WRITING";
      break;
    default:
      cerr << "Error: invalid ioOption (" << ioOption << ")" << endl;
      return -1;
  }

  // Record the start time
  double timeStart, timeInit, timeCreate, timeRead, timeEnd;
  copyDM.mpiPrc().sync();
  if(copyDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
    timeStart = MPI_Wtime();
  }

  // Add Layers to the DataManager
  mcRPL::Layer *pInLyr = NULL;
  if(readOpt == mcRPL::PGT_READING) {
    pInLyr = copyDM.addLayerByPGTIOL("INPUT", inFilename.c_str(), 1, true);
  }
  else {
    pInLyr = copyDM.addLayerByGDAL("INPUT", inFilename.c_str(), 1, true);
  }

  mcRPL::Layer *pOutLyr = copyDM.addLayer("OUTPUT");
  pOutLyr->initCellspaceInfo(*(pInLyr->cellspaceInfo()));

  // Declare a Transition
  mcRPL::Transition copyTrans;
  copyTrans.addInputLyr(pInLyr->name(), false);
  copyTrans.addOutputLyr(pOutLyr->name(), true);

  // Decompose the Layers
  //cout << copyDM.mpiPrc().id() << ": decomposing Cellspaces...." << endl;
  if(!copyDM.dcmpLayers(copyTrans, nRowSubspcs, nColSubspcs)) {
    copyDM.mpiPrc().abort();
    return -1;
  }
  mcRPL::pCuf pf=&mcRPL::Transition::cuLocalOperator<Copy>;
  copyDM.mpiPrc().sync();
  if(copyDM.mpiPrc().isMaster()) {
    timeInit = MPI_Wtime();
  }

  if(writeOpt != mcRPL::NO_WRITING) {
    char nPrcs[10]; sprintf(nPrcs, "%d", copyDM.mpiPrc().nProcesses());
    outFilename.assign(workspace + "copy_" + nPrcs + ".tif");
    if(writeOpt == mcRPL::PGTDEL_WRITING) {
      if(!copyDM.createLayerPGTIOL(pOutLyr->name(), outFilename.c_str(), NULL)) {
        copyDM.mpiPrc().abort();
        return -1;
      }
    }
    else {
      if(!copyDM.createLayerGDAL(pOutLyr->name(), outFilename.c_str(), "GTiff", NULL)) {
        copyDM.mpiPrc().abort();
        return -1;
      }
    }
  }

  copyDM.mpiPrc().sync();
  if(copyDM.mpiPrc().isMaster()) {
    timeCreate = MPI_Wtime();
  }

  if(taskFarming) {
    // Initialize task farming
   // cout << copyDM.mpiPrc().id() << ": initializing task farm...." << endl;
    int nSubspcs2Map = withWriter ? 2*(copyDM.mpiPrc().nProcesses()-2) : 2*(copyDM.mpiPrc().nProcesses()-1);
    if(!copyDM.initTaskFarm(copyTrans, mcRPL::CYLC_MAP, nSubspcs2Map, readOpt)) {
      return -1;
    }

    copyDM.mpiPrc().sync();
    if(copyDM.mpiPrc().isMaster()) {
      timeRead = MPI_Wtime();
    }

    // Task farming
    //cout << copyDM.mpiPrc().id() << ": task farming...." << endl;
    if(copyDM.evaluate_TF(mcRPL::EVAL_ALL, copyTrans, readOpt, writeOpt, false, false) != mcRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  else {
    //cout << copyDM.mpiPrc().id() << ": initializing static tasking...." << endl;
    if(!copyDM.initStaticTask(copyTrans, mcRPL::CYLC_MAP, readOpt)) {
      return -1;
    }
    copyDM.mpiPrc().sync();
    if(copyDM.mpiPrc().isMaster()) {
      timeRead = MPI_Wtime();
    }
    //cout << copyDM.mpiPrc().id() << ": static tasking...." << endl;
    if(copyDM.evaluate_ST(mcRPL::EVAL_ALL,copyTrans, writeOpt,true, pf,false) != mcRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }

  // Save the output data
  copyDM.closeDatasets();

  // Record the end time, log computing time
  copyDM.mpiPrc().sync();
  if(copyDM.mpiPrc().isMaster()) {
    //cout << "-------- Completed --------" << endl;
    timeEnd = MPI_Wtime();

    ofstream timeFile;
    string timeFilename(workspace + "copy_time.csv");
    timeFile.open(timeFilename.c_str(), ios::app);
    if(!timeFile) {
      cerr << "Error: unable to open the time log file" << endl;
    }
    timeFile << inFilename << "," \
        << outFilename << "," \
        << copyDM.mpiPrc().nProcesses() << "," \
        << (withWriter?"WITH_WRITER":"NO_WRITER") << "," \
        << nRowSubspcs << "," \
        << nColSubspcs << "," \
        << (taskFarming?"TF":"ST") << "," \
        << sReadOpt << "," \
        << sWriteOpt << "," \
        << timeInit - timeStart << "," \
        << timeCreate - timeInit << "," \
        << timeRead - timeCreate << "," \
        << timeEnd - timeRead << "," \
        << timeEnd - timeStart << "," \
        << endl;
    timeFile.close();

    //cout << copyDM.ownershipMap() << endl;
  }

  // Finalize MPI
  copyDM.finalizeMPI();
  return 0;
}
