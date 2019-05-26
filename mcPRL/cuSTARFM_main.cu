#include "mcPRL-dataManager.h"
#include <iostream>
#include <fstream>
#include"mcPRLcuda.h"
#include "OperatorDevice.h"
using namespace std;
int main(int argc, char *argv[])
{
 // const string usage("usage: pAspect workspace input-demFilename num-row-subspaces num-col-subspaces task-farming(1/0) io-option(0/1/2/3/4) with-writer(1/0)");

  // Declare a DataManager and initialize MPI
 // bool withWriter = (bool)atoi(argv[argc-1]);
  bool withWriter = 0;
  mcPRL::DataManager aspDM;
  if(!aspDM.initMPI(MPI_COMM_WORLD, withWriter)) {
    cerr << "Error: unable to initialize MPI" << endl;
    return -1;
  }

  // Handle arguments
  //if(argc != 8) {
  //  if(aspDM.mpiPrc().isMaster()) {
  //    cout << usage << endl;
  //  }
  //  aspDM.finalizeMPI();
  //  return -1;
  //}
 // int *d_data;
 //int *pAtest_host=new int[20];
 //for(int i=0;i<20;i++)
 //{
	// pAtest_host[i]=i+9;
 //}
 // cudaMalloc((void**)&d_data, sizeof(int)*10 * 2);
 // cudaMemcpy(d_data,pAtest_host,sizeof(int)*20,cudaMemcpyHostToDevice);
 //  devicePrt(d_data);
  string workspace,firstfineFilename, firstcorseFilename, secondcorseFilename,secondfineFilename;
  //workspace.assign(argv[1]);
  //demFilename.assign(workspace + argv[2]);
  int nRowSubspcs = atoi(argv[1]);
  int nColSubspcs = atoi(argv[2]);
  //bool taskFarming = (bool)atoi(argv[5]);
  //int ioOption = atoi(argv[6]);
  workspace.assign("../input/STARFM_DATA/");
  firstfineFilename.assign("../input/STARFM_DATA/2009-329-flaash.dat"); 
  firstcorseFilename.assign("../input/STARFM_DATA/MODO9A1.A2009329-0.dat"); 
  secondcorseFilename.assign("../input/STARFM_DATA/MODO9A1.A2009249.dat");
 // int nRowSubspcs =2;
 // int nColSubspcs = 2;
  bool taskFarming =0;
  int ioOption =3;
  mcPRL::ReadingOption readOpt;
  mcPRL::WritingOption writeOpt;
  string sReadOpt, sWriteOpt;
  switch(ioOption) {
    case 0:
      readOpt = mcPRL::CENTDEL_READING;
      writeOpt = mcPRL::NO_WRITING;
      sReadOpt = "CENTDEL_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 1:
      readOpt = mcPRL::PARA_READING;
      writeOpt = mcPRL::NO_WRITING;
      sReadOpt = "PARA_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 2:
      readOpt = mcPRL::PGT_READING;
      writeOpt = mcPRL::NO_WRITING;
      sReadOpt = "PGT_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 3:
      readOpt = mcPRL::CENTDEL_READING;
      writeOpt = mcPRL::CENTDEL_WRITING;
      sReadOpt = "CENTDEL_READING";
      sWriteOpt = "CENTDEL_WRITING";
      break;
    case 4:
      readOpt = mcPRL::PARA_READING;
      writeOpt = mcPRL::PARADEL_WRITING;
      sReadOpt = "PARA_READING";
      sWriteOpt = "PARADEL_WRITING";
      break;
    case 5:
      readOpt = mcPRL::PGT_READING;
      writeOpt = mcPRL::PGTDEL_WRITING;
      sReadOpt = "PGT_READING";
      sWriteOpt = "PGTDEL_WRITING";
      break;
    default:
      cerr << "Error: invalid ioOption (" << ioOption << ")" << endl;
      return -1;
  }

  // Record the start time
  double timeStart, timeInit, timeCreate, timeRead, timeEnd;
  aspDM.mpiPrc().sync();
  if(aspDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
    timeStart = MPI_Wtime();
  }

  // Add Layers to the DataManager
  mcPRL::Layer *pfirstfineLyr = NULL;
  mcPRL::Layer *pfirstcorseLyr = NULL;
   mcPRL::Layer *psecondcorseLyr = NULL;
  if(readOpt == mcPRL::PGT_READING) {
    pfirstfineLyr = aspDM.addLayerByPGTIOL("firstfine", firstfineFilename.c_str(), 3, true);
	pfirstcorseLyr = aspDM.addLayerByPGTIOL("firstcorse", firstcorseFilename.c_str(), 1, true);
	psecondcorseLyr = aspDM.addLayerByPGTIOL("secondcorse", secondcorseFilename.c_str(), 1, true);
  }
  else {
    pfirstfineLyr = aspDM.addLayerByGDAL("firstfine", firstfineFilename.c_str(), 3, true);
	pfirstcorseLyr = aspDM.addLayerByGDAL("firstcorse", firstcorseFilename.c_str(), 1, true);
	psecondcorseLyr = aspDM.addLayerByGDAL("secondcorse",  secondcorseFilename.c_str(), 1, true);
  }
  const mcPRL::SpaceDims &glbDims = *(pfirstfineLyr->glbDims());
  const mcPRL::CellspaceGeoinfo *pGlbGeoinfo = pfirstfineLyr->glbGeoinfo();
  long tileSize = pfirstfineLyr->tileSize();
  
  mcPRL::Layer *psecondfineLyr = aspDM.addLayer("secondfine");
 psecondfineLyr->initCellspaceInfo(glbDims, typeid(int).name(), sizeof(int), pGlbGeoinfo, tileSize);

  //mcPRL::Layer *pAspLyr = aspDM.addLayer("ASP");
 // pAspLyr->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo, tileSize);
  
  // Add a 3X3 Neighborhood to the DataManager
  mcPRL::Neighborhood* pNbrhd31x31 = aspDM.addNbrhd("Moore31x31");
  pNbrhd31x31->initMoore(31, 1.0, mcPRL::CUSTOM_VIRTUAL_EDGES, 0);

  // Declare a Transition
  mcPRL::Transition aspTrans;
 // aspTrans.scale(1.0);
  aspTrans.setNbrhdByName(pNbrhd31x31->name());
  aspTrans.addInputLyr(pfirstfineLyr->name(), false);
  aspTrans.addInputLyr(psecondcorseLyr->name(), false);
  aspTrans.addInputLyr(pfirstcorseLyr->name(), false);
  aspTrans.addOutputLyr(psecondfineLyr->name(),true);
  //aspTrans.addOutputLyr(pAspLyr->name(), true);
  
  // Decompose the Layers
  //cout << aspDM.mpiPrc().id() << ": decomposing Cellspaces...." << endl;
  if(!aspDM.dcmpLayers(aspTrans, nRowSubspcs, nColSubspcs)) {
    aspDM.mpiPrc().abort();
    return -1;
  }

  aspDM.mpiPrc().sync();
  if(aspDM.mpiPrc().isMaster()) {
    timeInit = MPI_Wtime();
  }

  // Create the output datasets
  if(writeOpt != mcPRL::NO_WRITING) {
    char nPrcs[10]; sprintf(nPrcs, "%d", aspDM.mpiPrc().nProcesses());
    secondfineFilename.assign(workspace + "custarfm_" + "TEST_"+nPrcs + ".tif");
    //aspectFilename.assign(workspace + "asp_" +"TEST_"+ nPrcs + ".tif");
    if(writeOpt == mcPRL::PGTDEL_WRITING) {
		if(!aspDM.createLayerPGTIOL(psecondfineLyr->name(), secondfineFilename.c_str(), NULL) ) {
        aspDM.mpiPrc().abort();
        return -1;
      }
    }
    else {
      if(!aspDM.createLayerGDAL(psecondfineLyr->name(), secondfineFilename.c_str(), "GTiff", NULL) ) {
        aspDM.mpiPrc().abort();
        return -1;
      }
    }
  }
  mcPRL::pCuf pf;
  pf=&mcPRL::Transition::cuFocalMutiOperator<cuSTARFM>;
  //  pf=&mcPRL::Transition::cuFocalOperator<short,float, float,SlopeMPI>;
 // InitCUDA(aspDM.mpiPrc().id()); 
  aspDM.mpiPrc().sync();
  if(aspDM.mpiPrc().isMaster()) {
    timeCreate = MPI_Wtime();
  }
  if(taskFarming) {
    // Initialize task farming
    //cout << aspDM.mpiPrc().id() << ": initializing task farm...." << endl;
    int nSubspcs2Map = withWriter ? 2*(aspDM.mpiPrc().nProcesses()-2) : 2*(aspDM.mpiPrc().nProcesses()-1);
    if(!aspDM.initTaskFarm(aspTrans, mcPRL::CYLC_MAP, nSubspcs2Map, readOpt)) {
      return -1;
    }

    aspDM.mpiPrc().sync();
    if(aspDM.mpiPrc().isMaster()) {
      timeRead = MPI_Wtime();
    }
	
    // Task farming
    //cout << aspDM.mpiPrc().id() << ": task farming...." << endl;
	if(aspDM.evaluate_TF(mcPRL::EVAL_ALL, aspTrans, readOpt, writeOpt,NULL, false, false) != mcPRL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  else {
    //cout << aspDM.mpiPrc().id() << ": initializing static tasking...." << endl;
    if(!aspDM.initStaticTask(aspTrans, mcPRL::CYLC_MAP, readOpt)) {
      return -1;
    }

    aspDM.mpiPrc().sync();
    if(aspDM.mpiPrc().isMaster()) {
      timeRead = MPI_Wtime();
    }

    //cout << aspDM.mpiPrc().id() << ": static tasking...." << endl;
	if(aspDM.evaluate_ST(mcPRL::EVAL_ALL, aspTrans, writeOpt,true, pf,false) != mcPRL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  
  //cudaMemcpy(pAtest_host,d_data,sizeof(int)*20,cudaMemcpyDeviceToHost);
 // cout<<pAtest_host[10]<<endl;
  // Save the output data
  aspDM.closeDatasets();

  // Record the end time, log computing time
  aspDM.mpiPrc().sync();

  if(aspDM.mpiPrc().isMaster()) {
    //cout << "-------- Completed --------" << endl;
    timeEnd = MPI_Wtime();

    ofstream timeFile;
    string timeFilename(workspace + "asp_time.csv");
    timeFile.open(timeFilename.c_str(), ios::app);
    if(!timeFile) {
      cerr << "Error: unable to open the time log file" << endl;
    }
	timeFile << firstfineFilename << "," \
		<< firstcorseFilename << "," \
		<< secondcorseFilename<< "," \
		<< secondfineFilename<< "," \
        << aspDM.mpiPrc().nProcesses() << "," \
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

    //cout << aspDM.ownershipMap() << endl;
  }

  // Finalize MPI
  aspDM.finalizeMPI();
  return 0;
}
