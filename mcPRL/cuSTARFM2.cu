#include "prpl-dataManager.h"
#include <iostream>
#include <fstream>
#include"prplcuda.h"
#include "FocalOperatorDevice.h"
#include"aspectTrans.h"
#include"cuSTARFM.h"
using namespace std;
int main()
{
 // const string usage("usage: pAspect workspace input-demFilename num-row-subspaces num-col-subspaces task-farming(1/0) io-option(0/1/2/3/4) with-writer(1/0)");

  // Declare a DataManager and initialize MPI
 // bool withWriter = (bool)atoi(argv[argc-1]);
  bool withWriter = 0;
  pRPL::DataManager aspDM;
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
 // cudaSetDevice(0);
  string workspace,firstfineFilename, firstcorseFilename, secondcorseFilename,secondfineFilename,secondfineFilename2;
  //workspace.assign(argv[1]);
  //demFilename.assign(workspace + argv[2]);
  //int nRowSubspcs = atoi(argv[3]);
  //int nColSubspcs = atoi(argv[4]);
  //bool taskFarming = (bool)atoi(argv[5]);
  //int ioOption = atoi(argv[6]);
 // workspace.assign("D:\\data\\newdata");
 // firstfineFilename.assign("D:\\data\\newdata\\L_2002_01_04.tif"); 
  //firstcorseFilename.assign("D:\\data\\newdata\\M_2002_01_04.tif"); 
  //secondcorseFilename.assign("D:\\data\\newdata\\M_2002_02_12.tif");
   workspace.assign("D:\\data\\newdata\\");
  firstfineFilename.assign("D:\\data\\newdata\\2009-329-flaash.dat"); 
  firstcorseFilename.assign("D:\\data\\newdata\\MODO9A1.A2009329-0.dat"); 
  secondcorseFilename.assign("D:\\data\\newdata\\MODO9A1.A2009249.dat");
  int nRowSubspcs =3;
  int nColSubspcs = 1;
  bool taskFarming =0;
  int ioOption =3;
  pRPL::ReadingOption readOpt;
  pRPL::WritingOption writeOpt;
  string sReadOpt, sWriteOpt;
  switch(ioOption) {
    case 0:
      readOpt = pRPL::CENTDEL_READING;
      writeOpt = pRPL::NO_WRITING;
      sReadOpt = "CENTDEL_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 1:
      readOpt = pRPL::PARA_READING;
      writeOpt = pRPL::NO_WRITING;
      sReadOpt = "PARA_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 2:
      readOpt = pRPL::PGT_READING;
      writeOpt = pRPL::NO_WRITING;
      sReadOpt = "PGT_READING";
      sWriteOpt = "NO_WRITING";
      break;
    case 3:
      readOpt = pRPL::CENTDEL_READING;
      writeOpt = pRPL::CENTDEL_WRITING;
      sReadOpt = "CENTDEL_READING";
      sWriteOpt = "CENTDEL_WRITING";
      break;
    case 4:
      readOpt = pRPL::PARA_READING;
      writeOpt = pRPL::PARADEL_WRITING;
      sReadOpt = "PARA_READING";
      sWriteOpt = "PARADEL_WRITING";
      break;
    case 5:
      readOpt = pRPL::PGT_READING;
      writeOpt = pRPL::PGTDEL_WRITING;
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
  pRPL::Layer *pfirstfineLyr = NULL;
  pRPL::Layer *pfirstcorseLyr = NULL;
   pRPL::Layer *psecondcorseLyr = NULL;
  if(readOpt == pRPL::PGT_READING) {
    pfirstfineLyr = aspDM.addLayerByPGTIOL("firstfine", firstfineFilename.c_str(), 3, true);
	pfirstcorseLyr = aspDM.addLayerByPGTIOL("firstcorse", firstcorseFilename.c_str(), 1, true);
	psecondcorseLyr = aspDM.addLayerByPGTIOL("secondcorse", secondcorseFilename.c_str(), 1, true);
  }
  else {
    pfirstfineLyr = aspDM.addLayerByGDAL("firstfine", firstfineFilename.c_str(), 3, true);
	pfirstcorseLyr = aspDM.addLayerByGDAL("firstcorse", firstcorseFilename.c_str(), 1, true);
	psecondcorseLyr = aspDM.addLayerByGDAL("secondcorse",  secondcorseFilename.c_str(), 1, true);
  }
  const pRPL::SpaceDims &glbDims = *(pfirstfineLyr->glbDims());
  const pRPL::CellspaceGeoinfo *pGlbGeoinfo = pfirstfineLyr->glbGeoinfo();
  long tileSize = pfirstfineLyr->tileSize();
  
  pRPL::Layer *psecondfineLyr = aspDM.addLayer("secondfine");
  pRPL::Layer *psecondfineLyr2 = aspDM.addLayer("secondfine2");
 psecondfineLyr->initCellspaceInfo(glbDims, typeid( int).name(), sizeof( int), pGlbGeoinfo, tileSize);
 psecondfineLyr2->initCellspaceInfo(glbDims, typeid( int).name(), sizeof( int), pGlbGeoinfo, tileSize);

  //pRPL::Layer *pAspLyr = aspDM.addLayer("ASP");
 // pAspLyr->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo, tileSize);
  
  // Add a 3X3 Neighborhood to the DataManager
  pRPL::Neighborhood* pNbrhd31x31 = aspDM.addNbrhd("Moore31x31");
  pNbrhd31x31->initMoore(3, 1.0, pRPL::CUSTOM_VIRTUAL_EDGES, 0);

  // Declare a Transition
  pRPL::Transition aspTrans;
 // aspTrans.scale(1.0);
  aspTrans.setNbrhdByName(pNbrhd31x31->name());
  //aspTrans.setNbrhd();
  aspTrans.addInputLyr(pfirstfineLyr->name(), false);
  aspTrans.addInputLyr(psecondcorseLyr->name(), false);
  aspTrans.addInputLyr(pfirstcorseLyr->name(), false);
  aspTrans.addOutputLyr(psecondfineLyr->name(),true);
  //aspTrans.addOutputLyr(psecondfineLyr2->name(),true);
  //aspTrans.addOutputLyr(pAspLyr->name(), true);
  
  // Decompose the Layers
  cout << aspDM.mpiPrc().id() << ": decomposing Cellspaces...." << endl;
  if(!aspDM.dcmpLayers(aspTrans, nRowSubspcs, nColSubspcs)) {
    aspDM.mpiPrc().abort();
    return -1;
  }
  
  cout << aspDM.mpiPrc().id() << ": decomposed Cellspaces...." << endl;
  aspDM.mpiPrc().sync();
  if(aspDM.mpiPrc().isMaster()) {
    timeInit = MPI_Wtime();
  }

  // Create the output datasets
  if(writeOpt != pRPL::NO_WRITING) {
    char nPrcs[10]; sprintf(nPrcs, "%d", aspDM.mpiPrc().nProcesses());
    secondfineFilename.assign(workspace + "test_" + "TEST_"+nPrcs + ".tif");
	secondfineFilename2.assign(workspace + "test2_" + "TEST_"+nPrcs + ".tif");
    //aspectFilename.assign(workspace + "asp_" +"TEST_"+ nPrcs + ".tif");
    if(writeOpt == pRPL::PGTDEL_WRITING) {
		if(!aspDM.createLayerPGTIOL(psecondfineLyr->name(), secondfineFilename.c_str(), NULL) ) {
        aspDM.mpiPrc().abort();
        return -1;
      }
		if(!aspDM.createLayerPGTIOL(psecondfineLyr2->name(), secondfineFilename2.c_str(), NULL) ) {
        aspDM.mpiPrc().abort();
        return -1;
      }
    }
    else {
      if(!aspDM.createLayerGDAL(psecondfineLyr->name(), secondfineFilename.c_str(), "GTiff", NULL) ) {
        aspDM.mpiPrc().abort();
        return -1;
      }
	  if(!aspDM.createLayerGDAL(psecondfineLyr2->name(), secondfineFilename2.c_str(), "GTiff", NULL) ) {
        aspDM.mpiPrc().abort();
        return -1;
      }
    }
  }
  pRPL::pCuf pf;
  pf=&pRPL::Transition::cuFocalMutiOperator<cuSTARFM>;
  //pf=&pRPL::Transition::cuLocalOperator<Copy>;
  //  pf=&pRPL::Transition::cuFocalOperator<short,float, float,SlopeMPI>;
 // InitCUDA(aspDM.mpiPrc().id()); 
  aspDM.mpiPrc().sync();
  if(aspDM.mpiPrc().isMaster()) {
    timeCreate = MPI_Wtime();
  }
  if(taskFarming) {
    // Initialize task farming
    //cout << aspDM.mpiPrc().id() << ": initializing task farm...." << endl;
    int nSubspcs2Map = withWriter ? 2*(aspDM.mpiPrc().nProcesses()-2) : 2*(aspDM.mpiPrc().nProcesses()-1);
    if(!aspDM.initTaskFarm(aspTrans, pRPL::CYLC_MAP, nSubspcs2Map, readOpt)) {
      return -1;
    }

    aspDM.mpiPrc().sync();
    if(aspDM.mpiPrc().isMaster()) {
      timeRead = MPI_Wtime();
    }
	
    // Task farming
    //cout << aspDM.mpiPrc().id() << ": task farming...." << endl;
	if(aspDM.evaluate_TF(pRPL::EVAL_ALL, aspTrans, readOpt, writeOpt,NULL, false, false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  else {
    //cout << aspDM.mpiPrc().id() << ": initializing static tasking...." << endl;
    if(!aspDM.initStaticTask(aspTrans, pRPL::CYLC_MAP, readOpt)) {
      return -1;
    }

    aspDM.mpiPrc().sync();
    //if(aspDM.mpiPrc().isMaster()) {
    //  timeRead = MPI_Wtime();
    //}

    //cout << aspDM.mpiPrc().id() << ": static tasking...." << endl;
	if(aspDM.evaluate_ST(pRPL::EVAL_ALL, aspTrans, writeOpt,true, pf,false) != pRPL::EVAL_SUCCEEDED) {
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