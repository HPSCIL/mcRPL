//////#include <iostream>
//////#include <fstream>
//////#include <sstream>
//////
//////#include "CuLayer.h"
//////
//////#include "CuEnvControl.h"
//////#include "LocalOperator.h"
//#include "FocalOperator.h"
//////#include "NeighborhoodSlope.h"
//////#include <time.h>
//////
//////#include "cputest.h"
//////
//////using namespace std;
//////using namespace CuPRL;
//////
//////int main(int argc, char* argv[])
//////{
//////
//////	/*
//////	int testdata[36] = { 1, 1, 1, 1, 1, 1,
//////		1, 3, 3, 2, 1, 10,
//////		1, 1, 3, 2, 2, 2,
//////		1, 2, 2, 2, 2, 2,
//////		1, 1, 1, 2, 2, 2,
//////		1, 1, 1, 1, 1, 2 };
//////
//////
//////	CuLayer<int>testlayer1(testdata, 6, 6);
//////	testlayer1.setCellHeight(1);
//////	testlayer1.setCellWidth(1);
//////	testlayer1.setNoDataValue(10);
//////
//////	printLayer(testlayer1);
//////	*/
//////	//CuEnvControl::setBlockDim(16, 16);
//////
//////	CuLayer<int>testlayer1;
//////	//testlayer1.Read("D:\\cuda\\shikong\\testdata\\M_2002_01_04.tif");
//////	
//////	testlayer1.Read("D:\\cuda\\shikong\\testdata\\nd_dem.tif");
//////	
//////	/*int max = 0;
//////	for (int i = 0; i < testlayer1.getHeight(); i++)
//////	{
//////		for (int j = 0; j < testlayer1.getWidth(); j++)
//////		{
//////			if (testlayer1[i*testlayer1.getWidth() + j] > max&&testlayer1[i*testlayer1[i*testlayer1.getWidth()+j]]!=10000)
//////			{
//////				max = testlayer1[i*testlayer1.getWidth() + j];
//////			}
//////		}
//////	}
//////	cout << "max=" << max << endl;
//////	*/
//////	//cout << testlayer1[0] << endl;
//////
//////
//////
//////	clock_t t1, t2;
//////
//////	t1 = clock();
//////
//////	//NeighborhoodSlope neiSlope;
//////
//////	//CuLayer<double>testgpulayer = cuFocalOperatorFn<int, double, int, SlopeCal>(testlayer1, &neiSlope);
//////
//////
//////	//NeighborhoodRect<int>neiRect(3, 3);
//////	//CuLayer<float>testSumLayer = focalStatisticsMean<int, float, int>(testlayer1, &neiRect, NOUSE, IGNORE);
//////	////CuLayer<int>testSumLayer = testlayer1;
//////	//CuLayer<float>testSumLayer = focalStatisticsMean<int, float, int>(testlayer1, &neiRect, NOUSE, IGNORE);
//////	NeighborhoodSlope neiSlope;
//////CuLayer<double>testgpulayer = cuFocalOperatorFn<int, double, int, SlopeCal>(testlayer1, &neiSlope);
//////	t2 = clock();
//////
//////	cout << t2 - t1 << endl;
//////
//////	t1 = clock();
//////	
//////	cudaDeviceProp device_prop;
//////        /* cudaGetDeviceProperties: 获取指定的GPU设备属性相关信息 */
//////        cudaGetDeviceProperties(&device_prop, 0);
//////		fprintf(stdout, "设备上多处理器的数量: %d\n", device_prop.multiProcessorCount);
//////	CuLayer<double>testcpulayer = CPUSlopeCal(testlayer1, &neiSlope);
//////
//////	t2 = clock();
//////
//////	cout << t2 - t1 << endl;
//////	/*
//////	if (compareLayer(testcpulayer, testgpulayer) == false)
//////	{
//////		cout << "result error" << endl;
//////	}
//////	else
//////	{
//////		cout << "result right" << endl;
//////	}
//////	*/
//////	/*
//////	int t = 58 + 3251 * testlayer1.getWidth();
//////
//////	for (int i = -1; i < 1; i++)
//////	{
//////		for (int j = -1; j <= 1; j++)
//////		{
//////			cout << testlayer1[t + i*testlayer1.getWidth() + j] << " ";
//////		}
////		cout << endl;
////	}
////	*/
////	testgpulayer.Write("D:\\cuda\\shikong\\testdata\\cuSlope.tif");
////
////	system("pause");
////	return 0;
////}
////
//
//
//
//#include "prpl-dataManager.h"
//#include "aspectTrans.h"
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <cstring>
//#include <cstdio>
//#include"prplcuda.h"
////#include"FocalOperatorDevice.h"
//#include <cstdlib>
//using namespace std;
//
//int main(int argc, char *argv[]) {
//  const string usage("usage: pAspect workspace input-demFilename num-row-subspaces num-col-subspaces task-farming(1/0) io-option(0/1/2/3/4) with-writer(1/0)");
//
//  // Declare a DataManager and initialize MPI
// // bool withWriter = (bool)atoi(argv[argc-1]);
//  bool withWriter = 0;
//  pRPL::DataManager aspDM;
//  if(!aspDM.initMPI(MPI_COMM_WORLD, withWriter)) {
//    cerr << "Error: unable to initialize MPI" << endl;
//    return -1;
//  }
//
//  // Handle arguments
//  //if(argc != 8) {
//  //  if(aspDM.mpiPrc().isMaster()) {
//  //    cout << usage << endl;
//  //  }
//  //  aspDM.finalizeMPI();
//  //  return -1;
//  //}
//  
//  string workspace, demFilename, slopeFilename, aspectFilename;
//  //workspace.assign(argv[1]);
//  //demFilename.assign(workspace + argv[2]);
//  //int nRowSubspcs = atoi(argv[3]);
//  //int nColSubspcs = atoi(argv[4]);
//  //bool taskFarming = (bool)atoi(argv[5]);
//  //int ioOption = atoi(argv[6]);
//  workspace.assign("D:\\cuda\\shikong\\testdata\\");
//  demFilename.assign("D:\\cuda\\shikong\\testdata\\nd_dem.tif");
//  int nRowSubspcs =2;
//  int nColSubspcs = 2;
//  bool taskFarming =0;
//  int ioOption =3;
//  pRPL::ReadingOption readOpt;
//  pRPL::WritingOption writeOpt;
//  string sReadOpt, sWriteOpt;
//  switch(ioOption) {
//    case 0:
//      readOpt = pRPL::CENTDEL_READING;
//      writeOpt = pRPL::NO_WRITING;
//      sReadOpt = "CENTDEL_READING";
//      sWriteOpt = "NO_WRITING";
//      break;
//    case 1:
//      readOpt = pRPL::PARA_READING;
//      writeOpt = pRPL::NO_WRITING;
//      sReadOpt = "PARA_READING";
//      sWriteOpt = "NO_WRITING";
//      break;
//    case 2:
//      readOpt = pRPL::PGT_READING;
//      writeOpt = pRPL::NO_WRITING;
//      sReadOpt = "PGT_READING";
//      sWriteOpt = "NO_WRITING";
//      break;
//    case 3:
//      readOpt = pRPL::CENTDEL_READING;
//      writeOpt = pRPL::CENTDEL_WRITING;
//      sReadOpt = "CENTDEL_READING";
//      sWriteOpt = "CENTDEL_WRITING";
//      break;
//    case 4:
//      readOpt = pRPL::PARA_READING;
//      writeOpt = pRPL::PARADEL_WRITING;
//      sReadOpt = "PARA_READING";
//      sWriteOpt = "PARADEL_WRITING";
//      break;
//    case 5:
//      readOpt = pRPL::PGT_READING;
//      writeOpt = pRPL::PGTDEL_WRITING;
//      sReadOpt = "PGT_READING";
//      sWriteOpt = "PGTDEL_WRITING";
//      break;
//    default:
//      cerr << "Error: invalid ioOption (" << ioOption << ")" << endl;
//      return -1;
//  }
//
//  // Record the start time
//  double timeStart, timeInit, timeCreate, timeRead, timeEnd;
//  aspDM.mpiPrc().sync();
//  if(aspDM.mpiPrc().isMaster()) {
//    //cout << "-------- Start --------" << endl;
//    timeStart = MPI_Wtime();
//  }
//
//  // Add Layers to the DataManager
//  pRPL::Layer *pDemLyr = NULL;
//  if(readOpt == pRPL::PGT_READING) {
//    pDemLyr = aspDM.addLayerByPGTIOL("DEM", demFilename.c_str(), 1, true);
//  }
//  else {
//    pDemLyr = aspDM.addLayerByGDAL("DEM", demFilename.c_str(), 1, true);
//  }
//  const pRPL::SpaceDims &glbDims = *(pDemLyr->glbDims());
//  const pRPL::CellspaceGeoinfo *pGlbGeoinfo = pDemLyr->glbGeoinfo();
//  long tileSize = pDemLyr->tileSize();
//  
//  pRPL::Layer *pSlpLyr = aspDM.addLayer("SLOPE");
// pSlpLyr->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo, tileSize);
//
//  pRPL::Layer *pAspLyr = aspDM.addLayer("ASP");
//  pAspLyr->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo, tileSize);
//  
//  // Add a 3X3 Neighborhood to the DataManager
//  pRPL::Neighborhood* pNbrhd3x3 = aspDM.addNbrhd("Moore3x3");
//  pNbrhd3x3->initMoore(3, 1.0, pRPL::CUSTOM_VIRTUAL_EDGES, 0);
//
//  // Declare a Transition
//  AspectTransition aspTrans;
//  aspTrans.scale(1.0);
//  aspTrans.setNbrhdByName(pNbrhd3x3->name());
//  aspTrans.addInputLyr(pDemLyr->name(), false);
//  aspTrans.addOutputLyr(pSlpLyr->name(), false);
//  aspTrans.addOutputLyr(pAspLyr->name(), true);
//  
//  // Decompose the Layers
//  //cout << aspDM.mpiPrc().id() << ": decomposing Cellspaces...." << endl;
//  if(!aspDM.dcmpLayers(aspTrans, nRowSubspcs, nColSubspcs)) {
//    aspDM.mpiPrc().abort();
//    return -1;
//  }
//
//  aspDM.mpiPrc().sync();
//  if(aspDM.mpiPrc().isMaster()) {
//    timeInit = MPI_Wtime();
//  }
//
//  // Create the output datasets
//  if(writeOpt != pRPL::NO_WRITING) {
//    char nPrcs[10]; sprintf(nPrcs, "%d", aspDM.mpiPrc().nProcesses());
//    slopeFilename.assign(workspace + "slp_" + nPrcs + ".tif");
//    aspectFilename.assign(workspace + "asp_" + nPrcs + ".tif");
//    if(writeOpt == pRPL::PGTDEL_WRITING) {
//      if(!aspDM.createLayerPGTIOL(pSlpLyr->name(), slopeFilename.c_str(), NULL) ||
//         !aspDM.createLayerPGTIOL(pAspLyr->name(), aspectFilename.c_str(), NULL)) {
//        aspDM.mpiPrc().abort();
//        return -1;
//      }
//    }
//    else {
//      if(!aspDM.createLayerGDAL(pSlpLyr->name(), slopeFilename.c_str(), "GTiff", NULL) ||
//         !aspDM.createLayerGDAL(pAspLyr->name(), aspectFilename.c_str(), "GTiff", NULL)) {
//        aspDM.mpiPrc().abort();
//        return -1;
//      }
//    }
//  }
//  pRPL::pCuf pf;
//  pf=&pRPL::Transition::cuFocalOperator<short,float,SlopeMPI>;
//  //  pf=&pRPL::Transition::cuFocalOperator<short,float, float,SlopeMPI>;
// // InitCUDA(aspDM.mpiPrc().id()); 
//  aspDM.mpiPrc().sync();
//  if(aspDM.mpiPrc().isMaster()) {
//    timeCreate = MPI_Wtime();
//  }
//  if(taskFarming) {
//    // Initialize task farming
//    //cout << aspDM.mpiPrc().id() << ": initializing task farm...." << endl;
//    int nSubspcs2Map = withWriter ? 2*(aspDM.mpiPrc().nProcesses()-2) : 2*(aspDM.mpiPrc().nProcesses()-1);
//    if(!aspDM.initTaskFarm(aspTrans, pRPL::CYLC_MAP, nSubspcs2Map, readOpt)) {
//      return -1;
//    }
//
//    aspDM.mpiPrc().sync();
//    if(aspDM.mpiPrc().isMaster()) {
//      timeRead = MPI_Wtime();
//    }
//	
//    // Task farming
//    //cout << aspDM.mpiPrc().id() << ": task farming...." << endl;
//	if(aspDM.evaluate_TF(pRPL::EVAL_ALL, aspTrans, readOpt, writeOpt,pf, false, false) != pRPL::EVAL_SUCCEEDED) {
//      return -1;
//    }
//  }
//  else {
//    //cout << aspDM.mpiPrc().id() << ": initializing static tasking...." << endl;
//    if(!aspDM.initStaticTask(aspTrans, pRPL::CYLC_MAP, readOpt)) {
//      return -1;
//    }
//
//    aspDM.mpiPrc().sync();
//    if(aspDM.mpiPrc().isMaster()) {
//      timeRead = MPI_Wtime();
//    }
//
//    //cout << aspDM.mpiPrc().id() << ": static tasking...." << endl;
//	if(aspDM.evaluate_ST(pRPL::EVAL_ALL, aspTrans, writeOpt, pf,false) != pRPL::EVAL_SUCCEEDED) {
//      return -1;
//    }
//  }
//
//  // Save the output data
//  aspDM.closeDatasets();
//
//  // Record the end time, log computing time
//  aspDM.mpiPrc().sync();
//
//  if(aspDM.mpiPrc().isMaster()) {
//    //cout << "-------- Completed --------" << endl;
//    timeEnd = MPI_Wtime();
//
//    ofstream timeFile;
//    string timeFilename(workspace + "asp_time.csv");
//    timeFile.open(timeFilename.c_str(), ios::app);
//    if(!timeFile) {
//      cerr << "Error: unable to open the time log file" << endl;
//    }
//    timeFile << demFilename << "," \
//        << slopeFilename << "," \
//        << aspectFilename << "," \
//        << aspDM.mpiPrc().nProcesses() << "," \
//        << (withWriter?"WITH_WRITER":"NO_WRITER") << "," \
//        << nRowSubspcs << "," \
//        << nColSubspcs << "," \
//        << (taskFarming?"TF":"ST") << "," \
//        << sReadOpt << "," \
//        << sWriteOpt << "," \
//        << timeInit - timeStart << "," \
//        << timeCreate - timeInit << "," \
//        << timeRead - timeCreate << "," \
//        << timeEnd - timeRead << "," \
//        << timeEnd - timeStart << "," \
//        << endl;
//    timeFile.close();
//
//    //cout << aspDM.ownershipMap() << endl;
//  }
//
//  // Finalize MPI
//  aspDM.finalizeMPI();
//  return 0;
// /* int locId,data[100], tag=8888;  
//    MPI_Status status;  
//    MPI_Init(&argc, &argv) ;  
//    MPI_Comm_rank(MPI_COMM_WORLD, &locId) ;  
//    if(locId == 0) {  
//        MPI_Request events;  
//        MPI_Isend(data, 100, MPI_INT, 1, tag , MPI_COMM_WORLD, &events) ;  
//        MPI_Wait(&events, &status) ;  
//    }  
//    if(locId == 1) {  
//        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);  
//        if (status.MPI_SOURCE==0)  
//            MPI_Recv(data, 100, MPI_INT, 0, tag, MPI_COMM_WORLD, &status) ;  
//    }  
//    MPI_Finalize() ;  
//}*/
// // system("pause");
//}