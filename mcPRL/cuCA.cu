//
#include "prpl-dataManager.h"
#include <iostream>
#include <fstream>
#include"prplcuda.h"
#include "OperatorDevice.h"
//#include"cuLogisticCA.h"
using namespace std;

int main(int argc, char *argv[]) {
  const string usage("usage: pCCA workspace number-of-urbanization-per-year number-of-years output-option(0/1/2) num-row-subspaces num-col-subspaces task-farming(1/0) parallelIO(1/0) with-writer(1/0)");
  
  bool withWriter = (bool)atoi(argv[argc-1]);
  pRPL::DataManager caDM;
  if(!caDM.initMPI(MPI_COMM_WORLD, withWriter)) {
    cerr << "Error: unable to initialize MPI" << endl;
    return -1;
  }
  //cudaSetDevice(0);
 // InitCUDA(caDM.mpiPrc().id());
  // Handle arguments
  if(argc != 10) {
    if(caDM.mpiPrc().isMaster()) {
      cout << usage << endl;
    }
    caDM.finalizeMPI();
    return -1;
  }
  string workspace(argv[1]);
  int nCells2UrbanPerYear = atoi(argv[2]);
  int nSimYears = atoi(argv[3]);
  int opOption = atoi(argv[4]);
  if(opOption < 0 || opOption >2) {
    cerr << "Error: invalid output option (" << opOption << ")" << endl;
    return -1;
  }
  int nRowSubspcs = atoi(argv[5]);
  int nColSubspcs = atoi(argv[6]);
  bool taskFarming = (bool)atoi(argv[7]);
  bool parallelIO = (bool)atoi(argv[8]);
  pRPL::ReadingOption readOpt = parallelIO ? pRPL::PARA_READING : pRPL::CENTDEL_READING;
  pRPL::WritingOption writeOpt;
  if(opOption != 0) {
    writeOpt = parallelIO ? pRPL::PARA_WRITING : pRPL::CENT_WRITING;
  }
  else {
    writeOpt = pRPL::NO_WRITING;
  }
  char nPrcs[10]; sprintf(nPrcs, "%d", caDM.mpiPrc().nProcesses());
  // Record the start time
  caDM.mpiPrc().sync();
  double timeStart;
  if(caDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
    timeStart = MPI_Wtime();
  }

  string urbanFile(workspace + "lcUrban.tif");
  string landuseFile(workspace + "landuse92.tif");
  string elevFile(workspace + "dem30m_nrm.tif");
  string slopeFile(workspace + "slope30m_nrm.tif");
  string dist2CityCtrFile(workspace + "dist2cityCtrs_nrm.tif");
  string dist2TranspFile(workspace + "dist2transp_nrm.tif");
  string excludedFile(workspace + "excluded.tif");
  string glboutfilename;
  string glb2(workspace + "glb_cpu2.tif");
  const char *aFormatName = "GTiff";

  /* ----------------------- Initialize Layers ----------------------- */
  pRPL::Layer *pUrban0 = caDM.addLayerByGDAL("URBAN0", urbanFile.c_str(), 1, true);
  const pRPL::SpaceDims &glbDims = *(pUrban0->glbDims());
  const pRPL::CellspaceGeoinfo *pGlbGeoinfo = pUrban0->glbGeoinfo();

  pRPL::Layer *pLandUse = caDM.addLayerByGDAL(_aInputNames[0], landuseFile.c_str(), 1, true);
  pRPL::Layer *pElev = caDM.addLayerByGDAL(_aInputNames[1], elevFile.c_str(), 1, true);
  pRPL::Layer *pSlope = caDM.addLayerByGDAL(_aInputNames[2], slopeFile.c_str(), 1, true);
  pRPL::Layer *pDist2CityCtr = caDM.addLayerByGDAL(_aInputNames[3], dist2CityCtrFile.c_str(), 1, true);
  pRPL::Layer *pDist2Transp = caDM.addLayerByGDAL(_aInputNames[4], dist2TranspFile.c_str(), 1, true);
  pRPL::Layer *pExcluded = caDM.addLayerByGDAL("EXCLUDED", excludedFile.c_str(), 1, true);
  //pRPL::Layer *pGlbProb2 = caDM.addLayerByGDAL("GLBPROB2",  glb2.c_str(), 1, true);   //201811 15£»
  pRPL::Layer *pGlbProb = caDM.addLayer("GLBPROB");
  pGlbProb->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo);
  pRPL::Layer *pRandNumber=caDM.addLayer("RAND");
  pRPL::Layer *pJointProb = caDM.addLayer("JOINTPROB");
  pRandNumber->initCellspaceInfo(glbDims,typeid(float).name(),sizeof(float), pGlbGeoinfo);
  pJointProb->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo);
  //pRPL::Layer *pGlbProb2 = caDM.addLayer("GLBPROB2");
 // pGlbProb2->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo);
  pRPL::Layer *pDistDecayProb = caDM.addLayer("DISTDECAYPROB");
  pDistDecayProb->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo);

  pRPL::Layer *pUrban1 = caDM.addLayer("URBAN1");
  pUrban1->initCellspaceInfo(glbDims, pUrban0->dataType(), pUrban0->dataSize(), pGlbGeoinfo);

  // Add a 3X3 Neighborhood to the DataManager
  pRPL::Neighborhood* pNbrhd3x3 = caDM.addNbrhd("Moore3x3");
  pNbrhd3x3->initMoore(3, 1.0);
  if(!caDM.dcmpAllLayers(pNbrhd3x3->name(), nRowSubspcs, nColSubspcs)) {
    caDM.mpiPrc().abort();
    return -1;
  }
 // glboutfilename.assign(workspace + "glb_1115" + nPrcs + ".tif");
	//if(!caDM.createLayerGDAL(pGlbProb->name(),glboutfilename.c_str(), "GTiff", NULL)) {
 //       caDM.mpiPrc().abort();
 //       return -1;
 //     }
  /* compute global probability */
    pRPL::pCuf pf;
   pf=&pRPL::Transition::cuFocalMutiOperator<cuglbProb>;
  cout << caDM.mpiPrc().id() << ": computing global probilities... " << endl;
  pRPL::Transition  glbProbTrans;
  glbProbTrans.setNbrhdByName(pNbrhd3x3->name());
  for(int iInLyr = 0; iInLyr <5; iInLyr++) {
    glbProbTrans.addInputLyr(_aInputNames[iInLyr], false);
  }
  glbProbTrans.addOutputLyr(pGlbProb->name(), true);
 //  double timeInitTasking;
  if(taskFarming) {
    int nSubspcs2Map = withWriter ? 2*(caDM.mpiPrc().nProcesses()-2) : 2*(caDM.mpiPrc().nProcesses()-1);
    if(!caDM.initTaskFarm(glbProbTrans, pRPL::CYLC_MAP, nSubspcs2Map, readOpt)) {
      return -1;
    }

    // Task farming
    cout << caDM.mpiPrc().id() << ": task farming...." << endl;
    if(caDM.evaluate_TF(pRPL::EVAL_ALL, glbProbTrans, readOpt, pRPL::NO_WRITING, false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  else {
    cout << caDM.mpiPrc().id() << ": initializing static tasking...." << endl;
    if(!caDM.initStaticTask(glbProbTrans, pRPL::CYLC_MAP, readOpt)) {
      return -1;
    }
 // caDM.mpiPrc().sync();
 // if(caDM.mpiPrc().isMaster()) {
 //   //cout << "-------- Start --------" << endl;
 //  // timeStart = MPI_Wtime();
	//  timeInitTasking=MPI_Wtime();
	//cout<<"initializing static tasking cost time::"<<timeInitTasking-timeStart;
 // }
	//cudaSetDevice(caDM.mpiPrc().id());
    cout << caDM.mpiPrc().id() << ": static tasking...." << endl;
	if(caDM.evaluate_ST(pRPL::EVAL_ALL, glbProbTrans, pRPL::NO_WRITING, true,pf,false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  //caDM.closeDatasets();
  caDM.mpiPrc().sync();
  double timeEndGlb;
  if(caDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
   // timeStart = MPI_Wtime();
	  timeEndGlb=MPI_Wtime();
	cout<<"computing static tasking cost time::"<<timeEndGlb-timeStart<<endl;
  }
 //  caDM.closeDatasets();
  for(int iInLyr = 0; iInLyr < 5; iInLyr++) {
    if(!caDM.rmvLayerByName(_aInputNames[iInLyr], true)) {
      return -1;
    }
  }

  /* ----------------------- GROWTH ----------------------- */
 pRPL::Transition jointProbTrans;
  jointProbTrans.setNbrhdByName(pNbrhd3x3->name());

  pRPL::Transition distDecProbTrans;
  distDecProbTrans.setNbrhdByName(pNbrhd3x3->name());
  distDecProbTrans.addInputLyr(pJointProb->name(), false);
  distDecProbTrans.addOutputLyr(pDistDecayProb->name(), true);

 pRPL::Transition consProbTrans;
  consProbTrans.setNbrhdByName(pNbrhd3x3->name());

  // Read input data
  if(!caDM.beginReadingLayer(pUrban0->name(), readOpt) ||
     !caDM.beginReadingLayer(pExcluded->name(), readOpt)) {
    return -1;
  }
  
  caDM.beginGetRand(pRandNumber->name());
  caDM.finishReadingLayers(readOpt);
  int yearStart = 1992, yearEnd = yearStart + nSimYears;
  //int nCells2UrbanPerYear = 158667; // whole CA
  //int nCells2UrbanPerYear = 16101; // San Diego data
  //int nCells2UrbanPerYear = 25000; // San Diego data
  long nTotCellsConvd = 0;

  for(int year = yearStart+1; year <= yearEnd; year++) {
    if(caDM.mpiPrc().isMaster()) {
      cout << "simulating " << year  << "..." << endl;
    }
    float lclMaxJointProb = 0.0, glbMaxJointProb = 0.0;
    float lclSumDistDecayProb = 0.0, glbSumDistDecayProb = 0.0;
    long lclNumCellsConvd = 0, glbNumCellsConvd = 0;

    jointProbTrans.clearLyrSettings();
    consProbTrans.clearLyrSettings();
    if(year % 2 == 1) {
      jointProbTrans.addInputLyr(pUrban0->name(), false);
      consProbTrans.addInputLyr(pUrban0->name(), false);
      consProbTrans.addOutputLyr(pUrban1->name(), true);
    }
    else {
      jointProbTrans.addInputLyr(pUrban1->name(), false);
      consProbTrans.addInputLyr(pUrban1->name(), false);
      consProbTrans.addOutputLyr(pUrban0->name(), true);
    }
    jointProbTrans.addInputLyr(pExcluded->name(), false);
    jointProbTrans.addInputLyr(pGlbProb->name(), false);
//	jointProbTrans.addInputLyr(pGlbProb2->name(), false);
    jointProbTrans.addOutputLyr(pJointProb->name(), true);
    consProbTrans.addInputLyr(pDistDecayProb->name(), false);
	//#ifdef DEBUG
	consProbTrans.addInputLyr(pRandNumber->name(),false);
//    #endif
    /* compute joint probability */
    resetMax();
	  string jointoutfilename;
	/* jointoutfilename.assign(workspace + "joint_" + nPrcs + ".tif");
	if(!caDM.createLayerGDAL(pJointProb->name(),jointoutfilename.c_str(), "GTiff", NULL)) {
        caDM.mpiPrc().abort();
        return -1;
      }    */                                                        //½¨ÎÄ¼þ
	 pf=&pRPL::Transition::cuFocalMutiOperator<jointProb>;
	 if(caDM.evaluate_ST(pRPL::EVAL_ALL, jointProbTrans, pRPL::NO_WRITING,true, pf,false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    
	 }
//	 	 caDM.closeDatasets();
	lclMaxJointProb = jointMax();
    MPI_Allreduce(&lclMaxJointProb, &glbMaxJointProb, 1, MPI_FLOAT, MPI_MAX, caDM.mpiPrc().comm());
    cout << caDM.mpiPrc().id() << ": glbMaxJointProb " << glbMaxJointProb << endl;
    //caDM.mpiPrc().sync();
	  caDM.mpiPrc().sync();
  double timeEndjoint;
  if(caDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
   // timeStart = MPI_Wtime();
	  timeEndjoint=MPI_Wtime();
	cout<<"computing jointProbTrans cost time::"<<timeEndjoint-timeEndGlb<<endl;;
  }
    /* compute distance decay probability */
   resetSum();
   maxJointProb(glbMaxJointProb);
   /*string distoutfilename;
	 distoutfilename.assign(workspace + "dist_" + nPrcs + ".tif");
	if(!caDM.createLayerGDAL(pDistDecayProb->name(),distoutfilename.c_str(), "GTiff", NULL)) {
        caDM.mpiPrc().abort();
        return -1;
      }*/
	 pf=&pRPL::Transition::cuFocalMutiOperator<DistDecayProb>;
	 if(caDM.evaluate_ST(pRPL::EVAL_ALL, distDecProbTrans, pRPL::NO_WRITING, true,pf,false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
	//  caDM.closeDatasets();
	lclSumDistDecayProb = distDecySum();
    MPI_Allreduce(&lclSumDistDecayProb, &glbSumDistDecayProb, 1, MPI_FLOAT, MPI_SUM, caDM.mpiPrc().comm());
    cout << caDM.mpiPrc().id() << ": glbSumDistDecayProb " << glbSumDistDecayProb << endl;
    //caDM.mpiPrc().sync();
	 caDM.mpiPrc().sync();
  double timeEnddist;
  if(caDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
   // timeStart = MPI_Wtime();
	  timeEnddist=MPI_Wtime();
	cout<<"computing distDecProbTrans cost time::"<< timeEnddist-timeEndjoint<<endl;;
  }
    /* compute stochastic probability and convert */
    resetNumUrbanized();
    sumDistDecayProb(glbSumDistDecayProb);
    convertLimit(nCells2UrbanPerYear);
    pRPL::WritingOption currentWriteOpt = pRPL::NO_WRITING;
    if((opOption == 1 && (year-yearStart)%5 == 0)||
       (opOption == 2 && year == yearEnd)) {
      currentWriteOpt = writeOpt;
      char sYear[20]; sprintf(sYear, "%d", year);

      string outputUrbanFileName(workspace + "urban_" + nPrcs + "_" + sYear + ".tif");
      if(!caDM.createLayerGDAL(consProbTrans.getOutLyrNames().begin()->c_str(),
                               outputUrbanFileName.c_str(), aFormatName)) {
        return -1;
      }
    }
	pf=&pRPL::Transition::cuFocalMutiOperator<constarined>;
	//#ifdef DEBUG
	if(caDM.evaluate_ST(pRPL::EVAL_ALL, consProbTrans,currentWriteOpt, true,pf,false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
	
//	#endif9    lclNumCellsConvd = numUrbanized();
	 lclNumCellsConvd = numUrbanized();
    MPI_Allreduce(&lclNumCellsConvd, &glbNumCellsConvd, 1, MPI_LONG, MPI_SUM, caDM.mpiPrc().comm());
    nTotCellsConvd += glbNumCellsConvd;
    cout << caDM.mpiPrc().id() << ": glbNumCellsConvd " << glbNumCellsConvd << endl;
    //caDM.mpiPrc().sync();

    if(caDM.mpiPrc().isMaster()) {
      cout << nTotCellsConvd << " cells have been urbanized" << endl;
    }
	 caDM.mpiPrc().sync();
  double timeEnd;
  if(caDM.mpiPrc().isMaster()) {
    //cout << "-------- Completed --------" << endl;
    timeEnd = MPI_Wtime();
	cout<<"computing consProbTrans cost time::"<< timeEnd-timeEnddist<<endl;
  }
  } // end -- for(year)
  caDM.closeDatasets();
  // Record the end time, log computing time
  caDM.mpiPrc().sync();
  double timeEnd;
  if(caDM.mpiPrc().isMaster()) {
    //cout << "-------- Completed --------" << endl;
    timeEnd = MPI_Wtime();
    ofstream timeFile;
    string timeFileName(workspace + "time.csv");
    timeFile.open(timeFileName.c_str(), ios::app);
    if(!timeFile) {
      cerr << "Error: unable to open the time log file" << endl;
    }
    timeFile << yearStart << "," \
        << yearEnd << "," \
        << caDM.mpiPrc().nProcesses() << "," \
        << (caDM.mpiPrc().hasWriter() ?"WITH_WRITER":"NO_WRITER") << "," \
        << nRowSubspcs << "," \
        << nColSubspcs << "," \
        << (taskFarming?"TF":"ST") << "," \
        << (parallelIO?"PARA_IO":"CENT_IO") << "," \
        << opOption << "," \
        << timeEnd - timeStart << "," \
        << endl;
    timeFile.close();

    cout << caDM.ownershipMap() << endl;
  }

  caDM.finalizeMPI();
  return 0;
}
