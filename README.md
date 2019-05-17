# mcPRL
With the development of GPGPU, GPU has natural support and efficient performance for parallel raster computing. Single GPU can achieve good acceleration. However, most of the existing raster parallel computing libraries implement multi-process parallelism based on MPI. A small number of raster parallel computing libraries based on GPU multi-threaded model can not support multi-GPU parallel computing.
MPI+CUDA parallel raster Library (mcPRL) is a C++ programming framework based on MPI+CUDA. It provides an easy-to-use interface to call multiple GPUs to simultaneously calculate parallel raster/image process. Using this framework, users can easily write efficient parallel algorithms for multi-GPU computing without knowing much about parallel computing.

<b>1. To Compile</b> <br>
===============================
Note: this version is not a final release, and some components are still under testsing. The program has been tested on a cluster with four Linux computing nodes (Centos7.0) and eight GPUs, compiled using g++ 4.8.5, OpenMPI 2.1.1,CUDA9.0,GDAL 1.9, and LibTIFF 4.0.9. The makefile (i.e., make_pAspect) will compile a demonstration program, pAspect, which is able to calculate aspect and slope from DEM data in parallel. <br>
(1) Before compiling, make sure MPI, GDAL,CUDA and LibTIFF libraries have been installed. <br>
(2) Open <b>make_ mcSTARFM</b> and modify the lines that specify the locations of libraries. <br>
(3) Type 'make -f make_mcSTARFM depend'.<br>
(4)  Type 'make -f make_mcSTARFM to compile. <br>
After successful compilation, an executable file named <b>mcSTARFM</b> will be generated.

<b>2. Key features of mcRPL</b>
===============================
<br>Supports a wide range of CUDA-enabled GPUs (https://developer.nvidia.com/cuda-gpus)
<br>Supports a wide range of image formats (see http://gdal.org/formats_list.html)
<br>Support multi-layer input of different data types
<br>Supporting arbitrary neighborhoods
<br>Adaptive Cluster GPU Environment，Allocate appropriate GPUs to processes.
<br>Adaptive cyclic task assignment to achieve better load balance

<b>3. To Run</b>
===============================
## <b>3.1 Usage:</b><br>
mpirun -np \<num_proc\> -machinefile \<machinefile\> mcSTARFM \<workspace\> \<num-row-subspaces\> \<num-col-subspaces\> \<task-farming(1/0)\> \<io-option(0/1/2/3/4/5)\> \<with-writer(1/0)\>  <br>
<b>machinefile</b>:Configuration files consisting of the pairs of node names and number of processes. The number of processes of a node corresponds to the number of GPUs available. If not,it can be used but there will be multiple processes using the same GPU or incomplete use of GPU computing resources, which will affect cluster efficiency.
<b>workspace</b>: the directory where the input file is located and the output files will be written. <br>
<b>input-demFilename</b>: the input file in the GeoTIFF format, usually the DEM data. <br>
<b>num-row-subspaces</b>: the number of sub-domains in the Y axis, for domain decomposition. If num-row-subspaces > 1 and num-col-subspaces = 1, the domain is decomposed as row-wise; if num-row-subspaces = 1 and num-col-subspaces > 1, the domain is decomposed as column-wise; if both > 1, the domain is decomposed as block-wise. <br>
<b>num-col-subspaces</b>: the number of sub-domains in the X axis, for domain decomposition. <br>
<b>task-farming</b>: load-balancing option, either 0 or 1. if 0, static load-balancing; if 1, task farming. <br>
<b>io-option</b>: I/O option, ranges within [0, 5]. Option 0: GDAL-based centralized reading, no writing; Option 1: GDAL-based parallel reading, no writing; Option 2: pGTIOL-based parallel reading, no writing; Option 3: GDAL-based centralized reading and writing; Option 4: GDAL-based parallel reading and pseudo parallel writing; Option 5: pGTIOL-based parallel reading and parallel writing. <br>
<b>with-writer</b>: an option that specify whether a writer process will be used. If 0, no writer; if 1, use a writer. <br>

## <b>3.2 Example:</b><br>
mpirun -np 6 -machinefile machinefile  mcSTARFM  ./ 6 1 0 2 0 <br>

<b>machinefile</b>:
<br>compute-0-0 slots=2
<br>compute-0-0 slots=2
<br>hpscil slots=2
<br>    Note: The computational performance of mcPRL largely depends on the GPUs The more powerful is the GPUs the better performance. The more GPUs the better performance.

## <b>3.3 An example of programming with mcRPL:</b><br>
To illustrate how to use mcRPL, a simple example is given in the following section.
<br>In this example，there are two input layers and two output layers. The first output layer is the focal sum of the first input layer with a 3×3 neighborhood. The second output layer is the focal sum of the second input layer with a 3×3 neighborhood.
+ Implementing this algorithm includes the following three steps：
   - Fisrt，You have to implement custom device functions based on the template of neighborhood computation.
 ```
   class testFocal
{
public:
	__device__ void operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo)
	{
		int nbrsize = nbrinfo.nbrsize;
		int* nbrcood = nbrinfo.nbrcood;
		int nwidth = rasterinfo.width;
		int nheight = rasterinfo.height;
		int nrow=nIndex/nwidth;
		int ncolumn=nIndex%nwidth;
		int nbrIndex=nIndex;
		short sum1=0;
		int sum2=0;
		long dim=nwidth*nheight;
		for(int i = 0; i < nbrsize; i++)
		{
			nbrIndex+=nbrcood[i * 2] + nbrcood[i * 2 + 1] * nwidth;
			sum1+=cuGetDataAs<int>(nbrIndex,0,focalIn,DataInType[0],dim);
			sum2+=cuGetDataAs<int>(nbrIndex,1,focalIn,DataInType[1],dim);
		}
		cuupdateCellAs<short>(nIndex,0,focalOut,DataOutType[0],dim,sum1);
		cuupdateCellAs<int>(nIndex,1,focalOut,DataOutType[1],dim,sum2);
	}
};
```
   - second Specify a neighborhood.In this example, Moore neighborhood which is a 3*3 neighborhood is used.
   
```
Moore neighborhood ：MooreNbrLocs[16] = {-1, 0, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, 0, -1, -1, -1};
```
   - In the end，you must write a main function：
```
int main()
{
bool withWriter = 0;
  pRPL::DataManager testDM;
  if(!testDM.initMPI(MPI_COMM_WORLD, withWriter)) {
    cerr << "Error: unable to initialize MPI" << endl;
    return -1;
  }
string workspace,firstInName, secondInName, secondcorseFilename,secondfineFilename,secondfineFilename2;
workspace.assign("D:\\data\\newdata\\");
  firstInName.assign("D:\\data\\newdata\\2009-329-flaash.dat"); 
  secondInName.assign("D:\\data\\newdata\\MODO9A1.A2009329-0.dat"); 
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
double timeStart, timeInit, timeCreate, timeRead, timeEnd;
  testDM.mpiPrc().sync();
  if(testDM.mpiPrc().isMaster()) {
    //cout << "-------- Start --------" << endl;
    timeStart = MPI_Wtime();
  }
pRPL::Layer *fisrtIn = NULL;
  pRPL::Layer *secondIn = NULL;
  if(readOpt == pRPL::PGT_READING) {
    fisrtIn = testDM.addLayerByPGTIOL("firstfine", firstInName.c_str(), 3, true);
	secondIn = testDM.addLayerByPGTIOL("secondcorse", secondcorseFilename.c_str(), 1, true);
  }
  else {
    fisrtIn = testDM.addLayerByGDAL("firstfine", firstInName.c_str(), 3, true);
	secondIn = testDM.addLayerByGDAL("secondcorse",  secondcorseFilename.c_str(), 1, true);
  }
  const pRPL::SpaceDims &glbDims = *(fisrtIn->glbDims());
  const pRPL::CellspaceGeoinfo *pGlbGeoinfo = fisrtIn->glbGeoinfo();
  long tileSize = fisrtIn->tileSize();
  
  pRPL::Layer *firstOut = testDM.addLayer("secondfine");
  pRPL::Layer *firstOut2 = testDM.addLayer("secondfine2");
 firstOut->initCellspaceInfo(glbDims, typeid( int).name(), sizeof( int), pGlbGeoinfo, tileSize);
 firstOut2->initCellspaceInfo(glbDims, typeid( int).name(), sizeof( int), pGlbGeoinfo, tileSize);

  //pRPL::Layer *pAspLyr = testDM.addLayer("ASP");
 // pAspLyr->initCellspaceInfo(glbDims, typeid(float).name(), sizeof(float), pGlbGeoinfo, tileSize);
  
  // Add a 3X3 Neighborhood to the DataManager
  pRPL::Neighborhood* pNbrhd31x31 = testDM.addNbrhd("Moore31x31");
  pNbrhd31x31->initMoore(3, 1.0, pRPL::CUSTOM_VIRTUAL_EDGES, 0);

  // Declare a Transition
  pRPL::Transition aspTrans;
 // aspTrans.scale(1.0);
  aspTrans.setNbrhdByName(pNbrhd31x31->name());
  //aspTrans.setNbrhd();
  aspTrans.addInputLyr(fisrtIn->name(), false);
  aspTrans.addInputLyr(secondIn->name(), false);
  aspTrans.addOutputLyr(firstOut->name(),true);
  aspTrans.addOutputLyr(firstOut2->name(),true);
  //aspTrans.addOutputLyr(pAspLyr->name(), true);
  
  // Decompose the Layers
  cout << testDM.mpiPrc().id() << ": decomposing Cellspaces...." << endl;
  if(!testDM.dcmpLayers(aspTrans, nRowSubspcs, nColSubspcs)) {
    testDM.mpiPrc().abort();
    return -1;
  }
  
  cout << testDM.mpiPrc().id() << ": decomposed Cellspaces...." << endl;
  testDM.mpiPrc().sync();
  if(testDM.mpiPrc().isMaster()) {
    timeInit = MPI_Wtime();
  }
// Create the output datasets
  if(writeOpt != pRPL::NO_WRITING) {
    char nPrcs[10]; sprintf(nPrcs, "%d", testDM.mpiPrc().nProcesses());
    secondfineFilename.assign(workspace + "test_" + "TEST_"+nPrcs + ".tif");
	secondfineFilename2.assign(workspace + "test2_" + "TEST_"+nPrcs + ".tif");
    //aspectFilename.assign(workspace + "asp_" +"TEST_"+ nPrcs + ".tif");
    if(writeOpt == pRPL::PGTDEL_WRITING) {
		if(!testDM.createLayerPGTIOL(firstOut->name(), secondfineFilename.c_str(), NULL) ) {
        testDM.mpiPrc().abort();
        return -1;
      }
		if(!testDM.createLayerPGTIOL(firstOut2->name(), secondfineFilename2.c_str(), NULL) ) {
        testDM.mpiPrc().abort();
        return -1;
      }
    }
    else {
      if(!testDM.createLayerGDAL(firstOut->name(), secondfineFilename.c_str(), "GTiff", NULL) ) {
        testDM.mpiPrc().abort();
        return -1;
      }
	  if(!testDM.createLayerGDAL(firstOut2->name(), secondfineFilename2.c_str(), "GTiff", NULL) ) {
        testDM.mpiPrc().abort();
        return -1;
      }
    }
  }
  pRPL::pCuf pf;
  pf=&pRPL::Transition::cuFocalMutiOperator<testFocal>;
  //pf=&pRPL::Transition::cuLocalOperator<Copy>;
  //  pf=&pRPL::Transition::cuFocalOperator<short,float, float,SlopeMPI>;
 // InitCUDA(testDM.mpiPrc().id()); 
  testDM.mpiPrc().sync();
  if(testDM.mpiPrc().isMaster()) {
    timeCreate = MPI_Wtime();
  }
  if(taskFarming) {
    // Initialize task farming
    //cout << testDM.mpiPrc().id() << ": initializing task farm...." << endl;
    int nSubspcs2Map = withWriter ? 2*(testDM.mpiPrc().nProcesses()-2) : 2*(testDM.mpiPrc().nProcesses()-1);
    if(!testDM.initTaskFarm(aspTrans, pRPL::CYLC_MAP, nSubspcs2Map, readOpt)) {
      return -1;
    }

    testDM.mpiPrc().sync();
    if(testDM.mpiPrc().isMaster()) {
      timeRead = MPI_Wtime();
    }
	
    // Task farming
    //cout << testDM.mpiPrc().id() << ": task farming...." << endl;
	if(testDM.evaluate_TF(pRPL::EVAL_ALL, aspTrans, readOpt, writeOpt,NULL, false, false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  else {
    //cout << testDM.mpiPrc().id() << ": initializing static tasking...." << endl;
    if(!testDM.initStaticTask(aspTrans, pRPL::CYLC_MAP, readOpt)) {
      return -1;
    }

    testDM.mpiPrc().sync();
    //if(testDM.mpiPrc().isMaster()) {
    //  timeRead = MPI_Wtime();
    //}

    //cout << testDM.mpiPrc().id() << ": static tasking...." << endl;
	if(testDM.evaluate_ST(pRPL::EVAL_ALL, aspTrans, writeOpt,true, pf,false) != pRPL::EVAL_SUCCEEDED) {
      return -1;
    }
  }
  
  //cudaMemcpy(pAtest_host,d_data,sizeof(int)*20,cudaMemcpyDeviceToHost);
 // cout<<pAtest_host[10]<<endl;
  // Save the output data
  testDM.closeDatasets();

  // Record the end time, log computing time
  testDM.mpiPrc().sync();

  if(testDM.mpiPrc().isMaster()) {
    //cout << "-------- Completed --------" << endl;
    timeEnd = MPI_Wtime();

    ofstream timeFile;
    string timeFilename(workspace + "test_time.csv");
    timeFile.open(timeFilename.c_str(), ios::app);
    if(!timeFile) {
      cerr << "Error: unable to open the time log file" << endl;
    }
	timeFile << firstInName << "," \
		<< secondInName << "," \
		<< secondcorseFilename<< "," \
		<< secondfineFilename<< "," \
        << testDM.mpiPrc().nProcesses() << "," \
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
    //cout << testDM.ownershipMap() << endl;
  }
  // Finalize MPI
  testDM.finalizeMPI();
  return 0;
}
```


