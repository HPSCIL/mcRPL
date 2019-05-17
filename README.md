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
   - In the end，you must write a main function：

