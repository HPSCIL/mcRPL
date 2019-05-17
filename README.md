# mcPRL
With the development of GPGPU, GPU has natural support and efficient performance for parallel raster computing. Single GPU can achieve good acceleration. However, most of the existing raster parallel computing libraries implement multi-process parallelism based on MPI. A small number of raster parallel computing libraries based on GPU multi-threaded model can not support multi-GPU parallel computing.
MPI+CUDA parallel raster Library (mcPRL) is a C++ programming framework based on MPI+CUDA. It provides an easy-to-use interface to call multiple GPUs to simultaneously calculate parallel raster/image process. Using this framework, users can easily write efficient parallel algorithms for multi-GPU computing without knowing much about parallel computing.

<b>1. To Compile</b> <br>
NNote: this version is not a final release, and some components are still under testsing. The program has been tested on a cluster with four Linux computing nodes (Centos7.0) and eight GPUs, compiled using g++ 4.8.5, OpenMPI 2.1.1,CUDA9.0,GDAL 1.9, and LibTIFF 4.0.9. The makefile (i.e., make_pAspect) will compile a demonstration program, pAspect, which is able to calculate aspect and slope from DEM data in parallel. <br>
(1) Before compiling, make sure MPI, GDAL,CUDA and LibTIFF libraries have been installed. <br>
(2) Open <b>make_ mcSTARFM</b> and modify the lines that specify the locations of libraries. <br>
(3) Type 'make -f make_mcSTARFM depend'.<br>
(4)  Type 'make -f make_mcSTARFM to compile. <br>
After successful compilation, an executable file named <b>mcSTARFM</b> will be generated.

<b>2. Key features of mcRPL</b>
<br>Supports a wide range of CUDA-enabled GPUs (https://developer.nvidia.com/cuda-gpus)
<br>Supports a wide range of image formats (see http://gdal.org/formats_list.html)
<br>Support multi-layer input of different data types
<br>Supporting arbitrary neighborhoods
<br>Adaptive Cluster GPU Environment，Allocate appropriate GPUs to processes.
<br>Adaptive cyclic task assignment to achieve better load balance

<b>3. To Run</b>


<b>2.1 Usage:</b><br>
mpirun -np \<num_proc\> pAspect \<workspace\> \<input-demFilename\> \<num-row-subspaces\> \<num-col-subspaces\> \<task-farming(1/0)\> \<io-option(0/1/2/3/4/5)\> \<with-writer(1/0)\>  <br>
<b>workspace</b>: the directory where the input file is located and the output files will be written. <br>
<b>input-demFilename</b>: the input file in the GeoTIFF format, usually the DEM data. <br>
<b>num-row-subspaces</b>: the number of sub-domains in the Y axis, for domain decomposition. If num-row-subspaces > 1 and num-col-subspaces = 1, the domain is decomposed as row-wise; if num-row-subspaces = 1 and num-col-subspaces > 1, the domain is decomposed as column-wise; if both > 1, the domain is decomposed as block-wise. <br>
<b>num-col-subspaces</b>: the number of sub-domains in the X axis, for domain decomposition. <br>
<b>task-farming</b>: load-balancing option, either 0 or 1. if 0, static load-balancing; if 1, task farming. <br>
<b>io-option</b>: I/O option, ranges within [0, 5]. Option 0: GDAL-based centralized reading, no writing; Option 1: GDAL-based parallel reading, no writing; Option 2: pGTIOL-based parallel reading, no writing; Option 3: GDAL-based centralized reading and writing; Option 4: GDAL-based parallel reading and pseudo parallel writing; Option 5: pGTIOL-based parallel reading and parallel writing. <br>
<b>with-writer</b>: an option that specify whether a writer process will be used. If 0, no writer; if 1, use a writer. <br>

<b>2.2 Example:</b><br>
mpirun -np 8 ./pAspect ./ nd_dem.tif 8 1 0 5 0 <br>
