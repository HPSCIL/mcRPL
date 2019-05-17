#include <iostream>
#include <fstream>
#include <sstream>

#include "CuLayer.h"

#include "CuEnvControl.h"
#include "LocalOperator.h"
#include "FocalOperator.h"
#include <time.h>
#include <string>

#include "cputest.h"

using namespace std;
using namespace CuPRL;

#define TIMELOGOUT		//时间文件输出

#define DOUBLETEST		//double型数据测试
//#define INTTEST			//int型数据测试

#define MATHTEST		//局部数学函数测试
#define MATHADD
//#define MATHDIV

//#define RECLASSVALUEUPDATE

//#define FOCALOPERATORTEST
//#define FOCALSUMTEST
//#define FOCALMEANTEST
//#define FOCALMAXIMUMTEST
//#define FOCALRANGETEST


#ifdef INTTEST
typedef int DataType;
#endif
#ifdef DOUBLETEST
typedef double DataType;
#endif



int main(int argc, char* argv[])
{
	const int width = 8000; const int height = 8000;
	//const int width = 10; const int height = 10;

	DataType* numA = NULL;
	DataType* numB = NULL;
	DataType* numC = NULL;

	numA = new DataType[width*height];
	//DataType* numB = new DataType[width*height];
	//DataType* numC = new DataType[width*height];


	

	CuLayer<DataType>layer1;
	CuLayer<DataType>layer2;
	CuLayer<DataType>gpuOutIntLayer;
	CuLayer<DataType>cpuOutIntLayer;


#ifdef TIMELOGOUT

#ifdef DOUBLETEST
	std::string sType = "Double";
#endif
#ifdef INTTEST
	string sType = "Int";
#endif

	std::string filename = "";

	ostringstream sWid;
	sWid << width;
	ostringstream sHei;
	sHei << height;
	filename.append("time-" + sWid.str() + "-" + sHei.str() + "-" + sType + ".csv");
	ofstream timelogOut(filename);

	if (!timelogOut)
	{
		cerr << "time file open failed." << endl;
	}

	ostream& timeout = timelogOut;
#else
	ostream& timeout = cout;
#endif // TIMELOGOUT

	
	clock_t t1, t2, t3;
	
//------------------------------LocalOperatorTest-----------------------------
#ifdef MATHTEST

#ifdef MATHADD


	numB = new DataType[width*height];

	for (int i = 0; i < width*height; i++)
	{
		numA[i] = i%1000;
		//numB[i] = i + 1;
	}

	layer1.resize(numA, width, height);
	//layer2.resize(numB, width, height);

	layer1.setNoDataValue(2000);

	t1 = clock();

	gpuOutIntLayer = sin(layer1);// +layer2;

	t2 = clock();

	cpuOutIntLayer.resize(width, height);

	for (int idxpixel = 0; idxpixel < height*width; idxpixel++)
	{
		cpuOutIntLayer[idxpixel] = layer1[idxpixel] + layer2[idxpixel];
	}
	t3 = clock();

	

	if (compareLayer(gpuOutIntLayer, cpuOutIntLayer))
	{
		timeout << "LocalOperator" << "," << "+" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "LocalOperator" << "," << "+" << "," << "error" << endl;
	}
#endif //MATHADD

#ifdef MATHDIV


	t1 = clock();
	gpuOutIntLayer = layer1 / layer2;
	t2 = clock();

	cpuOutIntLayer.resize(width, height);

	for (int idxpixel = 0; idxpixel < height*width; idxpixel++)
	{
		cpuOutIntLayer[idxpixel] = layer1[idxpixel] / layer2[idxpixel];
	}
	t3 = clock();

	if (compareLayer(gpuOutIntLayer, cpuOutIntLayer))
	{
		timeout << "LocalOperator" << "," << "/" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "LocalOperator" << "," << "/" << "," << "error" << endl;
	}

#endif //MATHDIV


#ifdef RECLASSVALUEUPDATE

	vector<DataType>oldValueSet;
	vector<DataType>newvalueSet;

	for (int idxvalue = 0; idxvalue < 10; idxvalue++)
	{
		oldValueSet.push_back(idxvalue);
		newvalueSet.push_back(idxvalue + 10);
	}


	for (int idxpixel = 0; idxpixel < width*height; idxpixel++)
	{
		numA[idxpixel] = rand() % 10;
	}

	layer1.resize(numA, width, height);

	t1 = clock();

	gpuOutIntLayer = reclassValueUpdate(layer1, oldValueSet, newvalueSet);

	t2 = clock();

	cpuOutIntLayer.resize(width, height);

	for (int idxpixel = 0; idxpixel < width*height; idxpixel++)
	{
		for (int idxvalue = 0; idxvalue < 10; idxvalue++)
		{
			if ((layer1[idxpixel] - oldValueSet[idxvalue]) < 1e-6)
			{
				cpuOutIntLayer[idxpixel] = newvalueSet[idxvalue];
			}
		}
	}

	t3 = clock();

	if (gpuOutIntLayer == cpuOutIntLayer)
	{
		timeout << "LocalOperator" << "," << "ReclassValueUpdate" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "LocalOperator" << "," << "ReclassValueUpdate" << "," << "error" << endl;
	}


#endif

#ifdef RECLASSVALUEUPDATE

	vector<DataType>oldRangeSet;
	vector<DataType>newRangevalueSet;
	for (int idxvalue = 0; idxvalue < 2; idxvalue++)
	{
		oldRangeSet.push_back(idxvalue * 5);
		newRangevalueSet.push_back(idxvalue);
	}


	for (int idxpixel = 0; idxpixel < width*height; idxpixel++)
	{
		numA[idxpixel] = rand() % 10;
	}

	layer1.resize(numA, width, height);

	t1 = clock();

	gpuOutIntLayer = reclassRangeUpdate(layer1, oldRangeSet, newRangevalueSet);

	t2 = clock();

	cpuOutIntLayer.resize(width, height);

	for (int idxpixel = 0; idxpixel < width*height; idxpixel++)
	{
		for (int idxvalue = 0; idxvalue < 2; idxvalue++)
		{
			if ((oldRangeSet[idxvalue]<layer1[idxpixel])&&(layer1[idxpixel]<=oldRangeSet[idxvalue]))
			{
				cpuOutIntLayer[idxpixel] = newRangevalueSet[idxvalue];
			}
		}
	}

	t3 = clock();

	if (gpuOutIntLayer == cpuOutIntLayer)
	{
		timeout << "LocalOperator" << "," << "ReclassRangeUpdate" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "LocalOperator" << "," << "ReclassRangeUpdate" << "," << "error" << endl;
	}


#endif


#endif //MATHTEST

//-----------------------------End LocalOperatorTest--------------------------


//-----------------------------FocalOperatorTest------------------------------

#ifdef FOCALOPERATORTEST

#ifdef FOCALSUMTEST


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			numA[i*width + j] = rand() % 100;
		}
	}

	layer1.resize(numA, width, height);

	//printLayer(layer1);

	NeighborhoodRect<int>nbrrect(3, 3);

	//gpuOutIntLayer = focalStatisticsSum<double, double, int>(layer1, &nbrrect, NOUSE);

	t1 = clock();

	gpuOutIntLayer = focalStatisticsSum<double,double,int>(layer1, &nbrrect, NOUSE);

	t2 = clock();

	//printLayer(gpuOutIntLayer);

	cpuOutIntLayer.resize(width, height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int m = -1; m <= 1; m++)
			{
				for (int n = -1; n <= 1; n++)
				{
					if ((i + m) >= 0 && (i + m) < height && (j + n) >= 0 && (j + n) < width)
						cpuOutIntLayer[i*width + j] += layer1[(i + m)*width + j + n];
				}
			}
		}
	}

	t3 = clock();

	cout << t2 - t1 << endl;
	cout << t3 - t2 << endl;

	if (gpuOutIntLayer == cpuOutIntLayer)
	{
		timeout << "FocalOperator" << "," << "focalStatisticsSum" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "FocalOperator" << "," << "focalStatisticsSum" << "," << "error" << endl;
	}


#endif

#ifdef FOCALMEANTEST

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			numA[i*width + j] = rand() % 100;
		}
	}

	layer1.resize(numA, width, height);

	//printLayer(layer1);

	NeighborhoodRect<int>nbrrectmean(3, 3);

	//gpuOutIntLayer = focalStatisticsSum<double, double, int>(layer1, &nbrrect, NOUSE);

	t1 = clock();

	gpuOutIntLayer = focalStatisticsMean<double, double, int>(layer1, &nbrrectmean, NOUSE);

	t2 = clock();

	//printLayer(gpuOutIntLayer);

	cpuOutIntLayer.resize(width, height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cpuOutIntLayer[i*width + j] = 0;
			for (int m = -1; m <= 1; m++)
			{
				for (int n = -1; n <= 1; n++)
				{
					if ((i + m) >= 0 && (i + m) < height && (j + n) >= 0 && (j + n) < width)
						cpuOutIntLayer[i*width + j] += layer1[(i + m)*width + j + n];
				}
			}
			cpuOutIntLayer[i*width + j] /= 9;
		}
	}

	t3 = clock();

	cout << t2 - t1 << endl;
	cout << t3 - t2 << endl;
	/*
	printLayer(layer1);
	cout << endl;
	printLayer(gpuOutIntLayer);
	cout << endl;
	printLayer(cpuOutIntLayer);
	*/


	if (gpuOutIntLayer == cpuOutIntLayer)
	{
		timeout << "FocalOperator" << "," << "focalStatisticsMean" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "FocalOperator" << "," << "focalStatisticsMean" << "," << "error" << endl;
	}

#endif

#ifdef FOCALMAXIMUMTEST
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			numA[i*width + j] = rand() % 256;
		}
	}

	layer1.resize(numA, width, height);

	//printLayer(layer1);

	NeighborhoodRect<int>nbrrectmax(5, 5);

	//gpuOutIntLayer = focalStatisticsSum<double, double, int>(layer1, &nbrrect, NOUSE);

	t1 = clock();

	gpuOutIntLayer = focalStatisticsMaximum(layer1, &nbrrectmax);
	//gpuOutIntLayer = focalStatisticsMinimum(layer1, &nbrrectmax);
	

	t2 = clock();

	cpuOutIntLayer.resize(width, height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cpuOutIntLayer[i*width + j] = layer1[i*width + j];
			for (int m = -2; m <= 2; m++)
			{
				for (int n = -2; n <= 2; n++)
				{
					if ((i + m) >= 0 && (i + m) < height && (j + n) >= 0 && (j + n) < width)
					{
						if (cpuOutIntLayer[i*width + j] < layer1[(i + m)*width + j + n])
						{
							cpuOutIntLayer[i*width + j] = layer1[(i + m)*width + j + n];
						}
					}
						
				}
			}
		}
	}

	t3 = clock();

	cout << t2 - t1 << endl;
	cout << t3 - t2 << endl;
	/*
	printLayer(layer1);
	cout << endl;
	printLayer(gpuOutIntLayer);
	cout << endl;
	printLayer(cpuOutIntLayer);
	*/

	if (gpuOutIntLayer == cpuOutIntLayer)
	{
		timeout << "FocalOperator" << "," << "focalStatisticsMaximum" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "FocalOperator" << "," << "focalStatisticsMaximum" << "," << "error" << endl;
	}


#endif

#ifdef FOCALRANGETEST

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			numA[i*width + j] = rand() % 256;
		}
	}

	layer1.resize(numA, width, height);

	//printLayer(layer1);

	NeighborhoodRect<int>nbrrectrange(5, 5);

	

	t1 = clock();

	gpuOutIntLayer = focalStatisticsRange(layer1, &nbrrectrange);


	t2 = clock();

	cpuOutIntLayer.resize(width, height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double minvalue = layer1[i*width + j];
			double maxvalue = layer1[i*width + j];
			for (int m = -2; m <= 2; m++)
			{
				for (int n = -2; n <= 2; n++)
				{
					if ((i + m) >= 0 && (i + m) < height && (j + n) >= 0 && (j + n) < width)
					{
						if (minvalue > layer1[(i + m)*width + j + n])
						{
							minvalue = layer1[(i + m)*width + j + n];
						}

						if (maxvalue < layer1[(i + m)*width + j + n])
						{
							maxvalue = layer1[(i + m)*width + j + n];
						}
					}

				}
			}

			cpuOutIntLayer[i*width + j] = maxvalue - minvalue;
		}
	}

	t3 = clock();

	cout << t2 - t1 << endl;
	cout << t3 - t2 << endl;

	/*
	printLayer(layer1);
	cout << endl;
	printLayer(gpuOutIntLayer);
	cout << endl;
	printLayer(cpuOutIntLayer);
	*/
	if (gpuOutIntLayer == cpuOutIntLayer)
	{
		timeout << "FocalOperator" << "," << "focalStatisticsRange" << "," << width << "," << height << "," << t2 - t1 << "," << t3 - t2 << endl;
	}
	else
	{
		timeout << "FocalOperator" << "," << "focalStatisticsRange" << "," << "error" << endl;
	}

#endif

//------------------------------End FocalOperatorTest-------------------------
#endif

#ifdef TIMELOGOUT
	timelogOut.close();
#endif

	if (numA != NULL)
		delete[] numA;
	if (numB != NULL)
		delete[] numB;
	if (numC != NULL)
		delete[] numC;

	system("pause");
	return 0;
}