#include <iostream>
#include <fstream>
#include <sstream>

#include "CuLayer.h"

#include "CuEnvControl.h"
#include "LocalOperator.h"
#include "FocalOperator.h"
#include "GlobalOperator.h"
#include "ZonalOperator.h"
//#include "ZonalOperatorDevice.h"


#include "NeighborhoodSlope.h"
#include <time.h>

#include "cputest.h"

using namespace std;
using namespace CuPRL;



int main(int argc, char* argv[])
{
	/*
	
	int testdata[36] = { 1, 1, 1, 1, 1, 1,
	1, 3, 3, 2, 1, 10,
	1, 1, 3, 2, 2, 2,
	1, 2, 2, 2, 2, 2,
	1, 1, 1, 2, 2, 2,
	1, 1, 1, 1, 1, 2 };


	CuLayer<int>testlayer1(testdata, 6, 6);
	testlayer1.setCellHeight(1);
	testlayer1.setCellWidth(1);
	testlayer1.setNoDataValue(10);

	printLayer(testlayer1);
	

	CuEnvControl::setBlockDim(8, 8);
	*/
	CuLayer<int>testlayer1;
	testlayer1.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\globalTest\\globalTest4.tif");
	//testlayer1.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\zonelTest\\zonel1.tif");

	//testlayer1.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\focalTest\\TestRasterWH.tif");

	/*vector<int>vZonels;

	for (int i = 0; i < 75; i++)
	{
		vZonels.push_back(i + 1);
	}
	*/

	vector<rasterCell>vRasterCell = getGlobalPoints(testlayer1);

	clock_t t1, t2;

	t1 = clock();

	//vector<double>areas = cuZonelStatisticSum<double, ZonelAreaCal>(testlayer1, vZonels);
	
	CuLayer<int>testgpulayer = cuGlobalOperatorFn<int, int, EucAlloCal>(testlayer1, vRasterCell);

	t2 = clock();

	cout << t2 - t1 << endl;

	t1 = clock();


	CuLayer<int>testcpulayer = CPUEucAlloCal(testlayer1, vRasterCell);

	t2 = clock();

	cout << t2 - t1 << endl;
	
	if (compareLayer(testcpulayer, testgpulayer) == false)
	{
		cout << "result error" << endl;
	}
	else
	{
		cout << "result right" << endl;
	}
	/*
	int t = 58 + 3251 * testlayer1.getWidth();

	for (int i = -1; i < 1; i++)
	{
	for (int j = -1; j <= 1; j++)
	{
	cout << testlayer1[t + i*testlayer1.getWidth() + j] << " ";
	}
	cout << endl;
	}
	*/
	

	for (int i = 0; i < 20; i++)
	{
		//std::cout << testgpulayer[i] << std::endl;
	}

	system("pause");
	return 0;
}