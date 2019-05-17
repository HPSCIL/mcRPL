#include <iostream>
#include <fstream>
#include <sstream>

#include "CuLayer.h"

#include "CuEnvControl.h"
#include "LocalOperator.h"
#include "FocalOperator.h"
#include <time.h>

#include "cputest.h"

using namespace std;
using namespace CuPRL;



int main(int argc, char* argv[])
{

	CuLayer<float>testlayer1;
	CuLayer<float>testlayer2;
	CuLayer<float>testlayer3;
	CuLayer<float>testlayer4;
	CuLayer<float>testlayer5;
	CuLayer<float>testlayer6;
	CuLayer<float>testlayer7;
	CuLayer<float>testlayer8;

	CuLayer<float>testlayer9;
	CuLayer<float>testlayer10;
	CuLayer<float>testlayer11;
	CuLayer<float>testlayer12;
	CuLayer<float>testlayer13;
	CuLayer<float>testlayer14;
	CuLayer<float>testlayer15;
	CuLayer<float>testlayer16;

	//CuEnvControl::setBlockDim(8, 8);

	testlayer1.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\testRaster11.tif");
	testlayer2.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\testRaster11.tif");
	testlayer3.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\testRaster11.tif");
	testlayer4.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\testRaster11.tif");
	testlayer5.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\testRaster11.tif");




	/*testlayer1.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\localTest1.tif");
	testlayer2.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\localTest2.tif");
	testlayer3.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\localTest3.tif");
	testlayer4.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\localTest4.tif");
	testlayer5.Read("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\localTest5.tif");*/

	testlayer6 = testlayer5;
	testlayer7 = testlayer5;
	testlayer8 = testlayer5;

	testlayer9 = testlayer5;
	testlayer10 = testlayer5;
	testlayer11 = testlayer5;
	testlayer12 = testlayer5;
	testlayer13 = testlayer5;
	testlayer14 = testlayer5;
	testlayer15 = testlayer5;
	testlayer16 = testlayer5;
	





	cout << testlayer1[0] << endl;
	cout << testlayer2[0] << endl;


	vector<CuLayer<float>>testlayerSet;
	testlayerSet.push_back(testlayer1);
	testlayerSet.push_back(testlayer2);
	
	//testlayerSet.push_back(testlayer3);
	//testlayerSet.push_back(testlayer4);
	/*
	testlayerSet.push_back(testlayer5);
	testlayerSet.push_back(testlayer6);
	testlayerSet.push_back(testlayer7);
	testlayerSet.push_back(testlayer8);

	testlayerSet.push_back(testlayer9);
	testlayerSet.push_back(testlayer10);
	testlayerSet.push_back(testlayer11);
	testlayerSet.push_back(testlayer12);
	testlayerSet.push_back(testlayer13);
	testlayerSet.push_back(testlayer14);
	testlayerSet.push_back(testlayer15);
	testlayerSet.push_back(testlayer16);
	*/


	
	clock_t t1, t2;

	t1 = clock();
	CuLayer<float>testgpulayer = cuLocalOperatorFn1<float, float, MultiRasterMean>(testlayerSet);

	//CuLayer<float>testgpulayer = pow<float,float,int>(testlayer1,2);
	//CuLayer<float>testgpulayer = sin<float, float>(testlayer1);



	t2 = clock();

	cout << t2 - t1 << endl;
	//cout << testgpulayer[0] << endl;

	testgpulayer.Write("C:\\Users\\HP\\Desktop\\WH-DEM\\localTest\\testRasterResult.tif");

	t1 = clock();
	//CuLayer<float>layercpulayer = (testlayer1 + testlayer2) / 2;
	//cout << layercpulayer[0] << endl;
	//layercpulayer = layercpulayer / 2;

	//CuLayer<float>layercpulayer = (testlayer1 + testlayer2 + testlayer3 + testlayer4) / 4;
	
	//CuLayer<float>layercpulayer = (testlayer1 + testlayer2 + testlayer3 + testlayer4 + testlayer5 + testlayer6 + testlayer7 + testlayer8) / 8;

	//CuLayer<float>layercpulayer = (testlayer1 + testlayer2 + testlayer3 + testlayer4 + testlayer5 + testlayer6 + testlayer7 + testlayer8+testlayer9 + testlayer10 + testlayer11 + testlayer12 + testlayer13 + testlayer14 + testlayer15 + testlayer16) / 16;


	t2 = clock();

	cout << t2 - t1 << endl;
	/*
	if (compareLayer(layercpulayer, testgpulayer) == false)
	{
		cout << "result error" << endl;
	}
	else
	{
		cout << "result right" << endl;
	}
	
	cout << layercpulayer[0] << endl;
	cout << testgpulayer[0] << endl;
	*/

	system("pause");
	return 0;
}