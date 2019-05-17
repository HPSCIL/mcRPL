#ifndef CPUTEST_H_H
#define CPUTEST_H_H


#include "CuLayer.h"
#include "NeighborhoodSlope.h"

using namespace pRPL;

template<class DataType>
bool compareLayer(CuLayer<DataType>&layer1, CuLayer<DataType>&layer2)
{

	if (layer1.getWidth() != layer2.getWidth())
	{
		return false;
	}

	if (layer1.getHeight() != layer2.getHeight())
	{
		return false;
	}

	if (layer1.getData() == NULL)
	{
		return false;
	}

	if (layer2.getData() == NULL)
	{
		return false;
	}

	int width = layer1.getWidth();
	int height = layer2.getHeight();

	for (int idxpixel = 0; idxpixel < width*height; idxpixel++)
	{
		if (abs(layer1[idxpixel] - layer2[idxpixel])>1e-6)
		{
			cout << "x=" << idxpixel%width << ",y=" << idxpixel / width;
			cout << "layer1:" << layer1[idxpixel] << " " << "layer2:" << layer2[idxpixel] << endl;
			return false;
		}
	}
	return true;
}

template<class DataType>
void printLayer(CuLayer<DataType>&layer)
{
	int width = layer.getWidth();
	int height = layer.getHeight();

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << layer[i*width + j] << " ";
		}
		cout << endl;
	}

}



CuLayer<double> CPUSlopeCal(CuLayer<int>&layer, NeighborhoodSlope* neiSlope)
{

	int width = layer.getWidth();
	int height = layer.getHeight();

	double cellWidth = layer.getCellWidth();


	double cellHeight = layer.getCellHeight();


	if (cellWidth == 0)
		cellWidth = 1;
	if (cellHeight == 0)
		cellHeight = cellWidth;

	int nodata = layer.getNoDataValue();

	CuLayer<double>outlayer(width, height);
	
	CuNbr<int>cuNbr = neiSlope->GetInnerNbr();
	vector<int>nbrcood = cuNbr.coords;
	vector<int>weights = cuNbr.weights;
	int nbrsize = weights.size();


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int focalValue = layer[y*width + x];
			int* focal = layer.getData() + y*width + x;
			if (focalValue == nodata)
			{
				outlayer[y*width + x] = nodata;
				continue;
			}
			double xz = 0;


			for (int i = 0; i < nbrsize / 2; i++)
			{
				int cx = x + nbrcood[i * 2];
				if (cx < 0 || cx >= width)
				{
					xz += focalValue*weights[i];
					continue;
				}
				int cy = y + nbrcood[i * 2 + 1];
				if (cy < 0 || cy >= height)
				{
					xz += focalValue*weights[i];
					continue;
				}
				int curnbr = focal[nbrcood[i * 2] + nbrcood[i * 2 + 1] * width];
				if (curnbr == nodata)
					xz += focalValue*weights[i];
				else
					xz += curnbr*weights[i];

			}

			xz /= (nbrsize*cellWidth / 2);

			double yz = 0;

			for (int i = nbrsize / 2; i < nbrsize; i++)
			{
				int cx = x + nbrcood[i * 2];
				if (cx < 0 || cx >= width)
				{
					yz += focalValue*weights[i];
					continue;
				}
				int cy = y + nbrcood[i * 2 + 1];
				if (cy < 0 || cy >= height)
				{
					yz += focalValue*weights[i];
					continue;
				}
				int curnbr = focal[nbrcood[i * 2] + nbrcood[i * 2 + 1] * width];
				if (curnbr == nodata)
					yz += focalValue*weights[i];
				else
					yz += curnbr*weights[i];
			}

			yz /= (nbrsize*cellHeight / 2);


			outlayer[y*width+x] = atan(sqrt(xz*xz + yz*yz))*57.29578;

		}
	}

	return outlayer;
}



CuLayer<int> CPUEucAlloCal(CuLayer<int>&layer, vector<rasterCell>&vRasterCell)
{
	int width = layer.getWidth();
	int height = layer.getHeight();

	double cellwidth = layer.getCellWidth();
	double cellheight = layer.getCellHeight();

	if (std::abs(cellwidth) < 1e-6)
		cellwidth = 1;
	if (std::abs(cellheight) < 1e-6)
		cellheight = cellwidth;

	int noDataValue = layer.getNoDataValue();

	CuLayer<int>outlayer(width, height);
	int cellnum = vRasterCell.size();

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (layer[width*y + x] != noDataValue)
			{
				outlayer[y*width + x] = layer[y*width + x];
			}
			double mindist = pow(width*cellwidth, 2) + pow(height*cellheight, 2);
			int minzonel = noDataValue;

			for (int i = 0; i < cellnum; i++)
			{
				double x_dist = abs(x - vRasterCell[i].x) * cellwidth;
				double y_dist = abs(y - vRasterCell[i].y) * cellheight;
				double dist = sqrt(x_dist*x_dist + y_dist*y_dist);
				if (dist < mindist)
				{
					mindist = dist;
					minzonel = vRasterCell[i].value;
				}
			}
			outlayer[y*width + x] = minzonel;
		}
	}

	return outlayer;
}


vector<double> CPUZonelAreaCal(CuLayer<int>&culayer, vector<int>&zonelvalues)
{
	int width = culayer.getWidth();
	int height = culayer.getHeight();

	double cellwidth = culayer.getCellWidth();
	double cellheight = culayer.getCellHeight();

	if (std::abs(cellwidth) < 1e-6)
		cellwidth = 1;
	if (std::abs(cellheight) < 1e-6)
		cellheight = cellwidth;

	int noDataValue = culayer.getNoDataValue();

	vector<double>output;
	output.resize(zonelvalues.size(), 0);

	int zonelnum = zonelvalues.size();
	double cellarea = cellwidth*cellheight;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (culayer[y*width + x] == noDataValue)
			{
				continue;
			}

			for (int idxzonel = 0; idxzonel < zonelnum; idxzonel++)
			{
				if (culayer[y*width + x] == zonelvalues[idxzonel])
				{
					output[idxzonel] += cellarea;
					break;
				}
			}

		}
	}


	return output;

}




#endif