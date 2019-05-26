#ifndef LAYERINFO_H_H
#define LYAERINFO_H_H
#include "CuPRL.h"
namespace mcPRL
{
	template<class DataType>
	class LayerInfo
	{
	public:
		__device__ LayerInfo(DataType* data, int x, int y, int width, int height, double cellSizeX, double cellSizeY, DataType nodata);
		__device__ DataType getCellValue(){ return m_data[m_x + m_y*m_width]; };
		__device__ DataType* getCellPtr(){ return m_data + m_y*m_width + m_x; };
		__device__ int getX(){ return m_x; };
		__device__ int getY(){ return m_y; };
		__device__ int getWidth(){ return m_width; };
		__device__ int getHeight(){ return m_height; };
		__device__ double getCellSizeX(){ return m_cellSizeX; };
		__device__ double getCellSizeY(){ return m_cellSizeY; };
		__device__ DataType getNoData(){ return m_nodata; };
		__device__ DataType getCellValue(int x, int y){ return m_data[x + y*m_width]; };
		__device__ DataType* getCellPtr(int x, int y){ return m_data + x + y*m_width; };
	private:
		DataType* m_data;
		int m_x;
		int m_y;
		int m_width;
		int m_height;
		double m_cellSizeX;
		double m_cellSizeY;
		DataType m_nodata;
	};
	template<class DataType>
	__device__ LayerInfo<DataType>::LayerInfo(DataType* data, int x, int y, int width, int height, double cellSizeX, double cellSizeY, DataType nodata)
	{
		this->m_data = data;
		this->m_x = x;
		this->m_y = y;
		this->m_width = width;
		this->m_height = height;
		this->m_cellSizeX = cellSizeX;
		this->m_cellSizeY = cellSizeY;
		this->m_nodata = nodata;
	}
}
#endif