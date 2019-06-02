#ifndef CULAYER_H_H
#define CULAYER_H_H

#include <string.h>
#include "mcrpl-errorhelper.h"
#include "mcrpl-OperatorControl.h"
#include "mcrpl-CuEnvControl.h"
#include <gdal_priv.h>



namespace mcRPL
{
	//template<class DataType> class OperControl;

		GDALDataType getGDALDataType(std::string dataTypeName)
	{
		GDALDataType gdalType = GDT_Unknown;
		if (dataTypeName == typeid(unsigned char).name()) 
		{
			gdalType = GDT_Byte;
		}
		else if (dataTypeName == typeid(unsigned short int).name()) 
		{
			gdalType = GDT_UInt16;
		}
		else if (dataTypeName == typeid(short int).name()) 
		{
			gdalType = GDT_Int16;
		}
		else if (dataTypeName == typeid(unsigned int).name()) 
		{
			gdalType = GDT_UInt32;
		}
		else if (dataTypeName == typeid(int).name()) 
		{
			gdalType = GDT_Int32;
		}
		else if (dataTypeName == typeid(float).name()) 
		{
			gdalType = GDT_Float32;
		}
		else if (dataTypeName == typeid(double).name()) 
		{
			gdalType = GDT_Float64;
		}
		return gdalType;
	}


	char* findImageTypeGDAL(std::string DstImgFileName)
	{
		int index = DstImgFileName.find_last_of('.');
		std::string dstExtension = DstImgFileName.substr(index + 1, DstImgFileName.length() - index - 1);
		char *Gtype = NULL;
		if (dstExtension == "bmp") Gtype = "BMP";
		else if (dstExtension == "jpg") Gtype = "JPEG";
		else if (dstExtension == "png") Gtype = "PNG";
		else if (dstExtension == "tif") Gtype = "GTiff";
		else if (dstExtension == "gif") Gtype = "GIF";
		else Gtype = NULL;

		return Gtype;
	}

	template<class DataType>
	class CuLayer
	{
	public:
		CuLayer();
		CuLayer(DataType* data, int width, int height);
		CuLayer(int width, int height);
		

		~CuLayer();

		CuLayer(const CuLayer<DataType>&culayer);
		CuLayer<DataType>& operator=(const CuLayer<DataType>&culayer);


		int Read(string filePath);
		int Write(string filePath);

		DataType& operator[](int idxpixel){ return this->m_data[idxpixel]; };

		bool operator==(CuLayer<DataType>&culayer);

		void resize(int width, int height);
		void resize(DataType* data, int width, int height);

		const int getWidth()const{ return this->m_width; };
		const int getHeight()const{ return this->m_height; };
		DataType* getData(){ return this->m_data; };

		DataType getNoDataValue(){ return this->m_noData; };
		double getCellWidth(){ return this->m_adfGeoTransform[1]; };
		double getCellHeight(){ return this->m_adfGeoTransform[5]; };
		double getGeoTransform(double* adfGeoTransform);
		string getProjection(){ return this->m_projection; };

		void setNoDataValue(DataType nodata){ this->m_noData = nodata; };
		void setCellHeight(double cellHeight){ this->m_adfGeoTransform[4] = cellHeight; };
		void setCellWidth(double cellWidth){ this->m_adfGeoTransform[1] = cellWidth; };
		void setGeoTransform(double* adfGeoTransform);
		void setProjection(string projection){ this->m_projection = projection; };


		CuLayer<DataType> operator+(const CuLayer<DataType>&culayer);

		template<class ParamType>
		CuLayer<DataType> operator+(const ParamType& param);

		template<class T,class ParamType>
		friend CuLayer<T> operator+(const ParamType& param, const CuLayer<T>&culayer);


		CuLayer<DataType> operator-(const CuLayer<DataType>&culayer);

		template<class ParamType>
		CuLayer<DataType> operator-(const ParamType& param);

		template<class T, class ParamType>
		friend CuLayer<T> operator-(const ParamType& param, const CuLayer<T>&culayer);


		CuLayer<DataType> operator*(const CuLayer<DataType>&culayer);

		template<class ParamType>
		CuLayer<DataType> operator*(const ParamType& param);

		template<class T, class ParamType>
		friend CuLayer<T> operator*(const ParamType& param, const CuLayer<T>&culayer);

		CuLayer<DataType> operator/(const CuLayer<DataType>&culayer);

		template<class ParamType>
		CuLayer<DataType> operator/(const ParamType& param);

		template<class T, class ParamType>
		friend CuLayer<T> operator/(const ParamType& param, const CuLayer<T>&culayer);

	private:

		DataType getDefaultNoDataVal();


		void layerAdd(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height);

		template<class ParamType>
		void layerAdd(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height);

		void layerSubtract(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height);

		template<class ParamType>
		void layerSubtract(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height);

		void layerMultiply(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height);

		template<class ParamType>
		void layerMultiply(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height);

		void layerDivide(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height);

		template<class ParamType>
		void layerDivide(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height);




	private:

		DataType* m_data;
		int m_width;
		int m_height;

		DataType m_noData;
		double m_adfGeoTransform[6];
		string m_projection;
		
		
	};

	template<class DataType>
	CuLayer<DataType>::CuLayer()
	{
		this->m_data = NULL;
		this->m_width = 0;
		this->m_height = 0;
		this->m_noData = this->getDefaultNoDataVal();
		this->m_projection = "";
		memset(m_adfGeoTransform, 0, sizeof(double) * 6);
	}

	template<class DataType>
	CuLayer<DataType>::CuLayer(DataType* data, int width, int height)
	{
		if (data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}
		this->m_width = width;
		this->m_height = height;

		this->m_data = new DataType[this->m_width*this->m_height];
		memcpy(this->m_data, data, sizeof(DataType)*this->m_width*this->m_height);

		this->m_noData = this->getDefaultNoDataVal();
		this->m_projection = "";
		memset(m_adfGeoTransform, 0, sizeof(double) * 6);

	}

	template<class DataType>
	CuLayer<DataType>::CuLayer(int width, int height)
	{
		if (0 == width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}
		this->m_width = width;
		this->m_height = height;
		this->m_data = new DataType[width*height];

		this->m_noData = this->getDefaultNoDataVal();
		this->m_projection = "";
		memset(m_adfGeoTransform, 0, sizeof(double) * 6);
	}


	
	template<class DataType>
	CuLayer<DataType>::~CuLayer()
	{
		if (m_data != NULL)
		{
			delete[] m_data;
			m_data = NULL;
		}
	}


	template<class DataType>
	CuLayer<DataType>::CuLayer(const CuLayer<DataType>&culayer)
	{

		if (culayer.m_data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}
		this->m_width = culayer.m_width;
		this->m_height = culayer.m_height;
		this->m_noData = culayer.m_noData;
		this->m_projection = culayer.m_projection;
		memcpy(this->m_adfGeoTransform, culayer.m_adfGeoTransform, sizeof(double) * 6);

		this->m_data = new DataType[m_width*m_height];

		memcpy(this->m_data, culayer.m_data, sizeof(DataType)*m_width*m_height);
	}

	template<class DataType>
	CuLayer<DataType>& CuLayer<DataType>::operator=(const CuLayer<DataType>&culayer)
	{
		if (culayer.m_data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		if (0 == m_width || 0 == m_height)
		{
			m_data = NULL;
		}

		if (m_data == NULL)
		{
			this->m_width = culayer.m_width;
			this->m_height = culayer.m_height;
			this->m_data = new DataType[m_width*m_height];
		}
		else
		{
			if (this->m_width != culayer.m_width || this->m_height != culayer.m_height)
			{
				delete[] m_data;
				this->m_data = new DataType[m_width*m_height];
				this->m_width = culayer.m_width;
				this->m_height = culayer.m_height;
			}
		}
		this->m_noData = culayer.m_noData;
		this->m_projection = culayer.m_projection;
		memcpy(this->m_adfGeoTransform, culayer.m_adfGeoTransform, sizeof(double) * 6);

		memcpy(this->m_data, culayer.m_data, sizeof(DataType)*m_width*m_height);
		return *(this);
	}


	template<class DataType>
	int CuLayer<DataType>::Read(string filePath)
	{

		GDALAllRegister();
		CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
		GDALDataset *ReadDataSet = (GDALDataset*)GDALOpen(filePath.c_str(), GA_ReadOnly);
		if (ReadDataSet == NULL)
		{
			return -1;
		}

		this->m_width = ReadDataSet->GetRasterXSize();
		this->m_height= ReadDataSet->GetRasterYSize();
		int bandCount = ReadDataSet->GetRasterCount();


		//暂时要求一个波段，后期考虑多个波段
		//if (bandCount > 1)
	//		return -1;
		
		ReadDataSet->GetGeoTransform(this->m_adfGeoTransform);
		m_noData = DataType(ReadDataSet->GetRasterBand(1)->GetNoDataValue());
		m_projection = ReadDataSet->GetProjectionRef();

		DataType *pImageBuf = NULL;
		pImageBuf = new DataType[m_width*m_height*bandCount];

		if (NULL == pImageBuf)
		{
			delete ReadDataSet; ReadDataSet = NULL;
			return -2;
		}

		GDALDataType rasterDataType = getGDALDataType(typeid(DataType).name());

		//cout << ReadDataSet->GetRasterBand(1)->GetRasterDataType() << endl;


		if (ReadDataSet->RasterIO(GF_Read, 0, 0, this->m_width, this->m_height, pImageBuf, this->m_width, this->m_height, rasterDataType, bandCount, NULL, 0, 0, 0) == CE_Failure)
		{
			delete ReadDataSet; ReadDataSet = NULL;
			delete[] pImageBuf; pImageBuf = NULL;
			return -3;
		}
		
		this->m_data = pImageBuf;

		delete ReadDataSet;

		return 0;

	}

	template<class DataType>
	int CuLayer<DataType>::Write(string filePath)
	{
		GDALAllRegister();
		CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
		char *GType = NULL;
		GType = findImageTypeGDAL(filePath);
		if (GType == NULL)
		{
			return -4;
		}

		GDALDriver *pMemDriver = NULL;
		pMemDriver = GetGDALDriverManager()->GetDriverByName(GType);
		if (pMemDriver == NULL)
		{
			return -5;
		}

		int bandCount = 1;

		GDALDataType rasterDataType = getGDALDataType(typeid(DataType).name());
		
		GDALDataset * pMemDataSet = pMemDriver->Create(filePath.c_str(), this->m_width, this->m_height, bandCount, rasterDataType, NULL);

		if (pMemDataSet == NULL)
		{
			printError("Create filePath failed.");
			return -6;
		}

		pMemDataSet->SetGeoTransform(this->m_adfGeoTransform);
		pMemDataSet->SetProjection(this->m_projection.c_str());


		GDALRasterBand *pBand = NULL;
		int nLineCount = this->m_width * bandCount;
		int silceSize = this->m_width*this->m_height;
		
		for (int i = 1; i <= bandCount; i++)
		{
			pBand = pMemDataSet->GetRasterBand(i);
			pBand->RasterIO(GF_Write,
				0,
				0,
				this->m_width,
				this->m_height,
				this->m_data + (i - 1)*silceSize,
				this->m_width,
				this->m_height,
				rasterDataType,
				0,
				0);
			pBand->SetNoDataValue(this->m_noData);
		}

		GDALClose(pMemDataSet);
		GetGDALDriverManager()->DeregisterDriver(pMemDriver);

	}





	template<class DataType>
	bool CuLayer<DataType>::operator==(CuLayer<DataType>&culayer)
	{
		if (this->m_width != culayer.m_width)
		{
			return false;
		}

		if (this->m_height != culayer.m_height)
		{
			return false;
		}

		if (this->m_data == NULL)
		{
			return false;
		}

		if (culayer.m_data == NULL)
		{
			return false;
		}

		for (int idxpixel = 0; idxpixel < m_width*m_height; idxpixel++)
		{
			if (fabs(this->m_data[idxpixel] - culayer.m_data[idxpixel])>1e-6)
			{
				//cout << abs(layer1[idxpixel] - layer2[idxpixel]) << endl;
				return false;
			}
		}
		return true;
	}


	template<class DataType>
	void CuLayer<DataType>::resize(int width, int height)
	{
		if (0 == width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		if (0 == m_width || 0 == m_height)
		{
			m_data = NULL;
		}


		if (m_width != width||m_height != height)
		{
			if (m_data!=NULL)
				delete[] m_data;
			this->m_data = new DataType[width*height];
			m_width = width;
			m_height = height;
		}
	}

	template<class DataType>
	void CuLayer<DataType>::resize(DataType* data, int width, int height)
	{
		if (0 == width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		if (0 == m_width || 0 == m_height)
		{
			m_data = NULL;
		}


		if (m_width != width || m_height != height)
		{
			if (m_data==NULL)
				delete[] m_data;
			this->m_data = new DataType[width*height];
			m_width = width;
			m_height = height;
		}

		memcpy(this->m_data, data, sizeof(DataType)*m_width*m_height);

	}

	template<class DataType>
	DataType CuLayer<DataType>::getDefaultNoDataVal()
	{
		string dataTypeName = typeid(DataType).name();

		if (dataTypeName == typeid(unsigned char).name())
		{
			return DEFAULT_NODATA_UCHAR;
		}
		else if (dataTypeName == typeid(unsigned short int).name())
		{
			return DEFAULT_NODATA_USHORT;
		}
		else if (dataTypeName == typeid(short int).name())
		{
			return DEFAULT_NODATA_SHORT;
		}
		else if (dataTypeName == typeid(unsigned int).name())
		{
			return DEFAULT_NODATA_UINT;
		}
		else if (dataTypeName == typeid(int).name())
		{
			return DEFAULT_NODATA_INT;
		}
		else if (dataTypeName == typeid(float).name())
		{
			return DEFAULT_NODATA_FLOAT;
		}
		else if (dataTypeName == typeid(double).name())
		{
			return DEFAULT_NODATA_DOUBLE;
		}
		else
		{
			printError("NoData type isn't supported");
			exit(EXIT_FAILURE);
		}

	}

	template<class DataType>
	double CuLayer<DataType>::getGeoTransform(double* adfGeoTransform)
	{
		if (adfGeoTransform == NULL)
		{
			printError("adfGeoTransform is null.");
			exit(EXIT_FAILURE);
		}

		memcpy(adfGeoTransform, m_adfGeoTransform, sizeof(double) * 6);

	}

	template<class DataType>
	void CuLayer<DataType>::setGeoTransform(double* adfGeoTransform)
	{
		if (adfGeoTransform == NULL)
		{
			printError("adfGeoTransform is null.");
			exit(EXIT_FAILURE);
		}

		memcpy(m_adfGeoTransform, adfGeoTransform, sizeof(double) * 6);

	}

	template<class DataType>
	CuLayer<DataType> CuLayer<DataType>::operator+(const CuLayer<DataType>&culayer)
	{
		
		if (culayer.m_width != this->m_width)
		{
			printError("The two layers are of different width.");
			exit(EXIT_FAILURE);
		}
		if (culayer.m_height != this->m_height)
		{
			printError("The two layers are of different height.");
			exit(EXIT_FAILURE);
		}

		CuLayer<DataType>outlayer(this->m_width, this->m_height);

		if (CuEnvControl::getCudaLayerAddState() == false)
		{
			this->layerAdd(this->m_data, culayer.m_data, outlayer.m_data, this->m_width, this->m_height);
		}
		outlayer.m_noData = this->m_noData;
		return outlayer;
	}

	/*template<class DataType>
	template<class ParamType> CuLayer<DataType> CuLayer<DataType>::operator+(const ParamType& param)
	{
		CuLayer<DataType>outlayer(this->m_width, this->m_height);
		outlayer.m_noData = this->m_noData;
		if (CuEnvControl::getCudaLayerAddState() == false)
		{
			this->layerAdd(this->m_data, outlayer.m_data, param, m_width, m_height);
		}

		return  outlayer;
	}*/

	template<class DataType>
	CuLayer<DataType> CuLayer<DataType>::operator-(const CuLayer<DataType>&culayer)
	{
		if (culayer.m_width != this->m_width)
		{
			printError("The two layers are of different width.");
			exit(EXIT_FAILURE);
		}
		if (culayer.m_height != this->m_height)
		{
			printError("The two layers are of different height.");
			exit(EXIT_FAILURE);
		}

		CuLayer<DataType>outlayer(this->m_width, this->m_height);

		if (CuEnvControl::getCudaLayerSubState == false)
		{
			this->layerSubstract(this->m_data, culayer.m_data, outlayer.m_data, this->m_width, this->m_height);
		}

		return outlayer;
	}

	//template<class DataType>
	//template<class ParamType> CuLayer<DataType> CuLayer<DataType>::operator-(const ParamType& param)
	//{
	//	CuLayer<DataType>outlayer(this->m_width, this->m_height);

	//	if (CuEnvControl::getCudaLayerSubState() == false)
	//	{
	//		this->layerSubtract(this->m_data, outlayer.m_data, param, m_width, m_height);
	//	}

	//	return  outlayer;
	//}

	template<class DataType>
	CuLayer<DataType> CuLayer<DataType>::operator*(const CuLayer<DataType>&culayer)
	{
		if (culayer.m_width != this->m_width)
		{
			printError("The two layers are of different width.");
			exit(EXIT_FAILURE);
		}
		if (culayer.m_height != this->m_height)
		{
			printError("The two layers are of different height.");
			exit(EXIT_FAILURE);
		}

		CuLayer<DataType>outlayer(this->m_width, this->m_height);

		if (CuEnvControl::getCudaLayerMulState() == false)
		{
			this->layerMultiply(this->m_data, culayer.m_data, outlayer.m_data, this->m_width, this->m_height);
		}

		return outlayer;
	}

	/*template<class DataType>
	template<class ParamType> CuLayer<DataType> CuLayer<DataType>::operator*(const ParamType& param)
	{
		CuLayer<DataType>outlayer(this->m_width, this->m_height);

		if (CuEnvControl::getCudaLayerMulState() == false)
		{
			this->layerMultiply(this->m_data, outlayer.m_data, param, m_width, m_height);
		}

		return  outlayer;
	}*/

	template<class DataType>
	CuLayer<DataType> CuLayer<DataType>::operator/(const CuLayer<DataType>&culayer)
	{
		if (culayer.m_width != this->m_width)
		{
			printError("The two layers are of different width.");
			exit(EXIT_FAILURE);
		}
		if (culayer.m_height != this->m_height)
		{
			printError("The two layers are of different height.");
			exit(EXIT_FAILURE);
		}

		CuLayer<DataType>outlayer(this->m_width, this->m_height);

		if (CuEnvControl::getCudaLayerDivState() == false)
		{
			this->layerDivide(this->m_data, culayer.m_data, outlayer.m_data, this->m_width, this->m_height);
		}

		return outlayer;
	}

	/*template<class DataType>
	template<class ParamType> CuLayer<DataType> CuLayer<DataType>::operator/(const ParamType& param)
	{
		CuLayer<DataType>outlayer(this->m_width, this->m_height);

		if (CuEnvControl::getCudaLayerDivState() == false)
		{
			this->layerDivide(this->m_data, outlayer.m_data, param, m_width, m_height);
		}

		return  outlayer;
	}*/




	template<class DataType>
	void CuLayer<DataType>::layerAdd(DataType* layer1data, DataType* layer2data, DataType* layerResult,int width,int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			if (std::abs(layer1data[idxpixel] - this->m_noData) > 1e-6&&std::abs(layer2data[idxpixel] - this->m_noData) > 1e-6)
			{
				layerResult[idxpixel] = layer1data[idxpixel] + layer2data[idxpixel];
			}
			else
			{
				layerResult[idxpixel] = this->m_noData;
			}
				
		}
	}

	template<class DataType>
	template<class ParamType> void CuLayer<DataType>::layerAdd(DataType* layer1data,DataType* layerResult, ParamType param, int width, int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			layerResult[idxpixel] = layer1data[idxpixel] + param;
		}
	}


	template<class DataType>
	void CuLayer<DataType>::layerSubtract(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			layerResult[idxpixel] = layer1data[idxpixel] - layer2data[idxpixel];
		}
	}

	template<class DataType>
	template<class ParamType> void CuLayer<DataType>::layerSubtract(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			layerResult[idxpixel] = layer1data[idxpixel] - param;
		}
	}
	

	template<class DataType>
	void CuLayer<DataType>::layerMultiply(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			layerResult[idxpixel] = layer1data[idxpixel] * layer2data[idxpixel];
		}
	}

	template<class DataType>
	template<class ParamType> void CuLayer<DataType>::layerMultiply(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			layerResult[idxpixel] = layer1data[idxpixel] * param;
		}
	}

	template<class DataType>
	void CuLayer<DataType>::layerDivide(DataType* layer1data, DataType* layer2data, DataType* layerResult, int width, int height)
	{
		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			try
			{
				if (layer2data[idxpixel] == 0)
				{
					throw 0;
				}
				else
				{
					layerResult[idxpixel] = layer1data[idxpixel] / layer2data[idxpixel];
				}
			}
			catch (int code)
			{
				printError("The divide() method tries to perform a division by zero and raises an exception.");
				exit(EXIT_FAILURE);
			}
			
		}
	}

	template<class DataType>
	template<class ParamType> void CuLayer<DataType>::layerDivide(DataType* layer1data, DataType* layerResult, ParamType param, int width, int height)
	{
		if (param == 0)
		{
			printError("The divide() method tries to perform a division by zero and raises an exception.");
			exit(EXIT_FAILURE);
		}


		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			if (std::abs(layer1data[idxpixel] - this->m_noData)>1e-6)
				layerResult[idxpixel] = layer1data[idxpixel] / param;
			else
				layerResult[idxpixel] = this->m_noData;
		}
	}


	template<class DataType,class ParamType> 
	CuLayer<DataType> operator+(const ParamType& param, const CuLayer<DataType>&culayer)
	{
		if (culayer.m_data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		int width = culayer.m_width;
		int height = culayer.m_height;

		CuLayer<DataType>outlayer(width, height);

		//layerAdd<ParamType>(culayer.m_data, outlayer.m_data, param, width, height);

		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			outlayer.m_data[idxpixel] = param + culayer.m_data[idxpixel];
		}

		return outlayer;

	}


	template<class DataType, class ParamType>
	CuLayer<DataType> operator-(const ParamType& param, const CuLayer<DataType>&culayer)
	{
		if (culayer.m_data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		int width = culayer.m_width;
		int height = culayer.m_height;

		CuLayer<DataType>outlayer(width, height);

		//layerAdd<ParamType>(culayer.m_data, outlayer.m_data, param, width, height);

		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			outlayer.m_data[idxpixel] = param - culayer.m_data[idxpixel];
		}

		return outlayer;

	}

	template<class DataType, class ParamType>
	CuLayer<DataType> operator*(const ParamType& param, const CuLayer<DataType>&culayer)
	{
		if (culayer.m_data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		int width = culayer.m_width;
		int height = culayer.m_height;

		CuLayer<DataType>outlayer(width, height);

		//layerAdd<ParamType>(culayer.m_data, outlayer.m_data, param, width, height);

		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			outlayer.m_data[idxpixel] = param * culayer.m_data[idxpixel];
		}

		return outlayer;

	}

	template<class DataType, class ParamType>
	CuLayer<DataType> operator/(const ParamType& param, const CuLayer<DataType>&culayer)
	{
		if (culayer.m_data == NULL)
		{
			printError("data is Null.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_width)
		{
			printError("width==0.");
			exit(EXIT_FAILURE);
		}
		if (0 == culayer.m_height)
		{
			printError("height==0.");
			exit(EXIT_FAILURE);
		}

		int width = culayer.m_width;
		int height = culayer.m_height;

		CuLayer<DataType>outlayer(width, height);

		//layerAdd<ParamType>(culayer.m_data, outlayer.m_data, param, width, height);

		int layerSize = width*height;

		for (int idxpixel = 0; idxpixel < layerSize; idxpixel++)
		{
			try
			{
				if (culayer.m_data[idxpixel == 0])
				{
					throw 0;
				}
				else
				{
					outlayer.m_data[idxpixel] = param / culayer.m_data[idxpixel];
				}
			}
			catch (int code)
			{
				if (code == 0)
				{
					printError("The divide() method tries to perform a division by zero and raises an exception.")
				}
			}
		}

		return outlayer;

	}

}







#endif