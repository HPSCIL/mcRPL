#ifndef DEVICEINFO_H
#define DEVICEINFO_H
#include<string>
#include<vector>
#include <math.h>
#include <iostream>
#include <curand_kernel.h>
using namespace std;

class deviceinfo
{
public:
	deviceinfo(int id)
	{
		_id=id;
		cudaDeviceProp prop1;
		cudaGetDeviceProperties(&prop1,_id);
		_multiProcessorCount=prop1.multiProcessorCount;
		_computeAblity=prop1.major;
	}
	int id()
	{
		return _id;
	}
	int smCount()
	{
		return _multiProcessorCount;
	}
	bool isValid()
	{
		if(_multiProcessorCount>1&&_computeAblity>=1.9)
			return true;
		else
			return false;
	}

private:
	int _id;
	int _multiProcessorCount;
	int _computeAblity;
};
class DeviceProcessor
{
public:
	DeviceProcessor(string name)
	{
		cudaGetDeviceCount(&_devCount);
		mndeviceOn=name;
		for(int i=0;i<_devCount;i++)
		{
			if(deviceinfo(i).isValid())
				_vValiddevice.push_back(i);
		}
		// MPI_Get_processor_name(name);
	}
	DeviceProcessor()
	{
		cudaGetDeviceCount(&_devCount);
		for(int i=0;i<_devCount;i++)
		{
			if(deviceinfo(i).isValid())
				_vValiddevice.push_back(i);
		}
	}
	void setDeviceOn(string strname)
	{
		mndeviceOn=strname;
	}
	vector<int> allDevice()
	{
		return _vValiddevice;
	}
	bool Valid()
	{
		// allDevice();
		if(_vValiddevice.empty())
		{
			cerr<<__FILE__<<"  "<<__FUNCTION__<<":"<<"there is no valid device"<<endl;
			return false;
		}
		return true;
	}
	int setdevice()
	{
		if(_deviceID<_devCount)
		{
			cudaSetDevice(_deviceID);
			if(_deviceInfo!=NULL)
			{
				_deviceInfo=new deviceinfo(_deviceID);
			}
			else
			{
				delete _deviceInfo;
				_deviceInfo=new deviceinfo(_deviceID);
			}
			return 1;
		}
		else
		{
			cudaSetDevice(0);
			if(_deviceInfo!=NULL)
			{
				_deviceInfo=new deviceinfo(0);
			}
			else
			{
				delete _deviceInfo;
				_deviceInfo=new deviceinfo(0);
			}
			return -1;
		}
	}
	void setDeviceID(int nDevice)
	{
		_deviceID=nDevice;
	}
	int getDeviceID()
	{
		return _deviceID;
	}
	deviceinfo *getDeviceinfo()
	{
		return _deviceInfo;
	}
private:
	int _deviceID;
	int _devCount;
	string mndeviceOn;
	vector<int> _vValiddevice;
	deviceinfo *_deviceInfo;
};
#endif