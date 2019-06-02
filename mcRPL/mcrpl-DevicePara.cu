#include"mcrpl-DevicePara.h"
__device__ bool cuisValid(long index,long dimLong)
{
	if(index<0||index>=dimLong)
		return false;
	else
		return true;
}