#ifndef DEVICE_PARA_H
#define DEVICE_PARA_H
#include<cuda_runtime.h>
__constant__  static const int              CUERROR_VAL = -9999;
__constant__ static bool             CUDEFAULT_NODATA_BOOL = false;
__constant__ static unsigned char    CUDEFAULT_NODATA_UCHAR = 255;
__constant__ static char            CUDEFAULT_NODATA_CHAR = -128;
__constant__ static unsigned short   CUDEFAULT_NODATA_USHORT = 65535;
__constant__ static short            CUDEFAULT_NODATA_SHORT = -32768;
__constant__ static unsigned int     CUDEFAULT_NODATA_UINT = 4294967295;
__constant__ static int              CUDEFAULT_NODATA_INT = -2147483648;
__constant__ static unsigned long    CUDEFAULT_NODATA_ULONG = 4294967295;
__constant__ static long             CUDEFAULT_NODATA_LONG = -2147483648;
__constant__ static float            CUDEFAULT_NODATA_FLOAT = -2147483648;
__constant__ static double           CUDEFAULT_NODATA_DOUBLE = -2147483648;
__constant__ static long double      CUDEFAULT_NODATA_LDOUBLE = -2147483648;
__constant__ static float            CUEPSION=1e-6;
__constant__ static float            CUDISPERSION=5.0;
__device__ bool cuisValid(long index,long dimLong);
 template<class DataType>
 __device__ DataType cuGetDataAs(long index,int i,void **p,int oldDataType,long dimLong)
 {
	 DataType val=(DataType)CUERROR_VAL;
	 if(cuisValid(index,dimLong))
	 {
		 switch(oldDataType)
		 {
		 case 0:
			 val= DataType(*((unsigned char *)p[i]+index));
			 break;
		 case 1:
			 val= DataType(*(( char *)p[i]+index));
			 break;
		 case 2:
			 val= DataType(*((unsigned short *)p[i]+index));
			 break;
		 case 3:
			 val= DataType(*((short *)p[i]+index));
			 break;
		 case 4:
			 val= DataType(*((unsigned int *)p[i]+index));
			 break;
		 case 5:
			 val= DataType(*((int *)p[i]+index));
			 break;
		 case 6:
			 val= DataType(*((unsigned long *)p[i]+index));
			 break;
	     case 7:
			 val= DataType(*((long *)p[i]+index));
			 break;
		 case 8:
			 val= DataType(*((float *)p[i]+index));
			 break;
		 case 9:
			 val= DataType(*((double *)p[i]+index));
			 break;
		 case 10:
			 val= DataType(*((double *)p[i]+index));
			 break;
		 case 11:
			 val= DataType(*((bool *)p[i]+index));
			 break;
			
		 }
	 }
	 return val;
 }
 ;
 template<class DataType>
 __device__ void  cuupdateCellAs(long index,int i,void **p,int oldDataType,long dimLong,const DataType &val)
 {
	 if(cuisValid(index,dimLong))
	 {

		 if(oldDataType==0)
		 {
			 unsigned char *pVal= (unsigned char *)p[i]+index;
			 *pVal=(unsigned char)val;
		 }
		 else if(oldDataType==1)
		 {
			 char *pVal= ( char *)p[i]+index;
			 *pVal=(char)val;
		 }
		 else if(oldDataType==2)
		 {
			 unsigned short *pVal= (unsigned short *)p[i]+index;
			 *pVal=(unsigned short)val;
		 }
		 else if(oldDataType==3)
		 {
			 short *pVal=(short *)p[i]+index;
			 *pVal=(short)val;
		 }
		 else if(oldDataType==4)
		 {
			 unsigned int *pVal= (unsigned int *)p[i]+index;
			 *pVal=(unsigned int)val;
		 }
		 else if(oldDataType==5)
		 {
			 int *pVal= (int *)p[i]+index;
			 *pVal=(int)val;
		 }
		 else if(oldDataType==6)
		 {
			 unsigned long *pVal= (unsigned long *)p[i]+index;
			 *pVal=(unsigned long )val;
		 }
		 else if(oldDataType==7)
		 {
			 long *pVal= (long *)p[i]+index;
			 *pVal=( long)val;
		 }
		 else if(oldDataType==8)
		 {
			 float *pVal= (float *)p[i]+index;
			 *pVal=(float)val;
		 }
		 else if(oldDataType==9)
		 {
			 double *pVal= (double *)p[i]+index;
			 *pVal=(double)val;
		 }
		 else if(oldDataType==10)
		 {
			 double *pVal= (double *)p[i]+index;
			 *pVal=(double)val;
		 }
		 else if(oldDataType==11)
		 {
			 bool *pVal= (bool *)p[i]+index;
			 *pVal=(bool )val;
		 }
	 }
 }
#endif
