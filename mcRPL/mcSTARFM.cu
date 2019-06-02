#include"mcSTARFM.h"
__device__ void cuSTARFM::operator()(void **focalIn,int *DataInType,void **focalOut,int *DataOutType,int nIndex,nbrInfo<double>nbrinfo,rasterInfo rasterinfo,double *para)
	{ 
		short *pB=(short*)focalIn[0]+nIndex;
		 int *pC=(int*)focalIn[1]+nIndex;
		 int *pD=(int*)focalIn[2]+nIndex;
		 int *pA=(int*)focalOut[0]+nIndex;
		double fresult=0;
		int njudge=0;
		double fwei=0,fweih,fdis;
		int fkk=0;
		double fsum1=0,fsum2=0;
		double fst=0.0;
		int nwidth = rasterinfo.width;
		int nheight = rasterinfo.height;
		int nrow=nIndex/nwidth;
		int ncolumn=nIndex%nwidth;
		double fr_center_LM=*pD-*pB+0.002;
		double fr_center_MM=*pD-*pC+1.412*0.005;
	    double fr_LM,fr_MM;
		int nbrsize = nbrinfo.nbrsize;
		int* nbrcood = nbrinfo.nbrcood;
		short test=cuGetDataAs<short>(nIndex,0,focalIn,DataInType[0],nwidth*nheight);
		for(int i = 0; i < nbrsize; i++)
		{
			int cx=ncolumn+ nbrcood[i * 2];
			int cy=nrow+nbrcood[i * 2 + 1];
			if (cx < 0 || cx >= nwidth || cy < 0 || cy >= nheight)
			{
				continue;
			}
			//if(nbrcood[i * 2]==0&&nbrcood[i * 2 + 1]==0)
            //{
             //  continue;
            //}
			double nBcurnbr = pB[nbrcood[i * 2] + nbrcood[i * 2 + 1] * nwidth];
			fsum1+=nBcurnbr*nBcurnbr;
			fsum2+=nBcurnbr;
		}
		 fst=sqrt(fsum1/nbrsize-(fsum2/nbrsize)*(fsum2/nbrsize))/4;
		for(int i = 0; i < nbrsize; i++)
		{
			int cx=ncolumn+ nbrcood[i * 2];
			int cy=nrow+nbrcood[i * 2 + 1];
			if (cx < 0 || cx >= nwidth || cy < 0 || cy >= nheight)
			{
				continue;
			}
			int ncurnbr=nbrcood[i * 2] + nbrcood[i * 2 + 1] * nwidth;
			if(abs(*pB-pB[ncurnbr])<fst)
			{
				fr_LM=pD[ncurnbr]-pB[ncurnbr];
				fr_MM=pD[ncurnbr]-pC[ncurnbr];
				if((fr_center_LM>0&&fr_LM<fr_center_LM)||(fr_center_LM<0&&fr_LM>fr_center_LM))
				{
					if((fr_center_MM>0&&fr_MM<fr_center_MM)||(fr_center_MM<0&&fr_MM>fr_center_MM))
					{
						fr_LM=fabs(fr_LM)+0.0001;
						fr_MM=fabs(fr_MM)+0.0001;
						if(ncurnbr==0)
							njudge=1;
						fdis=double(nbrcood[i*2]*nbrcood[i*2]+nbrcood[i*2+1]*nbrcood[i*2+1]);
						fdis=sqrt(fdis)/25+1.0;
						fweih=1.0/(fdis*fr_LM*fr_MM);
						fwei+=fweih;
						fresult+=fweih*(pC[ncurnbr]+pB[ncurnbr]-pD[ncurnbr]);
						fkk++;
					}
				}
			}

		}
		if(fkk==0)
		{
			*pA=abs(double(*pB+*pC-*pD));
			fwei=10000;
		}
		else
		{
			if(njudge==0)
			{
				fdis=1.0;
				fr_LM=abs(double(*pD-*pB))+0.0001;
				fr_MM=abs(double(*pD-*pC))+0.0001;
				fweih=1.0/(fdis*fr_LM*fr_MM);
				fresult+=fweih*(*pB+*pC-*pD);
				fwei+=fweih;
			}
			*pA=fresult/fwei;
		}
	}