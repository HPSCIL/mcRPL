#ifndef CUENVCONTROL_H_H
#define CUENVCONTROL_H_H


//#include "CuPRL.h"

namespace mcRPL
{
	class CuEnvControl
	{
	public:

		static void setCudaLayerAddState(bool state){ m_cudaLayerAddState = state; };
		static bool getCudaLayerAddState(){ return m_cudaLayerAddState; };

		static void setCudaLayerSubState(bool state){ m_cudaLayerSubState = state; };
		static bool getCudaLayerSubState(){ return m_cudaLayerSubState; };

		static void setCudaLayerMulState(bool state){ m_cudaLayerMulState = state; };
		static bool getCudaLayerMulState(){ return m_cudaLayerMulState; };

		static void setCudaLayerDivState(bool state){ m_cudaLayerDivState = state; };
		static bool getCudaLayerDivState(){ return m_cudaLayerDivState; };


		static void setBlock2Dim(int blockdim1, int blockdim2){ m_blockdim1 = blockdim1; m_blockdim2 = blockdim2; };
		static int getBlockDim1(){ return m_blockdim1; };
		static int getBlockDim2(){ return m_blockdim2; };
		static int getBlockDim(){ return m_blockdim; };

		static dim3 getBlock2D(){ dim3 cudablock(m_blockdim1, m_blockdim2); return cudablock; };
		static dim3 getBlock1D(){ dim3 cudablock(m_blockdim); return cudablock; };
		static dim3 getGrid(int width, int height){ dim3 cudagrid(width% m_blockdim1 == 0 ? width / m_blockdim1 : width / m_blockdim1 + 1, height % m_blockdim2 == 0 ? height / m_blockdim2 : height / m_blockdim2 + 1); return cudagrid; };
		static dim3 getGrid(int nTask){ dim3 cudagrid(nTask%m_blockdim == 0 ? nTask / m_blockdim : nTask / m_blockdim + 1); };

	private:
		static int m_blockdim1;
		static int m_blockdim2;
		static int m_blockdim;

		static bool m_cudaLayerAddState;
		static bool m_cudaLayerSubState;
		static bool m_cudaLayerMulState;
		static bool m_cudaLayerDivState;

	};

	bool CuEnvControl::m_cudaLayerAddState = false;
	bool CuEnvControl::m_cudaLayerSubState = false;
	bool CuEnvControl::m_cudaLayerMulState = false;
	bool CuEnvControl::m_cudaLayerDivState = false;
	int CuEnvControl::m_blockdim1 = 16;
	int CuEnvControl::m_blockdim2 = 16;

	int CuEnvControl::m_blockdim = 256;

}

#endif
