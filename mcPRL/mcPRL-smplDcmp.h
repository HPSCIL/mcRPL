#ifndef mcPRL_SMPLDCMP_H
#define mcPRL_SMPLDCMP_H

#include "mcPRL-basicTypes.h"
#include "mcPRL-neighborhood.h"
#include "mcPRL-subCellspaceInfo.h"

namespace mcPRL {
  class SmplDcmp{
    public:
      SmplDcmp() {}
      ~SmplDcmp() {}

      bool rowDcmp(list<mcPRL::SubCellspaceInfo> &lSubspcInfos,
                   const mcPRL::SpaceDims &glbDims,
                   int nSubspcs,
                   const mcPRL::Neighborhood *pNbrhd = NULL) const;
      bool colDcmp(list<mcPRL::SubCellspaceInfo> &lSubspcInfos,
                   const mcPRL::SpaceDims &glbDims,
                   int nSubspcs,
                   const mcPRL::Neighborhood *pNbrhd = NULL) const;
      bool blkDcmp(list<mcPRL::SubCellspaceInfo> &lSubspcInfos,
                   const mcPRL::SpaceDims &glbDims,
                   int nRowSubspcs,
                   int nColSubspcs = 1,
                   const mcPRL::Neighborhood *pNbrhd = NULL) const;

    private:
      bool _checkDcmpPrmtrs(list<mcPRL::SubCellspaceInfo> &lSubspcInfos,
                            const mcPRL::SpaceDims &glbDims,
                            int nRowSubspcs, int nColSubspcs,
                            const mcPRL::Neighborhood *pNbrhd) const;

      void _smpl1DDcmp(long &subBegin, long &subEnd,
                       long glbBegin, long glbEnd,
                       int nSubspcs, int iSubspc) const;
  };
};


#endif /* SMPLDCMP_H */
