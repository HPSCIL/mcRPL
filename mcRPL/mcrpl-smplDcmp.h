#ifndef MCRPL_SMPLDCMP_H
#define MCRPL_SMPLDCMP_H

#include "mcrpl-basicTypes.h"
#include "mcrpl-neighborhood.h"
#include "mcrpl-subCellspaceInfo.h"

namespace mcRPL {
  class SmplDcmp{
    public:
      SmplDcmp() {}
      ~SmplDcmp() {}

      bool rowDcmp(list<mcRPL::SubCellspaceInfo> &lSubspcInfos,
                   const mcRPL::SpaceDims &glbDims,
                   int nSubspcs,
                   const mcRPL::Neighborhood *pNbrhd = NULL) const;
      bool colDcmp(list<mcRPL::SubCellspaceInfo> &lSubspcInfos,
                   const mcRPL::SpaceDims &glbDims,
                   int nSubspcs,
                   const mcRPL::Neighborhood *pNbrhd = NULL) const;
      bool blkDcmp(list<mcRPL::SubCellspaceInfo> &lSubspcInfos,
                   const mcRPL::SpaceDims &glbDims,
                   int nRowSubspcs,
                   int nColSubspcs = 1,
                   const mcRPL::Neighborhood *pNbrhd = NULL) const;

    private:
      bool _checkDcmpPrmtrs(list<mcRPL::SubCellspaceInfo> &lSubspcInfos,
                            const mcRPL::SpaceDims &glbDims,
                            int nRowSubspcs, int nColSubspcs,
                            const mcRPL::Neighborhood *pNbrhd) const;

      void _smpl1DDcmp(long &subBegin, long &subEnd,
                       long glbBegin, long glbEnd,
                       int nSubspcs, int iSubspc) const;
  };
};


#endif /* SMPLDCMP_H */
