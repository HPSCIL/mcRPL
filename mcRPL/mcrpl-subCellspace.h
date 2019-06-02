#ifndef MCRPL_SUBCELLSPACE_H
#define MCRPL_SUBCELLSPACE_H

#include "mcrpl-cellspace.h"
#include "mcrpl-subCellspaceInfo.h"
#include "mcrpl-cellStream.h"

namespace mcRPL {
  class SubCellspace: public mcRPL::Cellspace {
    public:
      /* Constructors and Destructor */
      SubCellspace();
      SubCellspace(mcRPL::CellspaceInfo *pInfo,
                   mcRPL::SubCellspaceInfo *pSubInfo);
      SubCellspace(const mcRPL::SubCellspace &rhs);
      ~SubCellspace();

      /* Operators */
      mcRPL::SubCellspace& operator=(const mcRPL::SubCellspace &rhs);

      mcRPL::SubCellspaceInfo* subInfo();
      const mcRPL::SubCellspaceInfo* subInfo() const;

      bool add2UpdtStream(mcRPL::CellStream &updtStream,
                          int iDir,
                          int iNbr = 0) const;
      bool loadUpdtStream(const mcRPL::CellStream &updtStream);

    protected:
      mcRPL::SubCellspaceInfo *_pSubInfo;
  };
};

#endif
