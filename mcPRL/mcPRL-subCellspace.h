#ifndef mcPRL_SUBCELLSPACE_H
#define mcPRL_SUBCELLSPACE_H

#include "mcPRL-cellspace.h"
#include "mcPRL-subCellspaceInfo.h"
#include "mcPRL-cellStream.h"

namespace mcPRL {
  class SubCellspace: public mcPRL::Cellspace {
    public:
      /* Constructors and Destructor */
      SubCellspace();
      SubCellspace(mcPRL::CellspaceInfo *pInfo,
                   mcPRL::SubCellspaceInfo *pSubInfo);
      SubCellspace(const mcPRL::SubCellspace &rhs);
      ~SubCellspace();

      /* Operators */
      mcPRL::SubCellspace& operator=(const mcPRL::SubCellspace &rhs);

      mcPRL::SubCellspaceInfo* subInfo();
      const mcPRL::SubCellspaceInfo* subInfo() const;

      bool add2UpdtStream(mcPRL::CellStream &updtStream,
                          int iDir,
                          int iNbr = 0) const;
      bool loadUpdtStream(const mcPRL::CellStream &updtStream);

    protected:
      mcPRL::SubCellspaceInfo *_pSubInfo;
  };
};

#endif
