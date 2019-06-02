#ifndef MCRPL_EXCHANGEMAP_H
#define MCRPL_EXCHANGEMAP_H

#include "mcrpl-basicTypes.h"

namespace mcRPL {
  typedef struct {
    int subspcGlbID;
    int iDirection;
    int iNeighbor;
  } ExchangeNode;

  class ExchangeMap: public map<int, vector<mcRPL::ExchangeNode> > {
    public:
      bool add(int subspcID, int iDir, int iNbr, int prcID);
  };

};

#endif
