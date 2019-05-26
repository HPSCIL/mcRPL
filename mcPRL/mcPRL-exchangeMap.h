#ifndef mcPRL_EXCHANGEMAP_H
#define mcPRL_EXCHANGEMAP_H

#include "mcPRL-basicTypes.h"

namespace mcPRL {
  typedef struct {
    int subspcGlbID;
    int iDirection;
    int iNeighbor;
  } ExchangeNode;

  class ExchangeMap: public map<int, vector<mcPRL::ExchangeNode> > {
    public:
      bool add(int subspcID, int iDir, int iNbr, int prcID);
  };

};

#endif
