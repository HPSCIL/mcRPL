#ifndef mcPRL_OWNERSHIPMAP_H
#define mcPRL_OWNERSHIPMAP_H

#include "mcPRL-basicTypes.h"

namespace mcPRL {
  class OwnershipMap {
    public:
      OwnershipMap();
      ~OwnershipMap() {}

      void init(const mcPRL::IntVect &vSubspcIDs,
                const mcPRL::IntVect &vPrcIDs);
      void clearMapping();

      void mapping(mcPRL::MappingMethod mapMethod = mcPRL::CYLC_MAP,
                   int nSubspcs2Map = 0);
      int mappingNextTo(int prcID); // return the SubCellspace's ID
      bool mappingTo(int subspcID,
                     int prcID);
      bool allMapped() const;

      const mcPRL::IntVect* subspcIDsOnPrc(int prcID) const;
      mcPRL::IntVect mappedSubspcIDs() const;
      int findPrcBySubspcID(int subspcID) const;
      bool findSubspcIDOnPrc(int subspcID, int prcID) const;

      bool map2buf(vector<char> &buf) const;
      bool buf2map(const vector<char> &buf);

    private:
      mcPRL::IntVect _vSubspcIDs;
      map<int, mcPRL::IntVect> _mPrcSubspcs;

      mcPRL::IntVect::iterator _itrSubspc2Map;

    friend ostream& operator<<(ostream &os, const mcPRL::OwnershipMap &mOwnerships);
  };

  ostream& operator<<(ostream &os, const mcPRL::OwnershipMap &mOwnerships);
};


#endif /* OwnershipMap_H */
