#ifndef MCRPL_OWNERSHIPMAP_H
#define MCRPL_OWNERSHIPMAP_H

#include "mcrpl-basicTypes.h"

namespace mcRPL {
  class OwnershipMap {
    public:
      OwnershipMap();
      ~OwnershipMap() {}

      void init(const mcRPL::IntVect &vSubspcIDs,
                const mcRPL::IntVect &vPrcIDs);
      void clearMapping();

      void mapping(mcRPL::MappingMethod mapMethod = mcRPL::CYLC_MAP,
                   int nSubspcs2Map = 0);
      int mappingNextTo(int prcID); // return the SubCellspace's ID
      bool mappingTo(int subspcID,
                     int prcID);
      bool allMapped() const;

      const mcRPL::IntVect* subspcIDsOnPrc(int prcID) const;
      mcRPL::IntVect mappedSubspcIDs() const;
      int findPrcBySubspcID(int subspcID) const;
      bool findSubspcIDOnPrc(int subspcID, int prcID) const;

      bool map2buf(vector<char> &buf) const;
      bool buf2map(const vector<char> &buf);

    private:
      mcRPL::IntVect _vSubspcIDs;
      map<int, mcRPL::IntVect> _mPrcSubspcs;

      mcRPL::IntVect::iterator _itrSubspc2Map;

    friend ostream& operator<<(ostream &os, const mcRPL::OwnershipMap &mOwnerships);
  };

  ostream& operator<<(ostream &os, const mcRPL::OwnershipMap &mOwnerships);
};


#endif /* OwnershipMap_H */
