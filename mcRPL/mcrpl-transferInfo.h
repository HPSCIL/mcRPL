#ifndef MCRPL_TRANSFERINFO_H
#define MCRPL_TRANSFERINFO_H

#include "mcrpl-basicTypes.h"
#include "mcrpl-transition.h"

namespace mcRPL {
  class TransferInfo {
    public:
      TransferInfo();
      TransferInfo(int fromPrcID,
                   int toPrcID,
                   const string &lyrName,
                   int subspcGlbID = mcRPL::ERROR_ID,
                   mcRPL::TransferDataType tDataType = mcRPL::SPACE_TRANSFDATA);
      ~TransferInfo() {}

      int fromPrcID() const;
      int toPrcID() const;
      const string& lyrName() const;
      int subspcGlbID() const;
      mcRPL::TransferDataType transfDataType() const;
      bool completed() const;

      void complete();
      void restart();

    protected:
      int _fromPrcID;
      int _toPrcID;
      string _lyrName;
      int _subspcGlbID;
      mcRPL::TransferDataType _tDataType; // SPACE_TRANSF or CELL_TRANSF
      bool _cmplt;
  };

  class TransferInfoVect: public vector<mcRPL::TransferInfo> {
    public:
      const mcRPL::TransferInfo* findInfo(const string &lyrName,
                                         int subspcGlbID) const;

      /* Master checks if a Layer's Cellspace has been broadcasted */
      bool checkBcastCellspace(int prcID,
                               const string &lyrName) const;

  };
};

#endif

