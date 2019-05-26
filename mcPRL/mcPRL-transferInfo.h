#ifndef mcPRL_TRANSFERINFO_H
#define mcPRL_TRANSFERINFO_H

#include "mcPRL-basicTypes.h"
#include "mcPRL-transition.h"

namespace mcPRL {
  class TransferInfo {
    public:
      TransferInfo();
      TransferInfo(int fromPrcID,
                   int toPrcID,
                   const string &lyrName,
                   int subspcGlbID = mcPRL::ERROR_ID,
                   mcPRL::TransferDataType tDataType = mcPRL::SPACE_TRANSFDATA);
      ~TransferInfo() {}

      int fromPrcID() const;
      int toPrcID() const;
      const string& lyrName() const;
      int subspcGlbID() const;
      mcPRL::TransferDataType transfDataType() const;
      bool completed() const;

      void complete();
      void restart();

    protected:
      int _fromPrcID;
      int _toPrcID;
      string _lyrName;
      int _subspcGlbID;
      mcPRL::TransferDataType _tDataType; // SPACE_TRANSF or CELL_TRANSF
      bool _cmplt;
  };

  class TransferInfoVect: public vector<mcPRL::TransferInfo> {
    public:
      const mcPRL::TransferInfo* findInfo(const string &lyrName,
                                         int subspcGlbID) const;

      /* Master checks if a Layer's Cellspace has been broadcasted */
      bool checkBcastCellspace(int prcID,
                               const string &lyrName) const;

  };
};

#endif

