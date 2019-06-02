#ifndef MCRPL_PROCESS_H
#define MCRPL_PROCESS_H
#define MPICH_SKIP_MPICXX
#include "mcrpl-basicTypes.h"
#include "mpi.h"
#include "mcrpl-DeviceInfo.h"
namespace mcRPL {
  class Process {
    public:
      Process();
      ~Process() {}

      bool initialized() const;
      bool active() const;
      bool init(int argc = 0,
                char *argv[] = NULL);
	  bool initCuda();
      void abort() const;
      void finalize() const;
      void sync() const;
	  DeviceProcessor getDeive();
      bool set(MPI_Comm &comm,
               bool hasWriter = false,
               int groupID = -1);
      bool grouping(int nGroups,
                    bool incldMaster,
                    Process *pGrpedPrc,
                    Process *pGrpMaster = NULL) const;

      const MPI_Comm& comm() const;
      int id() const;
      const char* processorName() const;
      int groupID() const;
      int nProcesses() const;
      bool hasWriter() const;
      bool isMaster() const;
      bool isWriter() const;
      const mcRPL::IntVect allPrcIDs(bool incldMaster = false,
                                    bool incldWriter = false) const;

    private:
      MPI_Comm _comm;
      int _id;
      int _grpID;
      int _nTotalPrcs;
      bool _hasWriter;
      string _prcrName;
	  DeviceProcessor _device;
  };
};


#endif /* PROCESS_H */
