#include "mcrpl-process.h"

mcRPL::Process::
Process()
  :_comm(MPI_COMM_NULL),
   _id(-1),
   _grpID(-1),
   _nTotalPrcs(-1),
   _hasWriter(false) {
	 
}

void mcRPL::Process::
abort() const {
  MPI_Abort(_comm, -1);
}

void mcRPL::Process::
finalize() const {
  MPI_Finalize();
}

void mcRPL::Process::
sync() const {
  MPI_Barrier(_comm);
}
DeviceProcessor mcRPL::Process::
getDeive()
{
   return _device;
}
const MPI_Comm& mcRPL::Process::
comm() const {
  return _comm;
}

int mcRPL::Process::
id() const {
  return _id;
}

const char* mcRPL::Process::
processorName() const {
  return _prcrName.c_str();
}

int mcRPL::Process::
groupID() const {
  return _grpID;
}

int mcRPL::Process::
nProcesses() const {
  return _nTotalPrcs;
}

bool mcRPL::Process::
hasWriter() const {
  return _hasWriter;
}

bool mcRPL::Process::
isMaster() const {
  return (_id == 0);
}

bool mcRPL::Process::
isWriter() const {
  return (_hasWriter && _id == 1);
}

const mcRPL::IntVect mcRPL::Process::
allPrcIDs(bool incldMaster,
          bool incldWriter) const {
  mcRPL::IntVect vPrcIDs;
  for(int iPrc = 0; iPrc < _nTotalPrcs; iPrc++) {
    if(iPrc == 0 && !incldMaster) {
      continue;
    }
    if(_hasWriter && iPrc == 1 && !incldWriter) {
      continue;
    }
    vPrcIDs.push_back(iPrc);
  }
  return vPrcIDs;
}

bool mcRPL::Process::
initialized() const{
  int mpiStarted;
  MPI_Initialized(&mpiStarted);
  return static_cast<bool>(mpiStarted);
}

bool mcRPL::Process::
active() const {
  return (_comm != MPI_COMM_NULL &&
          _id != -1 &&
          _nTotalPrcs != -1);
}
bool mcRPL::Process::initCuda()
{
	MPI_Status status;
	if(!_device.Valid()&&!isMaster())
	{
		return false;
	}
	else
	{
	if(this->isMaster())
	{
		map<string,mcRPL::IntVect>mstrdevcount;
		map<string,mcRPL::IntVect>mstr;
		map<string,mcRPL::IntVect>::iterator itmstr;
		map<string,mcRPL::IntVect>::iterator itdevcount;
		mstr.insert(pair<string,mcRPL::IntVect>(_prcrName,mcRPL::IntVect()));
		mstr.begin()->second.push_back(0);
		mstrdevcount.insert(pair<string,mcRPL::IntVect>(_prcrName,_device.allDevice()));
		for(int i=1;i<_nTotalPrcs;i++)
		{
			char cname_recv[100];
			int devcount_recv;

			MPI_Recv(cname_recv,100,MPI_CHAR,i,i,_comm,&status);
			MPI_Recv(&devcount_recv,1,MPI_INT,i,2*i,_comm,&status);
			mcRPL::IntVect vavalible_recv(devcount_recv);
			MPI_Recv(&vavalible_recv[0],devcount_recv,MPI_INT,i,3*i,_comm,&status);
			string strname=cname_recv;
			//  if(i==2)
			//	  strname="hp2";
			itmstr=mstr.find(strname);
			if(itmstr==mstr.end())
			{
				mcRPL::IntVect vnpocesses;
				vnpocesses.push_back(i);
				mstr.insert(pair<string,mcRPL::IntVect>(strname, vnpocesses));
				vnpocesses.clear();
				mstrdevcount.insert(pair<string,mcRPL::IntVect>(strname,vavalible_recv));
			}
			else
			{
				itmstr->second.push_back(i);
			}
		}
		itdevcount=mstrdevcount.begin();
		for(itmstr=mstr.begin();itmstr!=mstr.end();itmstr++)
		{
		     int nMasterDevice=0; 
			cout<<itmstr->first<<" has process :";
			mcRPL::IntVect vnPoID=itmstr->second;
			mcRPL::IntVect vnPoID2=itdevcount->second;
			int nPo=vnPoID.size();
			for(int i=0;i<nPo;i++)
			{
				cout<<vnPoID[i]<<" ";
				int nSend=i%vnPoID2.size();
				if(vnPoID[i]!=0)
				{
				MPI_Send(&vnPoID2[nSend],1,MPI_INT,vnPoID[i],4*vnPoID[i],_comm);
				}
				else
				{
				    
					_device.setDeviceID(vnPoID2[nSend]);
					nMasterDevice=nSend;
					_device.setdevice();

				}
				if(i==(nPo-1))
				{
				cout<<endl;
				cout<<" the device of process 0"<<" uses is:"<<_prcrName.c_str()<<" "<<nMasterDevice<<endl;  
				}
			}
			cout<<endl;
			cout<<"the devices IDs of the  node "<<itmstr->first<<" has:";
			for(int i=0;i<vnPoID2.size();i++)
			{
				cout<<vnPoID2[i]<<" ";
			}
			cout<<endl;
			itdevcount++;
		}

	}
	else
	{
		const char *name=_prcrName.c_str();
		MPI_Send(name,100,MPI_CHAR,0,_id,_comm);
		int nCount=_device.allDevice().size();
		MPI_Send(&nCount,1,MPI_INT,0,2*_id,_comm);
		MPI_Send(&_device.allDevice()[0],nCount,MPI_INT,0,3*_id,_comm);
		int nRecv;
		MPI_Recv(&nRecv,1,MPI_INT,0,4*_id,_comm,&status);
		cout<<" the device of process "<<_id<<" uses is:"<<name<<" "<<nRecv<<endl; 
		_device.setDeviceID(nRecv);
		_device.setdevice();
	}
	
	  return true;
	}
}
bool mcRPL::Process::
init(int argc,
     char *argv[]) {
  bool done = true;
  if(_comm == MPI_COMM_NULL) {
    return done;
  }

  if(!initialized()) {
    MPI_Init(&argc, &argv);
  }
  MPI_Comm_rank(_comm, &_id);
  MPI_Comm_size(_comm, &_nTotalPrcs);
  
//  cudaSetDevice(_id);
  char aPrcrName[MPI_MAX_PROCESSOR_NAME];
  int prcrNameLen;
  MPI_Get_processor_name(aPrcrName, &prcrNameLen);
  _prcrName.assign(aPrcrName, aPrcrName+prcrNameLen);
  this->_device.setDeviceOn(_prcrName);
  
  if(_nTotalPrcs == 2 && _hasWriter) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Warning: only TWO Processes have been initialized," \
         << " including a master Process and a writer Process." \
         << " The writer Process will NOT participate in the computing when Task Farming."\
         << " And the writer Process will NOT participate in the computing in all cases." \
         << endl;
  }
  else if(_nTotalPrcs == 1) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Warning: only ONE Process has been initialized," \
         << " including a master Process." \
         << " The master Process will NOT participate in the computing when Task Farming."\
         << endl;
    if(_hasWriter) {
      cerr << __FILE__ << " " << __FUNCTION__ \
           << " Error: unable to initialize an writer Process" \
           << endl;
      done = false;
    }
  }

  return (done&&initCuda());
}

bool mcRPL::Process::
set(MPI_Comm &comm,
    bool hasWriter,
    int groupID) {
  _comm = comm;
  _hasWriter = hasWriter;
  _grpID = groupID;
  return(init());
}

bool mcRPL::Process::
grouping(int nGroups,
         bool incldMaster,
         Process *pGrpedPrc,
         Process *pGrpMaster) const {
  if(!initialized()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: Process has NOT been initialized," \
         << " unable to be grouped" << endl;
    return false;
  }

  if(!active()) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: inactive Process," \
         << " unable to group a Null communicator." \
         << " id = " << _id << " nTotPrcs = " << _nTotalPrcs << endl;
    return false;
  }

  if(nGroups <= 0 ||
     nGroups > _nTotalPrcs) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error: invalid number of groups (" \
         << nGroups << ") as the total number of processes is " \
         << _nTotalPrcs << endl;
    return false;
  }

  if(!incldMaster && _nTotalPrcs <= 1) {
    cerr << __FILE__ << " " << __FUNCTION__ \
         << " Error:  " << _nTotalPrcs << " processes can NOT" \
         << " be grouped without the master process" << endl;
    return false;
  }

  MPI_Group glbGrp;
  MPI_Comm glbComm = _comm;
  MPI_Comm_group(glbComm, &glbGrp);
  int myID = -1;
  int grpID = -1;
  MPI_Comm grpComm = MPI_COMM_NULL;

  if(incldMaster) {
    myID = _id;
    grpID = myID % nGroups;
    MPI_Comm_split(glbComm, grpID, myID, &grpComm);
    if(!pGrpedPrc->set(grpComm, _hasWriter, grpID)) {
      return false;
    }
    if(pGrpMaster != NULL) {
      MPI_Group masterGrp= MPI_GROUP_NULL;
      MPI_Comm masterComm = MPI_COMM_NULL;
      int grpMasterRange[1][3] = {{0, nGroups-1, 1}};
      MPI_Group_range_incl(glbGrp, 1, grpMasterRange, &masterGrp);
      MPI_Comm_create(glbComm, masterGrp, &masterComm);
      if(!pGrpMaster->set(masterComm)) {
        return false;
      }
    }
  }
  else {
    int excldRanks[1] = {0};
    MPI_Group glbGrp2 = MPI_GROUP_NULL;
    MPI_Group_excl(glbGrp, 1, excldRanks, &glbGrp2);
    MPI_Comm_create(_comm, glbGrp2, &glbComm);
    glbGrp = glbGrp2;
    if(!isMaster()) {
      MPI_Comm_rank(glbComm, &myID);
      grpID = myID % nGroups;
      MPI_Comm_split(glbComm, grpID, myID, &grpComm);
      if(!pGrpedPrc->set(grpComm, _hasWriter, grpID)) {
        return false;
      }
      if(pGrpMaster != NULL) {
        MPI_Group masterGrp= MPI_GROUP_NULL;
        MPI_Comm masterComm = MPI_COMM_NULL;
        int grpMasterRange[1][3] = {{0, nGroups-1, 1}};
        MPI_Group_range_incl(glbGrp, 1, grpMasterRange, &masterGrp);
        MPI_Comm_create(glbComm, masterGrp, &masterComm);
        if(!pGrpMaster->set(masterComm)) {
          return false;
        }
      }
    }
  }

  return true;
}

