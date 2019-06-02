#ifndef MCRPL_LAYER_H
#define MCRPL_LAYER_H

#include "mcrpl-basicTypes.h"
#include "mcrpl-cellspace.h"
#include "mcrpl-subCellspace.h"
#include "mcrpl-smplDcmp.h"

namespace mcRPL {
  class Layer {
    public:
      /* Constructors and destructor */
      Layer(const char *aName = NULL);
      ~Layer();
      
      /* Information */
      const char* name() const;
      void name(const char *aName);
      const mcRPL::SpaceDims* glbDims() const;
      const char* dataType() const;
      size_t dataSize() const;
      long tileSize() const;
	  
      const mcRPL::CellspaceGeoinfo * glbGeoinfo() const;

      /* GDAL */
      bool openGdalDS(const char *aFileName,
                      GDALAccess ioOption = GA_ReadOnly);
      int dsBand() const;
      void dsBand(int iBand);
      void closeGdalDS();
      bool createGdalDS(const char *aFileName,
                        const char *aGdalFormat,
                        char **aGdalOptions = NULL);
      /*PGTIOL*/
      bool openPgtiolDS(const char *aFileName,
	                      int prc,
                        PGTIOLAccess ioOption = PG_ReadOnly);
      void closePgtiolDS();
      bool createPgtiolDS(const char *aFileName,
	                        int _prc,
                          char **aPgtOptions = NULL);
      
      /* Decompose & Distribution */
      bool decompose(int nRowSubspcs,
                     int nColSubspcs,
                     const mcRPL::Neighborhood *pNbrhd = NULL);
      bool copyDcmp(const Layer *pFromLyr);
      bool isDecomposed() const;

      /* CellspaceInfo */
      void initCellspaceInfo();
      void initCellspaceInfo(const mcRPL::SpaceDims &dims,
                             const char *aTypeName,
                             size_t typeSize,
                             const mcRPL::CellspaceGeoinfo *pGeoinfo = NULL,
                             long  tileSize = TILEWIDTH);
      void initCellspaceInfo(const mcRPL::CellspaceInfo &cellspcInfo);
      
	    bool initCellspaceInfoByGDAL();
	    bool initCellspaceInfoByPGTIOL();

      void delCellspaceInfo();
      mcRPL::CellspaceInfo* cellspaceInfo();
      const mcRPL::CellspaceInfo* cellspaceInfo() const;

      /* Cellspace */
      bool createCellspace(double *pInitVal = NULL);
	    bool loadCellspaceByGDAL();
      bool writeCellspaceByGDAL();
	    bool loadCellspaceByPGTIOL();
      bool writeCellspaceByPGTIOL();

      void delCellspace();
      mcRPL::Cellspace* cellspace();
      const mcRPL::Cellspace* cellspace() const;

      /* SubCellspaceInfo */
      int nSubCellspaceInfos() const;
      const list<mcRPL::SubCellspaceInfo>* allSubCellspaceInfos() const;
      bool delSubCellspaceInfo_glbID(int glbID);
      bool delSubCellspaceInfo_lclID(int lclID);
      void clearSubCellspaceInfos();
      const mcRPL::IntVect allSubCellspaceIDs() const;
      int maxSubCellspaceID() const;
      bool calcAllBRs(bool onlyUpdtCtrCell,
                      const mcRPL::Neighborhood *pNbrhd);
      
      mcRPL::SubCellspaceInfo* subCellspaceInfo_glbID(int glbID,
                                                     bool warning = false);
      const mcRPL::SubCellspaceInfo* subCellspaceInfo_glbID(int glbID,
                                                           bool warning = false) const;
      mcRPL::SubCellspaceInfo* subCellspaceInfo_lclID(int lclID,
                                                     bool warning = false);
      const mcRPL::SubCellspaceInfo* subCellspaceInfo_lclID(int lclID,
                                                           bool warning = false) const;
      
      /* SubCellspace */
      int nSubCellspaces() const;
      mcRPL::SubCellspace* addSubCellspace(int glbID,
                                          double *pInitVal = NULL);
      bool delSubCellspace_glbID(int glbID);
      bool delSubCellspace_lclID(int lclID);
      void clearSubCellspaces();
      
      mcRPL::SubCellspace* subCellspace_glbID(int glbID,
                                             bool warning = false);
      const mcRPL::SubCellspace* subCellspace_glbID(int glbID,
                                                   bool warning = false) const;
      mcRPL::SubCellspace* subCellspace_lclID(int lclID,
                                             bool warning = false);
      const mcRPL::SubCellspace* subCellspace_lclID(int lclID,
                                                   bool warning = false) const;
      
      void setUpdateTracking(bool toTrack);
      void allUpdatedIdxs(vector<long> &vUpdtGlbIdxs) const;
      void clearUpdateTracks();
      bool loadCellStream(void *aStream, int nCells);

      /*ByGDAL------------*/
	  bool  loadSubCellspaceByRAND(int glbID);              //2018Ëæ»úÉú³ÉÍ¼²ã
      bool loadSubCellspaceByGDAL(int glbID);
      bool writeSubCellspaceByGDAL(int glbID);

      bool writeTmpSubCellspaceByGDAL(int glbID,
                                      int alias = 0);
      bool loadTmpSubCellspaceByGDAL(int glbID,
                                     int alias = 0);
      bool mergeSubCellspaceByGDAL(int glbID,
                                   int alias = 0);
      

      /*-------byPGTIOL--------------------*/
      bool loadSubCellspaceByPGTIOL(int glbID);
      bool writeSubCellspaceByPGTIOL(int glbID);

      /*
      bool writeTmpSubCellspaceByPGTIOL(int glbID,
                                        int alias = 0);
      bool loadTmpSubCellspaceByPGTIOL(int glbID,
                                       int alias = 0);
      bool mergeSubCellspaceByPGTIOL(int glbID,
                                     int alias = 0);
      */

    private:
      const string _tmpSubCellspaceFileName(int glbID,
                                            int alias = 0) const;
      bool _parseDsName(const char *aFileName);

    private:
      string _name;

      mcRPL::CellspaceInfo *_pCellspcInfo;
      mcRPL::Cellspace *_pCellspc;

      list<mcRPL::SubCellspaceInfo> _lSubInfos;
      list<mcRPL::CellspaceInfo> _lCellspcInfos;
      list<mcRPL::SubCellspace> _lSubCellspcs;

      GDALDataset *_pGdalDS;
	    PGTIOLDataset *_pPgtiolDS;
      int _iBand;
      string _dsPath;
      string _dsName;
  };
};

#endif
