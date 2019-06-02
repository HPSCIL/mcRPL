#ifndef MCRPL_CELLSPACEGEOINFO_H
#define MCRPL_CELLSPACEGEOINFO_H

#include "mcrpl-basicTypes.h"
#include "mcrpl-subCellspaceInfo.h"

namespace mcRPL {
  class CellspaceGeoinfo {
    public:
      CellspaceGeoinfo() {}
      CellspaceGeoinfo(const mcRPL::GeoCoord &nwCorner,
                       const mcRPL::GeoCoord &cellSize,
                       const string &projection);
      CellspaceGeoinfo(const mcRPL::CellspaceGeoinfo &rhs);
      ~CellspaceGeoinfo() {}

      mcRPL::CellspaceGeoinfo& operator=(const mcRPL::CellspaceGeoinfo &rhs);
      bool operator==(const mcRPL::CellspaceGeoinfo &rhs) const;

      void add2Buf(vector<char> &buf);
      bool initFromBuf(vector<char> &buf);

      void nwCorner(const mcRPL::GeoCoord &nw);
      void nwCorner(double x, double y);
      void cellSize(const mcRPL::GeoCoord &size);
      void cellSize(double xSize, double ySize);
      void projection(const string &proj);

      const mcRPL::GeoCoord& nwCorner() const;
      const mcRPL::GeoCoord& cellSize() const;

      const string& projection() const;
      void geoTransform(double aGeoTransform[6]) const;

      mcRPL::GeoCoord cellCoord2geoCoord(const mcRPL::CellCoord &cCoord) const;
      mcRPL::GeoCoord cellCoord2geoCoord(long iRow, long iCol) const;
      mcRPL::CellCoord geoCoord2cellCoord(const mcRPL::GeoCoord &gCoord) const;
      mcRPL::CellCoord geoCoord2cellCoord(double x, double y) const;

	  /*--------GDAL------------*/
      bool initByGDAL(GDALDataset *pDataset,
                      bool warning = true);
	  /*---------PGTIOL--------*/
	  bool initByPGTIOL(PGTIOLDataset *pDataset,
                      bool warning = true);
	  
      mcRPL::CellspaceGeoinfo subSpaceGeoinfo(const mcRPL::SubCellspaceInfo &subInfo) const;

    private:
      mcRPL::GeoCoord _nwCorner;
      mcRPL::GeoCoord _cellSize;
      string _projection;
  };  
};

#endif
