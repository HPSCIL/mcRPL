#ifndef mcPRL_CELLSPACEGEOINFO_H
#define mcPRL_CELLSPACEGEOINFO_H

#include "mcPRL-basicTypes.h"
#include "mcPRL-subCellspaceInfo.h"

namespace mcPRL {
  class CellspaceGeoinfo {
    public:
      CellspaceGeoinfo() {}
      CellspaceGeoinfo(const mcPRL::GeoCoord &nwCorner,
                       const mcPRL::GeoCoord &cellSize,
                       const string &projection);
      CellspaceGeoinfo(const mcPRL::CellspaceGeoinfo &rhs);
      ~CellspaceGeoinfo() {}

      mcPRL::CellspaceGeoinfo& operator=(const mcPRL::CellspaceGeoinfo &rhs);
      bool operator==(const mcPRL::CellspaceGeoinfo &rhs) const;

      void add2Buf(vector<char> &buf);
      bool initFromBuf(vector<char> &buf);

      void nwCorner(const mcPRL::GeoCoord &nw);
      void nwCorner(double x, double y);
      void cellSize(const mcPRL::GeoCoord &size);
      void cellSize(double xSize, double ySize);
      void projection(const string &proj);

      const mcPRL::GeoCoord& nwCorner() const;
      const mcPRL::GeoCoord& cellSize() const;

      const string& projection() const;
      void geoTransform(double aGeoTransform[6]) const;

      mcPRL::GeoCoord cellCoord2geoCoord(const mcPRL::CellCoord &cCoord) const;
      mcPRL::GeoCoord cellCoord2geoCoord(long iRow, long iCol) const;
      mcPRL::CellCoord geoCoord2cellCoord(const mcPRL::GeoCoord &gCoord) const;
      mcPRL::CellCoord geoCoord2cellCoord(double x, double y) const;

	  /*--------GDAL------------*/
      bool initByGDAL(GDALDataset *pDataset,
                      bool warning = true);
	  /*---------PGTIOL--------*/
	  bool initByPGTIOL(PGTIOLDataset *pDataset,
                      bool warning = true);
	  
      mcPRL::CellspaceGeoinfo subSpaceGeoinfo(const mcPRL::SubCellspaceInfo &subInfo) const;

    private:
      mcPRL::GeoCoord _nwCorner;
      mcPRL::GeoCoord _cellSize;
      string _projection;
  };  
};

#endif
