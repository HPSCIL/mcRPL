/*
 * zonalStatsTrans.h
 *
 *  Created on: Mar 18, 2014
 *      Author: sonicben
 */

#ifndef ZONALSTATSTRANS_H
#define ZONALSTATSTRANS_H

#include "prpl-transition.h"
#include <map>
#include <vector>
#include <cmath>
#include <functional>

using namespace std;

class ZonalStatsTransition: public pRPL::Transition {
  public:
    ZonalStatsTransition();
    ~ZonalStatsTransition();

    unsigned short getMinZoneID() const;
    void setMinZoneID(unsigned short minZoneID);
    unsigned short getMaxZoneID() const;
    void setMaxZoneID(unsigned short maxZoneID);

    bool minValVector(vector<unsigned short> &vMinVals) const;
    bool maxValVector(vector<unsigned short> &vMaxVals) const;
    bool countVector(vector<long> &vCounts) const;
    bool totalValVector(vector<long> &vTotVals) const;

    bool calcStats(vector<unsigned short> &vMinVals,
                   vector<unsigned short> &vMaxVals,
                   vector<long> &vTotVals,
                   vector<long> &vCounts);

    const map<unsigned short, double>& meanMap() const;
    const map<unsigned short, unsigned short>& minMap() const;
    const map<unsigned short, unsigned short>& maxMap() const;

    bool check() const;
    bool afterSetCellspaces(int subCellspcGlbIdx = pRPL::ERROR_ID);
    pRPL::EvaluateReturn evaluate(const pRPL::CellCoord &coord);

  protected:
    bool _checkZoneRange() const;

  protected:
    pRPL::Cellspace *_pValCellspc;
    pRPL::Cellspace *_pZoneCellspc;
    unsigned short _valNoData;
    unsigned short _zoneNoData;

    map<unsigned short, unsigned short> _mMins;
    map<unsigned short, unsigned short> _mMaxs;
    map<unsigned short, long> _mCounts;
    map<unsigned short, long> _mTotals;
    map<unsigned short, double> _mMeans;

    unsigned short _minZoneID;
    unsigned short _maxZoneID;
};

class ZonalOutputTransition: public pRPL::Transition {
  public:
    ZonalOutputTransition();
    ~ZonalOutputTransition();

    void setMeanMap(const map<unsigned short, double> *pmMeans);
    void setMinMap(const map<unsigned short, unsigned short> *pmMins);
    void setMaxMap(const map<unsigned short, unsigned short> *pmMaxs);

    bool check() const;
    bool afterSetCellspaces(int subCellspcGlbIdx = pRPL::ERROR_ID);
    pRPL::EvaluateReturn evaluate(const pRPL::CellCoord &coord);

  protected:
    pRPL::Cellspace *_pZoneCellspc;
    pRPL::Cellspace *_pMeanCellspc;
    pRPL::Cellspace *_pMinCellspc;
    pRPL::Cellspace *_pMaxCellspc;
    unsigned short _zoneNoData;
    float _meanNoData;
    unsigned short _minNoData;
    unsigned short _maxNoData;

    const map<unsigned short, double> *_pmMeans;
    const map<unsigned short, unsigned short> *_pmMins;
    const map<unsigned short, unsigned short> *_pmMaxs;
};


#endif /* ZONALSTATSTRANS_H_ */
