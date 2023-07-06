import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg

from business.model.timeframe import TimeFrame

# create logger
log = logging.getLogger("CellarLogger")


class CandlesticXAxis(pg.AxisItem):

    data: pd.DataFrame

    def __init__(
        self,
        data: pd.DataFrame,
        orientation,
        pen=None,
        linkView=None,
        parent=None,
        maxTickLength=-5,
        showValues=True,
    ):
        super().__init__(
            orientation,
            pen=pen,
            linkView=linkView,
            parent=parent,
            maxTickLength=maxTickLength,
            showValues=showValues,
        )
        self.data = data.reset_index().copy()

        self.dataDate = self.data.copy()
        self.dataDate = self.dataDate.set_index(["Datetime"])

        self.dataInt = self.data.copy()
        self.dataInt = self.dataInt.set_index(["id"])

    def tickValues(self, minVal, maxVal, size):
        # start = time.time()
        # aaa = super().tickValues(minVal, maxVal, size)
        # return aaa

        # print(f"min:{minVal}, max: {maxVal}")
        minVal = round(minVal)
        maxVal = round(maxVal)
        # print(f"min:{minVal}, max: {maxVal}")

        tickLevels = self.tickSpacing(minVal, maxVal, size)

        # FROM and To
        aFromIndex = 0
        aFromOrigin: datetime = self.dataInt.loc[aFromIndex]["Datetime"]
        aFrom = None

        aToIndex = self.data.shape[0] - 1
        aToOrigin: datetime = self.dataInt.loc[aToIndex]["Datetime"]
        aTo = None

        # FROM
        if minVal >= aFromIndex and minVal <= aToIndex:
            aFrom = self.dataInt.loc[minVal]["Datetime"]
        elif minVal < aFromIndex:
            numDays = aFromIndex - minVal
            aFrom = aFromOrigin - timedelta(days=numDays)
        elif minVal > aToIndex:
            numDays = minVal - aToIndex
            aFrom = aToOrigin + timedelta(days=numDays)
        else:
            log.info("???????????????")

        # TO
        if maxVal >= aFromIndex and maxVal <= aToIndex:
            aTo = self.dataInt.loc[maxVal]["Datetime"]
        elif maxVal < aFromIndex:
            numDays = aFromIndex - maxVal
            aTo = aFromOrigin - timedelta(days=numDays)
        elif maxVal > aToIndex:
            numDays = maxVal - aToIndex
            aTo = aToOrigin + timedelta(days=numDays)
        else:
            print("???????????????")

        # --------------------------------
        diff = 0
        result = []
        diff = (aTo - aFrom).days
        ratio = diff / size  # ratio between range of data / size of the screen

        # print(f"scale: {self.scale}")
        # print(f"diff: {diff}, size: {size}")
        # print(f"ratio: {ratio}")

        if diff > 8000:
            # decades - Offset alias - https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            # print("decades")
            result = pd.date_range(
                aFromOrigin, aToOrigin, freq="10AS"
            ).tolist()
        elif diff <= 8000 and diff > 1600:
            # years
            # print("years")

            if ratio <= 4.7:
                result = pd.date_range(aFrom, aTo, freq="AS").tolist()
            else:
                result = pd.date_range(
                    aFromOrigin, aToOrigin, freq="10AS"
                ).tolist()

        elif diff <= 1600 and diff > 700:
            # quarters
            # print("quarters")

            if ratio <= 1.2:
                result = pd.date_range(aFrom, aTo, freq="BQS").tolist()
            else:
                result = pd.date_range(aFrom, aTo, freq="AS").tolist()

        elif diff <= 700 and diff > 120:
            # months
            # print("months")

            if ratio <= 0.4:
                result = pd.date_range(aFrom, aTo, freq="MS").tolist()
            else:
                result = pd.date_range(aFrom, aTo, freq="BQS").tolist()

        elif diff <= 120 and diff > 20:
            # weeks
            # print("weeks")

            if ratio <= 0.09:
                result = pd.date_range(aFrom, aTo, freq="W").tolist()
            else:
                result = pd.date_range(aFrom, aTo, freq="MS").tolist()

        elif diff <= 20:
            # days
            # print("days")

            if ratio <= 0.013:
                result = pd.date_range(aFrom, aTo, freq="D").tolist()
            else:
                result = pd.date_range(aFrom, aTo, freq="W").tolist()

        resultFinal = []
        resultInIndex = self.dataDate[self.dataDate.index.isin(result)]

        for level in tickLevels:
            spacing, offset = level

            # print(f"spacing: {spacing}")

            resultFinal.append((spacing, resultInIndex["id"].to_list()))

        # end = time.time()
        # log.info(f"tickValues takes: {end - start} sec.")

        return resultFinal

    def tickStrings(self, values, scale, spacing):
        # start = time.time()
        # aaa = super().tickStrings(values, scale, spacing)
        # return aaa

        # print(f"scale: {scale}, spacing: {spacing}, values: {values}")

        result = []
        for value in values:
            if value >= 0 and value <= self.data.shape[0]:
                result.append(
                    self.dataInt.loc[value]["Datetime"].strftime("%Y%m%d")
                )
            else:
                result.append(value)

        # log.info(f"result:{result}")
        # end = time.time()
        # log.info(f"tickStrings takes: {end - start} sec.")

        return result
