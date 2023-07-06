import logging
import time
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot

from business.model.asset import Asset
from business.model.timeframe import Duration, TimeFrame
from business.modules.asset_bl import AssetBL
from ui.base.base_page import BasePage
from ui.components.multi_candlestick_chart.chart import MyMultiCandlestickChart
from ui.windows.asset_detail.shared.pages.history_chart.helpers import (
    fillGapsInDays,
    plotMonths,
    plotQuarters,
    plotSeasons,
    plotUSHolidays,
    plotWeeks,
    plotYears,
)

from ui.windows.asset_detail.futures.pages.history_chart.helpers import (
    plotWeekends,
)

# create logger
log = logging.getLogger("CellarLogger")


class FutureHistoryChartPage(BasePage):
    asset: Asset
    bl: AssetBL

    originData: pd.DataFrame
    dataDF: pd.DataFrame
    datesDF: pd.DataFrame

    currentRange: Tuple[int, int]

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/futures/pages/history_chart/history_chart.ui",
            self,
        )

        # load styles
        with open(
            "ui/windows/asset_detail/futures/pages/history_chart/history_chart.qss",
            "r",
        ) as fh:
            self.setStyleSheet(fh.read())

        # apply styles
        self.setAttribute(Qt.WA_StyledBackground, True)

        # INPUT data
        self.asset: Asset = kwargs["asset"]

        # BL
        self.bl: AssetBL = AssetBL()

        # DEFAULTS
        self.currentRange = (0, 0)
        self.timeframe = TimeFrame.day1
        # self.duration = Duration.year1.value
        self.duration = Duration.years20.value

        # signals
        self.durationComboBox.currentTextChanged.connect(
            self.durationComboBoxChanged
        )
        self.usHolidaysCheckBox.stateChanged.connect(
            self.usHolidaysCheckboxChanged
        )
        self.weeksCheckbox.stateChanged.connect(self.weeksCheckboxChanged)
        self.monthsCheckbox.stateChanged.connect(self.monthsCheckboxChanged)
        self.seasonsCheckbox.stateChanged.connect(self.seasonsCheckboxChanged)
        self.quartersCheckbox.stateChanged.connect(
            self.quartersCheckboxChanged
        )
        self.yearsCheckbox.stateChanged.connect(self.yearsCheckboxChanged)

        # init chart ----------------------------------------------
        log.info("Running")
        start = time.time()

        allHistData = pd.DataFrame(
            columns=[
                "LocalSymbol",
                "LastTradeDate",
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )

        # init chart ----------------------------------------------

        # collect all data from all contracts
        for cd in self.asset.contractDetails:

            fullSymbolName = f"{cd.contract.localSymbol}-{cd.contract.lastTradeDateOrContractMonth}"
            histData = self.__getHistData(fullSymbolName, self.timeframe)

            if histData is not None:
                histData = fillGapsInDays(histData, self.timeframe)

                histData["LocalSymbol"] = cd.contract.localSymbol
                histData[
                    "LastTradeDate"
                ] = cd.contract.lastTradeDateOrContractMonth
                histData = histData.reset_index()

                allHistData = allHistData.append(histData)

        if allHistData.shape[0] > 0:

            datesDf = pd.DataFrame(
                data=allHistData.groupby("Datetime").groups.keys(),
                columns=["Datetime"],
            )
            datesDf["id"] = np.arange(datesDf.shape[0])
            datesDf = datesDf.set_index("Datetime")

            allHistData["id"] = allHistData.apply(
                lambda x: self.__fillID(x, datesDf), axis=1
            )

            self.dataDF = allHistData.copy()
            self.originData = allHistData.copy()
            self.datesDF = datesDf.copy()

            self.dataDF_flat = (
                allHistData.sort_values("LastTradeDate")
                .groupby(["Datetime"])
                .first()
            )

            # CANDLESTICK
            start = time.time()
            self.__plotCandlestickChart(
                self.dataDF, self.datesDF, duration=self.duration
            )
            end = time.time()
            log.info(f"plotCandlestick data takes: {end - start} sec.")

            # WEEKENDS
            # start = time.time()
            # plotWeekends(self.dataDF, self.candlestickChart.candlestickPlot)
            # end = time.time()
            # log.info(f"plotWeekends data takes: {end - start} sec.")

        # ---------------------------------------------------------

    def __fillID(self, row, dates):
        return dates.index.get_loc(row["Datetime"])

    def __getHistData(self, symbol: str, value: TimeFrame) -> pd.DataFrame:
        start = time.time()
        data = self.bl.getHistoricalDataFromDB(symbol, value)
        if data is not None:
            data = data.sort_index()

            # check duplications
            # dupl = self.data.duplicated()
            # allresults = dupl[dupl == True]

        end = time.time()
        log.info(f"takes {end - start} sec.")

        return data

    def __plotCandlestickChart(
        self, data: pd.DataFrame, dates: pd.DataFrame, **kwargs: Any
    ):
        range = (0, 0)
        if "duration" in kwargs:
            range = self.getDuration(kwargs["duration"])
        elif "range" in kwargs:
            range = kwargs["range"]

        self.candlestickChart = MyMultiCandlestickChart(data, dates, range)
        self.candlestickChart.on_range_update.connect(self.__updateRange)
        self.chartBox.addWidget(self.candlestickChart, 0, 0, 0, 0)

    # --------------------------------------------------------
    # COMBO-BOXES
    # --------------------------------------------------------

    @pyqtSlot(str)
    def durationComboBoxChanged(self, value: Duration):
        log.info("Running")
        start = time.time()

        self.duration = value

        if self.dataDF is not None:
            range = self.getDuration(value)
            self.candlestickChart.overviewPlot.timeRegion.setRegion(range)

        end = time.time()
        log.info(f"durationComboBoxChanged data takes: {end - start} sec.")

    # --------------------------------------------------------
    # CHECK-BOXES
    # --------------------------------------------------------
    @pyqtSlot(int)
    def usHolidaysCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotUSHolidays(
                self.dataDF_flat, self.candlestickChart.candlestickPlot
            )
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def weeksCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotWeeks(self.dataDF_flat, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def monthsCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotMonths(self.dataDF_flat, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def seasonsCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotSeasons(
                self.dataDF_flat, self.candlestickChart.candlestickPlot
            )
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def quartersCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotQuarters(
                self.dataDF_flat, self.candlestickChart.candlestickPlot
            )
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def yearsCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotYears(self.dataDF_flat, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    # --------------------------------------------------------
    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    # --------------------------------------------------------

    def __reDrawChart(self):
        self.__plotCandlestickChart(
            self.originData, self.datesDF, range=self.currentRange
        )
        #  plotWeekends(self.originData, self.candlestickChart.candlestickPlot)

        if self.usHolidaysCheckBox.isChecked():
            plotUSHolidays(
                self.dataDF_flat, self.candlestickChart.candlestickPlot
            )

        if self.weeksCheckbox.isChecked():
            plotWeeks(self.dataDF_flat, self.candlestickChart.candlestickPlot)

        if self.monthsCheckbox.isChecked():
            plotMonths(self.dataDF_flat, self.candlestickChart.candlestickPlot)

        if self.seasonsCheckbox.isChecked():
            plotSeasons(
                self.dataDF_flat, self.candlestickChart.candlestickPlot
            )

        if self.quartersCheckbox.isChecked():
            plotQuarters(
                self.dataDF_flat, self.candlestickChart.candlestickPlot
            )

        if self.yearsCheckbox.isChecked():
            plotYears(self.dataDF_flat, self.candlestickChart.candlestickPlot)

    def __updateRange(self, range: Tuple[int, int]):
        self.currentRange = range

        # self.__reDrawChart()

    def getDuration(self, duration: Duration):
        minVal = datetime.now()
        maxVal = datetime.now()
        if duration == Duration.years20.value:
            minVal = maxVal - relativedelta(years=20)
        elif duration == Duration.years10.value:
            minVal = maxVal - relativedelta(years=10)
        elif duration == Duration.year5.value:
            minVal = maxVal - relativedelta(years=5)
        elif duration == Duration.year1.value:
            minVal = maxVal - relativedelta(years=1)
        elif duration == Duration.quarter1.value:
            minVal = maxVal - relativedelta(months=3)
        elif duration == Duration.month1.value:
            minVal = maxVal - relativedelta(months=1)
        elif duration == Duration.week1.value:
            minVal = maxVal - relativedelta(weeks=1)
        elif duration == Duration.all.value:
            minVal = datetime.min

        minIndex = 0
        maxIndex = self.dataDF["id"].max()

        if minVal != datetime.min:
            tempDf = self.dataDF[self.dataDF["Datetime"] > minVal]
            if tempDf.shape[0] > 0:
                minIndex = tempDf["id"].min()

        return (minIndex, maxIndex)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
