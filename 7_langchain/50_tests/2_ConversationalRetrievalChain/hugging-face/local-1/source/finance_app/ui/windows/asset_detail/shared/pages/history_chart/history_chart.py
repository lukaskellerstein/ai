import logging
import time
from datetime import datetime
from typing import Any, Tuple

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot

from business.model.asset import Asset
from business.model.timeframe import Duration, TimeFrame
from business.modules.asset_bl import AssetBL
from ui.base.base_page import BasePage
from ui.components.candlestick_chart.chart import MyCandlestickChart
from ui.windows.asset_detail.shared.pages.history_chart.helpers import (
    fillGapsInDays,
    plotMonths,
    plotQuarters,
    plotSeasons,
    plotUSHolidays,
    plotWeekends,
    plotWeeks,
    plotYears,
)

# create logger
log = logging.getLogger("CellarLogger")


class HistoryChartPage(BasePage):
    asset: Asset
    bl: AssetBL

    originData: pd.DataFrame
    dataDF: pd.DataFrame

    currentRange: Tuple[int, int]

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/shared/pages/history_chart/history_chart.ui",
            self,
        )

        # load styles
        with open(
            "ui/windows/asset_detail/shared/pages/history_chart/history_chart.qss",
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
        self.duration = Duration.year1.value

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

        # HIST DATA
        log.info("Running")
        start = time.time()
        self.data = self.__getHistData(self.asset.symbol, self.timeframe)
        end = time.time()
        log.info(f"get data takes: {end - start} sec.")

        # FILL GAPS IN DATA
        log.info("Running")
        start = time.time()
        if self.data is not None:
            self.data = fillGapsInDays(self.data, self.timeframe)
        end = time.time()
        log.info(f"enhance data takes: {end - start} sec.")

        if self.data is not None:
            self.data["id"] = np.arange(self.data.shape[0])

            # CANDLESTICK
            start = time.time()
            self.originData = self.data.copy()
            self.__plotCandlestickChart(self.data, duration=self.duration)
            end = time.time()
            log.info(f"plotCandlestickChart data takes: {end - start} sec.")

            # WEEKENDS
            # start = time.time()
            # plotWeekends(self.data, self.candlestickChart.candlestickPlot)
            # end = time.time()
            # log.info(f"plotWeekends data takes: {end - start} sec.")

        self.durationComboBox.setCurrentText(self.duration)
        # ---------------------------------------------------------

    def __getHistData(self, symbol: str, value: TimeFrame):
        start = time.time()
        data = self.bl.getHistoricalDataFromDB(symbol, value)
        if data is not None:
            data = data.sort_index()

            # # check duplications
            # dupl = self.data.duplicated()
            # allresults = dupl[dupl == True]

        end = time.time()
        log.info(f"takes {end - start} sec.")

        return data

    def __plotCandlestickChart(self, data: pd.DataFrame, **kwargs: Any):
        range = (0, 0)
        if "duration" in kwargs:
            range = self.getDuration(kwargs["duration"])
        elif "range" in kwargs:
            range = kwargs["range"]

        self.candlestickChart = MyCandlestickChart(data, range)
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

        if self.data is not None:
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
            plotUSHolidays(self.data, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def weeksCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotWeeks(self.data, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def monthsCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotMonths(self.data, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def seasonsCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotSeasons(self.data, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def quartersCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotQuarters(self.data, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    @pyqtSlot(int)
    def yearsCheckboxChanged(self, state: int):
        if state == Qt.Checked:
            plotYears(self.data, self.candlestickChart.candlestickPlot)
        else:
            self.__reDrawChart()

    # --------------------------------------------------------
    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    # --------------------------------------------------------

    def __reDrawChart(self):
        self.__plotCandlestickChart(self.originData, range=self.currentRange)
        # plotWeekends(self.originData, self.candlestickChart.candlestickPlot)

        if self.usHolidaysCheckBox.isChecked():
            plotUSHolidays(
                self.originData, self.candlestickChart.candlestickPlot
            )

        if self.weeksCheckbox.isChecked():
            plotWeeks(self.originData, self.candlestickChart.candlestickPlot)

        if self.monthsCheckbox.isChecked():
            plotMonths(self.originData, self.candlestickChart.candlestickPlot)

        if self.seasonsCheckbox.isChecked():
            plotSeasons(self.originData, self.candlestickChart.candlestickPlot)

        if self.quartersCheckbox.isChecked():
            plotQuarters(
                self.originData, self.candlestickChart.candlestickPlot
            )

        if self.yearsCheckbox.isChecked():
            plotYears(self.originData, self.candlestickChart.candlestickPlot)

    def __updateRange(self, range: Tuple[int, int]):
        self.currentRange = range

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
        maxIndex = self.data.shape[0]

        if minVal != datetime.min:
            tempDf = self.data[self.data.index > minVal]
            if tempDf.shape[0] > 0:
                minIndex = self.data.index.get_loc(tempDf.index[0])

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
