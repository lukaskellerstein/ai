import logging
import time
from typing import Tuple
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor

from business.model.timeframe import TimeFrame
from ui.components.multi_candlestick_chart.candlestick_plot import (
    CandlestickPlot,
)
from ui.components.candlestick_chart.overview_plot import OverviewTimePlot
from ui.components.multi_candlestick_chart.candlestick_plot import (
    CandlestickPlot,
)
from ui.components.multi_candlestick_chart.candlestick_x import CandlesticXAxis

from ui.components.multi_candlestick_chart.helpers import printOHLCInfo
from ui.components.multi_candlestick_chart.volume_plot import VolumePlot

# create logger
log = logging.getLogger("CellarLogger")


class MyMultiCandlestickChart(pg.GraphicsLayoutWidget):
    lastRange = None

    on_range_update = pyqtSignal(object)

    def __init__(
        self,
        dataDF: pd.DataFrame,
        datesDF: pd.DataFrame,
        range: Tuple[int, int],
        parent=None,
        **kargs,
    ):
        super().__init__(parent=parent, **kargs)

        pg.setConfigOptions(antialias=True)

        self.setBackground(QColor("white"))

        # INPUTS
        self.data = dataDF.reset_index()
        self.currentRange = range

        # CHART 1 ----------------------------

        start = time.time()

        date_axis = CandlesticXAxis(data=datesDF, orientation="bottom",)

        self.candlestickPlot = CandlestickPlot(
            self.data, self.currentRange, axisItems={"bottom": date_axis}
        )
        self.candlestickPlot.sigRangeChanged.connect(
            self.__updateCandlestickRegion
        )

        self.addItem(self.candlestickPlot, row=1, col=0, colspan=2, rowspan=1)

        self.proxy = pg.SignalProxy(
            self.candlestickPlot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouseMoved,
        )

        # # CHART 2 ----------------------------

        volumeYAxis = pg.AxisItem(orientation="left")
        volumeYAxis.setScale(0.00001)

        volumeXAxis = CandlesticXAxis(data=datesDF, orientation="bottom",)

        self.volumePlot = VolumePlot(
            self.data,
            self.currentRange,
            axisItems={"bottom": volumeXAxis, "left": volumeYAxis},
        )

        self.volumePlot.sigRangeChanged.connect(self.__updateVolumeRegion)

        self.addItem(self.volumePlot, row=3, col=0, colspan=2)

        self.proxy2 = pg.SignalProxy(
            self.volumePlot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouseMoved2,
        )

        # CHART 3 ----------------------------

        x = datesDF["id"].to_list()
        y = self.data.groupby("Datetime").mean()["Close"].ffill().to_list()

        overviewXAxis = CandlesticXAxis(data=datesDF, orientation="bottom",)

        self.overviewPlot = OverviewTimePlot(
            x, y, axisItems={"bottom": overviewXAxis}
        )
        self.overviewPlot.timeRegion.sigRegionChangeFinished.connect(
            self.__updateOverviewTimeRegion
        )

        self.addItem(self.overviewPlot, row=4, col=0, colspan=2)

        end = time.time()
        log.info(f"plot takes: {end - start} sec.")

    def mouseMoved(self, evt):
        pos = evt[0]
        # print(pos)

        if self.candlestickPlot.sceneBoundingRect().contains(pos):

            mousePoint = self.candlestickPlot.vb.mapSceneToView(pos)

            # set crosshair
            self.candlestickPlot.vLine.setPos(mousePoint.x())
            self.candlestickPlot.hLine.setPos(mousePoint.y())
            self.volumePlot.vLine.setPos(mousePoint.x())

            # other
            index = int(mousePoint.x())
            value = round(mousePoint.y(), 2)
            # print(index)
            # print(value)

            currentBar = self.data.loc[self.data["id"] == index]
            currentBar = currentBar.sort_values(by=["LastTradeDate"])
            # print(currentBar)

            if currentBar.shape[0] > 0:

                # Labels ---
                date = currentBar.iloc[0]["Datetime"]

                resultXHtml = f"<div><span style='color:black'>x={date}</span>, <span style='color:black'>y={value}</span></div>"
                resultOHLCHtml = printOHLCInfo(currentBar)
                self.candlestickPlot.labelOHLC.setHtml(
                    resultXHtml + resultOHLCHtml
                )

                # Opacity for candlestick ---
                validGroupNames = currentBar["LocalSymbol"].unique()
                # print(validGroupNames)

                otherGroups = self.data.loc[
                    ~self.data["LocalSymbol"].isin(
                        validGroupNames
                    )  # ~ is not in
                ]["LocalSymbol"].unique()

                currentGroupIndex = 1
                for vgn in validGroupNames:
                    opacity = 0
                    if currentGroupIndex == 1:
                        opacity = 1
                    elif currentGroupIndex == 2:
                        opacity = 0.3
                    elif currentGroupIndex == 3:
                        opacity = 0.2
                    else:
                        opacity = 0.1

                    vgnFullName = f"{vgn}-{currentBar[currentBar['LocalSymbol'] == vgn].iloc[0]['LastTradeDate']}"

                    self.candlestickPlot.setGroupOpacity(vgnFullName, opacity)
                    currentGroupIndex += 1

                for ogn in otherGroups:

                    ognFullName = f"{ogn}-{self.data[self.data['LocalSymbol'] == ogn].iloc[0]['LastTradeDate']}"

                    self.candlestickPlot.setGroupOpacity(ognFullName, 0.02)

    def mouseMoved2(self, evt):
        pos = evt[
            0
        ]  ## using signal proxy turns original arguments into a tuple
        if self.volumePlot.sceneBoundingRect().contains(pos):
            mousePoint = self.volumePlot.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            value = round(mousePoint.y(), 2)

            currentBar = self.data.loc[self.data["id"] == index]
            currentBar = currentBar.sort_values(by=["LastTradeDate"])

            if currentBar.shape[0] > 0:

                # Labels ---
                date = currentBar.iloc[0]["Datetime"]

                resultXHtml = f"<div><span style='color:black'>x={date}</span>, <span style='color:black'>y={value}</span></div>"
                resultOHLCHtml = printOHLCInfo(currentBar)
                self.candlestickPlot.labelOHLC.setHtml(
                    resultXHtml + resultOHLCHtml
                )

            # if index > 0 and index < self.data.shape[0]:
            #     row = self.data.iloc[index]

            #     self.labelXY.setText(f"x={row['Datetime']}, y={value}")
            #     self.labelOHLC.setText(
            #         f"O={row['Open']}, H={row['High']}, L={row['Low']}, C={row['Close']}, V={row['Volume']:.0f}"
            #     )
            self.candlestickPlot.vLine.setPos(mousePoint.x())
            self.volumePlot.vLine.setPos(mousePoint.x())

    def __updateCandlestickRegion(self, window, viewRange):
        xRange = viewRange[0]
        # yRange = viewRange[1]
        self.overviewPlot.timeRegion.setRegion(xRange)

    def __updateVolumeRegion(self, window, viewRange):
        xRange = viewRange[0]
        # yRange = viewRange[1]
        self.overviewPlot.timeRegion.setRegion(xRange)

    def __updateOverviewTimeRegion(self, region):

        region.setZValue(10)
        minX, maxX = region.getRegion()

        # round the values
        minVal = round(minX)
        maxVal = round(maxX)

        log.info(f"set Region: {minVal}, {maxVal}")

        if self.lastRange != (minVal, maxVal):
            self.lastRange = (minVal, maxVal)

            self.on_range_update.emit(self.lastRange)

            log.info(f"run update Range: {minVal}, {maxVal}")

            # udpate X axis of Volume
            self.volumePlot.setXRange(minVal, maxVal, padding=0)

            # udpate X axis of CANDLESTICK
            self.candlestickPlot.updateRange((minVal, maxVal))

            # udpate Y axis of Volume
            # if minVal < 0:
            #     minVal = 0
            # elif minVal > self.data.shape[0] - 1:
            #     minVal = self.data.shape[0] - 1

            # if maxVal < 0:
            #     maxVal = 0
            # elif maxVal > self.data.shape[0] - 1:
            #     maxVal = self.data.shape[0] - 1

            # tempDf = self.data[
            #     (self.data["id"] >= minVal) & (self.data["id"] <= maxVal)
            # ].copy()

            # # update Y axis of Candlestic
            # self.candlestickPlot.setYRange(
            #     tempDf["Low"].min(), tempDf["High"].max(), padding=0,
            # )

            # yPos = tempDf["High"].max()
            # print(f"Set text position ({minVal}, {yPos})")

            # self.candlestickPlot.labelOHLC.setPos(minVal, yPos)

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
