import pandas as pd
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from ui.windows.asset_detail.shared.pages.history_chart.helpers import (
    plotStatistic,
    __plotTimeRange,
    __plotTimeStatistics,
    __getRectangleObject,
)


def plotWeekends(data: pd.DataFrame, plot: pg.PlotItem):
    low = data["Low"].min()
    high = data["High"].max()

    tempData = data.ffill(axis=0)

    tempData = tempData.set_index("Datetime")

    # width
    w = (1 - 0) / 3.0
    ww = (1 - 0) / 2.0

    aaa = tempData[
        (tempData.index.weekday == 5) | (tempData.index.weekday == 6)
    ]

    aaa.apply(
        lambda x: plotWeekendBar(x, low, high, plot, w, ww), axis=1,
    )


def plotWeekendBar(row, low, high, plot, width1, width2):
    # BAR ----------------------------------
    # bar = pg.QtGui.QGraphicsRectItem(
    #     row["id"] + width2 - width1, low, width1 * 2, high - low,
    # )
    bar = pg.QtGui.QGraphicsRectItem(
        row["id"] + width2 - width1,
        row["Low"],
        width1 * 2,
        row["High"] - row["Low"],
    )
    bar.setPen(pg.mkPen(None))
    bar.setBrush(pg.mkBrush(QColor("#bdbdbd")))
    bar.setOpacity(0.5)
    plot.addItem(bar)


# YEARS --------------------------------------------------------
def plotYears(data: pd.DataFrame, plot: pg.PlotItem):

    # COLOR BAR
    __plotTimeRange(data, plotYearBar, plot)

    # STATISTICS
    # __plotTimeStatistics(
    #     "Y", data, plotStatistic, plot,
    # )


def plotYearBar(row, plot, minY, maxY):
    color = ""

    if (row["Datetime"].year % 2) == 0:
        # odd number
        color = "#37474f"
    else:
        # even number
        color = "#b0bec5"

    # BAR
    bar = __getRectangleObject(color, 0.2, row["id"], minY, 1, maxY - minY)
    plot.addItem(bar)
