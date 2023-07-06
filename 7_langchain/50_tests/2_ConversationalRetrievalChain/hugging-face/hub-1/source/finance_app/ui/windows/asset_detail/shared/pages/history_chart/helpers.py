import pandas as pd
import pyqtgraph as pg
from business.model.timeframe import TimeFrame
from PyQt5.QtGui import QColor
import numpy as np

from typing import Callable, Any
from holidays import US
from trading_calendars import get_calendar


def fillGapsInDays(data: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
    if timeframe.value == TimeFrame.day1.value:
        data = data.asfreq("D").reset_index().set_index("Datetime")

    return data


def __getRectangleObject(
    color: str, opacity: float, x: int, y: float, width: float, height: float
) -> pg.QtGui.QGraphicsRectItem:
    bar = pg.QtGui.QGraphicsRectItem(x, y, width, height)

    bar.setPen(pg.mkPen(None))
    bar.setOpacity(opacity)
    bar.setBrush(pg.mkBrush(QColor(color)))
    return bar


def __plotTimeRange(
    data: pd.DataFrame,
    method: Callable[[pd.Series, Any, float, float], Any],
    plot: pg.PlotItem,
):
    minY = data["Low"].min()
    maxY = data["High"].max()

    data.apply(
        lambda x: method(x, plot, minY, maxY), axis=1,
    )


def __plotTimeStatistics(
    rangeType: str,
    data: pd.DataFrame,
    method: Callable[[pd.Series, Any], Any],
    plot: pg.PlotItem,
):

    data["Change"] = data["Close"].pct_change() * 100

    sum_change_data = data["Change"].resample(rangeType).sum()
    sum_change_data.name = "Sum"

    avg_change_data = data["Change"].resample(rangeType).mean()
    avg_change_data.name = "Avg"
    var_change_data = data["Change"].resample(rangeType).var()  # variance
    var_change_data.name = "Var"
    std_change_data = data["Change"].resample(rangeType).std(ddof=2)  # std dev
    std_change_data.name = "Std"

    id_data = data["id"].resample(rangeType).max()
    price_data = data["Close"].resample(rangeType).last()

    full_data = pd.concat(
        [
            id_data,
            price_data,
            sum_change_data,
            avg_change_data,
            var_change_data,
            std_change_data,
        ],
        axis=1,
    )

    full_data.apply(
        lambda x: method(x, plot), axis=1,
    )


def plotStatistic(row, plot: pg.PlotItem):

    # ARROW -----------------------------------
    arrow = pg.ArrowItem(
        pos=(row["id"], row["Close"]),
        angle=0,
        tipAngle=60,
        headLen=20,
        tailLen=20,
        tailWidth=10,
        # pen={"color": "w", "width": 1},
        brush=QColor("#303f9f"),
    )
    plot.addItem(arrow)

    # LABELS ----------------------------------
    label = pg.TextItem(
        html=f"""
        <div style="text-align: center">
            <span style="color: #FFF;">Return: {round(row["Sum"])} %</span>
        </div>
        <div style="text-align: center">
            <span style="color: #FFF;">Daily Return: {round(row["Avg"],2)} %</span>
        </div>
        <div style="text-align: center">
            <span style="color: #FFF;">Variance: {round(row["Var"],2)}</span>
        </div>
        <div style="text-align: center">
            <span style="color: #FFF;">StdDev: {round(row["Std"],2)}</span>
        </div>
        """,
        # text=f"{round(row['Change'])} %",
        # border="w",
        # fill=(0, 0, 255, 100),
        fill=QColor("#303f9f"),
        anchor=(0.0, 0.0),
    )
    label.setPos(row["id"], row["Close"])
    plot.addItem(label)


# HOLIDAYS --------------------------------------------------------
def plotUSHolidays(data: pd.DataFrame, plot: pg.PlotItem):

    holidayNames = US()
    nyse = get_calendar("XNYS")
    holidays = nyse.regular_holidays.holidays()

    # width
    w = (1 - 0) / 3.0
    ww = (1 - 0) / 2.0

    tempData = data.ffill(axis=0)
    tempData["id"] = np.arange(data.shape[0])

    tempDataMin = tempData.index.min()
    tempDataMax = tempData.index.max()

    for holiday in holidays:
        if tempDataMin < holiday < tempDataMax:
            # idx_holiday = tempData.index.get_loc(holiday.to_pydatetime())
            # val_holiday = tempData.loc[holiday.to_pydatetime()]["High"]

            row = tempData.loc[holiday.to_pydatetime()]

            # ARROW -----------------------------------
            arrow = pg.ArrowItem(
                pos=(row["id"] + ww, row["High"]),
                angle=-75,
                tipAngle=60,
                headLen=20,
                tailLen=20,
                tailWidth=10,
                # pen={"color": "w", "width": 1},
                brush=QColor("#8d8d8d"),
            )
            plot.addItem(arrow)

            # LABEL ---------------------------------
            label = pg.TextItem(
                # html=f'<div style="text-align: center"><span style="color: #FFF;">{holidayNames.get(holiday)}</span></div>',
                text=f"{holidayNames.get(holiday)}",
                border="w",
                # fill=(0, 0, 255, 100),
                fill=QColor("#8d8d8d"),
                anchor=(0.0, 2.5),
            )
            label.setPos(row["id"] + ww, row["High"])
            plot.addItem(label)

            # BAR ----------------------------------
            bar = pg.QtGui.QGraphicsRectItem(
                row["id"] + ww - w,
                row["Low"],
                w * 2,
                row["High"] - row["Low"],
            )

            bar.setPen(pg.mkPen(None))
            bar.setBrush(pg.mkBrush(QColor("gray")))
            plot.addItem(bar)


# WEEKS --------------------------------------------------------


def plotWeeks(data: pd.DataFrame, plot: pg.PlotItem):

    # COLOR BAR
    __plotTimeRange(data, plotWeekBar, plot)

    # STATISTICS
    __plotTimeStatistics(
        "W-MON", data, plotStatistic, plot,
    )


def plotWeekBar(row: pd.Series, plot, minY: float, maxY: float):

    weekNum = row.name.isocalendar()[1]

    color = ""
    if (weekNum % 2) == 0:
        # odd number
        color = "#37474f"
    else:
        # even number
        color = "#b0bec5"

    # BAR
    bar = __getRectangleObject(color, 0.2, row["id"], minY, 1, maxY - minY)
    plot.addItem(bar)


# WEEKENDS --------------------------------------------------------


def plotWeekends(data: pd.DataFrame, plot: pg.PlotItem):
    tempData = data.ffill(axis=0)
    tempData["id"] = np.arange(data.shape[0])

    # width
    w = (1 - 0) / 3.0
    ww = (1 - 0) / 2.0

    aaa = tempData[
        (tempData.index.weekday == 5) | (tempData.index.weekday == 6)
    ]

    aaa.apply(
        lambda x: plotWeekendBar(x, plot, w, ww), axis=1,
    )


def plotWeekendBar(
    row: pd.DataFrame, plot: pg.PlotItem, width1: float, width2: float
):
    # BAR
    color = "#bdbdbd"
    bar = __getRectangleObject(
        color,
        1,
        row["id"] + width2 - width1,
        row["Low"],
        width1 * 2,
        row["High"] - row["Low"],
    )
    plot.addItem(bar)


# MONTHS --------------------------------------------------------
def plotMonths(data: pd.DataFrame, plot: pg.PlotItem):

    # COLOR BAR
    __plotTimeRange(data, plotMonthBar, plot)

    # STATISTICS
    __plotTimeStatistics(
        "M", data, plotStatistic, plot,
    )


def plotMonthBar(row, plot, minY, maxY):
    color = ""

    # January - Q1 - Winter
    if row.name.month == 1:
        color = "#d32f2f"
    # February - Q1 - Winter
    elif row.name.month == 2:
        color = "#9c27b0"
    # March - Q1 - Spring
    elif row.name.month == 3:
        color = "#3f51b5"
    # April - Q2 - Spring
    elif row.name.month == 4:
        color = "#2196f3"
    # May - Q2 - Spring
    elif row.name.month == 5:
        color = "#00bcd4"
    # Jun - Q2 - Summer
    elif row.name.month == 6:
        color = "#009688"
    # July - Q3 - Summer
    elif row.name.month == 7:
        color = "#4caf50"
    # August - Q3 - Summer
    elif row.name.month == 8:
        color = "#cddc39"
    # September - Q3 - Autumn
    elif row.name.month == 9:
        color = "#ffeb3b"
    # October - Q4 - Autumn
    elif row.name.month == 10:
        color = "#ffc107"
    # November - Q4 - Autumn
    elif row.name.month == 11:
        color = "#ff9800"
    # December - Q4 - Winter
    elif row.name.month == 12:
        color = "#ff5722"
    else:
        color = "red"

    # BAR
    bar = __getRectangleObject(color, 0.2, row["id"], minY, 1, maxY - minY)
    plot.addItem(bar)


# SEASONS --------------------------------------------------------
def plotSeasons(data: pd.DataFrame, plot: pg.PlotItem):
    # COLOR BAR
    __plotTimeRange(data, plotSeasonBar, plot)


def plotSeasonBar(row, plot, minY, maxY):
    color = ""

    # January - Q1 - Winter
    if row.name.month == 1:
        color = "#1e88e5"
    # February - Q1 - Winter
    elif row.name.month == 2:
        color = "#1e88e5"
    # March - Q1 - Spring
    elif row.name.month == 3:
        color = "#7cb342"
    # April - Q2 - Spring
    elif row.name.month == 4:
        color = "#7cb342"
    # May - Q2 - Spring
    elif row.name.month == 5:
        color = "#7cb342"
    # Jun - Q2 - Summer
    elif row.name.month == 6:
        color = "#e53935"
    # July - Q3 - Summer
    elif row.name.month == 7:
        color = "#e53935"
    # August - Q3 - Summer
    elif row.name.month == 8:
        color = "#e53935"
    # September - Q3 - Autumn
    elif row.name.month == 9:
        color = "#4e342e"
    # October - Q4 - Autumn
    elif row.name.month == 10:
        color = "#4e342e"
    # November - Q4 - Autumn
    elif row.name.month == 11:
        color = "#4e342e"
    # December - Q4 - Winter
    elif row.name.month == 12:
        color = "#1e88e5"
    else:
        color = "red"
        print("????????????????????????")

    # BAR ----------------------------------
    bar = __getRectangleObject(color, 0.2, row["id"], minY, 1, maxY - minY)
    plot.addItem(bar)


# QUARTERS --------------------------------------------------------
def plotQuarters(data: pd.DataFrame, plot: pg.PlotItem):

    # COLOR BAR
    __plotTimeRange(data, plotQuarterBar, plot)

    # STATISTICS
    __plotTimeStatistics(
        "Q", data, plotStatistic, plot,
    )


def plotQuarterBar(row, plot, minY, maxY):
    color = ""

    # January - Q1 - Winter
    if row.name.month == 1:
        color = "#b0bec5"
    # February - Q1 - Winter
    elif row.name.month == 2:
        color = "#b0bec5"
    # March - Q1 - Spring
    elif row.name.month == 3:
        color = "#b0bec5"
    # April - Q2 - Spring
    elif row.name.month == 4:
        color = "#78909c"
    # May - Q2 - Spring
    elif row.name.month == 5:
        color = "#78909c"
    # Jun - Q2 - Summer
    elif row.name.month == 6:
        color = "#78909c"
    # July - Q3 - Summer
    elif row.name.month == 7:
        color = "#546e7a"
    # August - Q3 - Summer
    elif row.name.month == 8:
        color = "#546e7a"
    # September - Q3 - Autumn
    elif row.name.month == 9:
        color = "#546e7a"
    # October - Q4 - Autumn
    elif row.name.month == 10:
        color = "#37474f"
    # November - Q4 - Autumn
    elif row.name.month == 11:
        color = "#37474f"
    # December - Q4 - Winter
    elif row.name.month == 12:
        color = "#37474f"
    else:
        color = "red"
        print("????????????????????????")

    # BAR ----------------------------------
    bar = __getRectangleObject(color, 0.2, row["id"], minY, 1, maxY - minY)
    plot.addItem(bar)


# YEARS --------------------------------------------------------
def plotYears(data: pd.DataFrame, plot: pg.PlotItem):

    # COLOR BAR
    __plotTimeRange(data, plotYearBar, plot)

    # STATISTICS
    __plotTimeStatistics(
        "Y", data, plotStatistic, plot,
    )


def plotYearBar(row, plot, minY, maxY):
    color = ""

    if (row.name.year % 2) == 0:
        # odd number
        color = "#37474f"
    else:
        # even number
        color = "#b0bec5"

    # BAR
    bar = __getRectangleObject(color, 0.2, row["id"], minY, 1, maxY - minY)
    plot.addItem(bar)

