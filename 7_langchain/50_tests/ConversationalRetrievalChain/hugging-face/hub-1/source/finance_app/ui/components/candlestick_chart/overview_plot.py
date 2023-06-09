import logging

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


# create logger
log = logging.getLogger("CellarLogger")


class OverviewTimePlot(pg.PlotItem):
    def __init__(self, x, y, **kwargs):
        pg.PlotItem.__init__(self, **kwargs)
        self.setFixedHeight(100)

        # axis
        # overviewXAxis = OverviewTimeXAxis(
        #     data=self.timeSeriesData,
        #     timeframe=self.timeframe,
        #     orientation="bottom",
        # )
        # self.setAxisItems({"bottom": overviewXAxis})

        # grid
        self.showGrid(x=True, y=True)

        # plot
        self.plot(
            x,
            y,
            name="Overview plot",
            pen=pg.mkPen(color=QColor("blue"), style=Qt.SolidLine),
        )

        # region
        self.timeRegion = pg.LinearRegionItem()
        self.timeRegion.setZValue(10)
        self.timeRegion.setRegion([len(x) - 1000, len(x)])

        self.addItem(self.timeRegion, ignoreBounds=True)
