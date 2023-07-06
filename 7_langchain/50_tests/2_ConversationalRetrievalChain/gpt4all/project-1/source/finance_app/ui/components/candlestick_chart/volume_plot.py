import logging
import pyqtgraph as pg

from typing import Tuple
from PyQt5 import QtGui

# create logger
log = logging.getLogger("CellarLogger")


class VolumePlot(pg.PlotItem):
    def __init__(self, x, y, currentRange: Tuple[int, int], **kwargs):
        pg.PlotItem.__init__(self, **kwargs)

        self.setFixedHeight(100)
        self.setXRange(*currentRange, padding=0)

        # cross hair
        self.vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(QtGui.QColor("black"))
        )
        self.addItem(self.vLine, ignoreBounds=True)

        # (Afrom, Ato) = currentRange
        # self.aaa = x[Afrom:Ato]
        # self.bbb = y[Afrom:Ato]

        self.addItem(pg.BarGraphItem(x=x, height=y, width=0.5))
