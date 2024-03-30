"""
Demonstrate creation of a custom graphic (a candlestick plot)
"""
import logging
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from typing import Tuple

# create logger
log = logging.getLogger("CellarLogger")


class CandlestickPlot(pg.PlotItem):
    def __init__(
        self, data: pd.DataFrame, currentRange: Tuple[int, int], **kwargs
    ):
        pg.PlotItem.__init__(self, **kwargs)

        self.data = data
        self.data["id"] = np.arange(self.data.shape[0])

        # grid
        self.showGrid(x=True)

        # cross hair
        self.vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(QtGui.QColor("black"))
        )
        self.hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(QtGui.QColor("black"))
        )
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)

        # current range
        self.setXRange(*currentRange, padding=0)
        (Afrom, Ato) = currentRange
        self.painted: pd.DataFrame = self.data[Afrom:Ato].copy()

        # plot
        self.plot = CandlestickGraphics(self.painted)
        self.addItem(self.plot)

    def updateRange(self, currentRange: Tuple[int, int]):

        (Afrom, Ato) = currentRange

        if Afrom < 0:
            Afrom = 0
        elif Afrom > self.data["id"].max():
            Afrom = self.data["id"].max()

        if Ato < 0:
            Ato = 0
        elif Ato > self.data["id"].max():
            Ato = self.data["id"].max()

        log.info(f"update Range: {Afrom}, {Ato}")

        if Afrom != Ato:

            # current range
            tempDf = self.data[
                (self.data["id"] >= Afrom) & (self.data["id"] <= Ato)
            ].copy()

            # difference with painted DF
            self.bbb = tempDf.drop(self.painted.index, axis=0, errors="ignore")

            if self.bbb.shape[0] > 0:

                # enhanced painted dataDF
                self.painted = self.painted.append(self.bbb)

                # DRAW ALL
                self.plot.generatePicture(self.painted)

            self.setXRange(*currentRange, padding=0)
            self.setYRange(
                tempDf["Low"].min(), tempDf["High"].max(), padding=0,
            )


## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect()
## (see QGraphicsItem documentation)
class CandlestickGraphics(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.picture = QtGui.QPicture()

        self.w = 1 / 3.0
        self.ww = 1 / 2.0

        self.painter = None

        self.generatePicture(data)

    def generatePicture(self, data: pd.DataFrame):
        start = time.time()
        if data.shape[0] > 0:
            ## pre-computing a QPicture object allows paint() to run much more quickly,
            ## rather than re-drawing the shapes every time.

            # log.info(data)

            if self.painter is None:
                self.painter = QtGui.QPainter(self.picture)

            p = self.painter

            p.begin(self.picture)
            p.setPen(pg.mkPen(QtGui.QColor("black")))

            data.apply(
                lambda x: self.rowDoSomething(p, x, self.w, self.ww), axis=1
            )
            p.end()

            self.update()

        end = time.time()
        log.info(f"generatePicture takes: {end - start} sec.")

    def rowDoSomething(self, p, row, width1, width2):
        xAxis = row["id"] + width2
        if pd.isna(row["Open"]) == False:
            p.drawLine(
                QtCore.QPointF(xAxis, row["Low"]),
                QtCore.QPointF(xAxis, row["High"]),
            )
            if row["Open"] > row["Close"]:
                p.setBrush(pg.mkBrush(QtGui.QColor("red")))
            else:
                p.setBrush(pg.mkBrush(QtGui.QColor("green")))

            p.drawRect(
                QtCore.QRectF(
                    xAxis - width1,
                    row["Open"],
                    width1 * 2,
                    row["Close"] - row["Open"],
                )
            )

    def paint(self, p, *args):
        log.info("Running ...")
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())
