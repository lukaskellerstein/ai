import logging
import threading
import time
from datetime import datetime
from typing import Any, Union
from ui.windows.main.pages.assets.helpers import downloadStock, updateStock

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot

from business.model.asset import Asset, AssetType
from business.model.timeframe import TimeFrame
from business.modules.asset_bl import AssetBL
from ui.base.base_page import BasePage
from ui.components.historical_data_table.table import HistoricalDataTable

from business.helpers import getTimeBlocks

# create logger
log = logging.getLogger("CellarLogger")


class HistoryTablePage(BasePage):

    subscriptions = []
    lock = threading.Lock()

    asset: Asset

    timeframe = TimeFrame.day1

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/shared/pages/history_table/history_table.ui",
            self,
        )

        # load styles
        with open(
            "ui/windows/asset_detail/shared/pages/history_table/history_table.qss",
            "r",
        ) as fh:
            self.setStyleSheet(fh.read())

        # apply styles
        self.setAttribute(Qt.WA_StyledBackground, True)

        # INPUT data
        self.asset: Asset = kwargs["asset"]

        # BL
        self.bl: AssetBL = AssetBL()

        # signals
        self.updateButton.clicked.connect(self.updateData)
        self.downloadButton.clicked.connect(self.downloadData)

        self.progressBar.hide()
        self.__updateProgress(0)

        self.table = None
        self.getHistData(self.timeframe)

    def getHistData(self, timeframe: TimeFrame):
        self.data = self.bl.getHistoricalDataFromDB(
            self.asset.symbol, timeframe
        )

        if self.data is not None:
            # start = time.time()
            self.data = self.data.sort_index()
            # end = time.time()
            # log.info(f"takes {end - start} sec.")

            # check duplications
            # dupl = self.data.duplicated()
            # allresults = dupl[dupl == True]

            self.barCountLabel.setText(str(self.data.shape[0]))
            self.fromLabel.setText(
                self.data.head(1).index[0].strftime("%Y%m%d %H:%M:%S")
            )
            self.toLabel.setText(
                self.data.tail(1).index[0].strftime("%Y%m%d %H:%M:%S")
            )

            self.updateButton.setDisabled(False)

            if self.table is not None:
                self.table.setData(self.data)
            else:
                self.table = HistoricalDataTable(self.data)
                self.gridLayout_2.addWidget(self.table, 3, 0, 1, 2)
        else:
            self.barCountLabel.setText("0")
            self.fromLabel.setText("")
            self.toLabel.setText("")

            self.updateButton.setDisabled(True)

            if self.table is not None:
                self.table.setData(self.data)
            else:
                self.table = HistoricalDataTable(None)
                self.gridLayout_2.addWidget(self.table, 3, 0, 1, 2)

    @pyqtSlot()
    def updateData(self):
        self.progressBar.show()

        subscriptionTemp = self.bl.updateHistoricalData(
            [self.asset]
        ).subscribe(self.__updateProgress)

        self.subscriptions.append(subscriptionTemp)

    @pyqtSlot()
    def downloadData(self):
        self.progressBar.show()

        subscriptionTemp = self.bl.downloadHistoricalData(
            [self.asset]
        ).subscribe(self.__updateProgress)

        self.subscriptions.append(subscriptionTemp)

    @pyqtSlot(int)
    def __updateProgress(self, value: int):
        self.lock.acquire()
        print(f"Progress: {value} %")
        try:
            self.progressBar.setValue(value)

            if value == 100:
                self.progressBar.hide()
                self.getHistData(self.timeframe)

        finally:
            self.lock.release()

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # Unsubscribe everything
        for sub in self.subscriptions:
            sub.dispose()

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
