import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Union

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot

from business.model.asset import Asset
from business.model.timeframe import TimeFrame
from business.modules.asset_bl import AssetBL
from ui.base.base_page import BasePage
from ui.components.historical_data_table.table import HistoricalDataTable
from ui.windows.main.pages.assets.helpers import downloadFutures, updateFutures

# create logger
log = logging.getLogger("CellarLogger")


class FutureHistoryTablePage(BasePage):

    subscriptions = []
    lock = threading.Lock()

    asset: Asset

    timeframe = TimeFrame.day1

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/futures/pages/history_table/history_table.ui",
            self,
        )

        # load styles
        with open(
            "ui/windows/asset_detail/futures/pages/history_table/history_table.qss",
            "r",
        ) as fh:
            self.setStyleSheet(fh.read())

        # apply styles
        self.setAttribute(Qt.WA_StyledBackground, True)

        # INPUT data
        self.asset: Asset = kwargs["asset"]

        # BL
        self.bl: AssetBL = AssetBL()

        # UI
        localSymbols = []
        for cd in self.asset.contractDetails:
            localSymbols.append(
                {
                    "localSymbol": cd.contract.localSymbol,
                    "lastTradeDate": cd.contract.lastTradeDateOrContractMonth,
                }
            )
        localSymbols.sort(key=lambda x: x["lastTradeDate"])
        self.localSymbolComboBox.addItems(
            [f"{o['localSymbol']}-{o['lastTradeDate']}" for o in localSymbols]
        )

        localSymbol = self.localSymbolComboBox.currentText()
        self.getHistData(self.timeframe, localSymbol)

        self.progressBar.hide()
        self.__updateProgress(0)

        # signals
        self.updateButton.clicked.connect(self.updateData)
        self.downloadButton.clicked.connect(self.downloadData)
        self.localSymbolComboBox.currentTextChanged.connect(
            self.localSymbolComboBoxChanged
        )

    def getHistData(self, value: TimeFrame, localSymbol: str):
        self.data = self.bl.getHistoricalDataFromDB(localSymbol, value)

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
            self.table = HistoricalDataTable(self.data)
            self.gridLayout_2.addWidget(self.table, 4, 0, 1, 2)
        else:
            self.barCountLabel.setText("0")
            self.fromLabel.setText("")
            self.toLabel.setText("")

            self.updateButton.setDisabled(True)
            self.table = HistoricalDataTable(None)
            self.gridLayout_2.addWidget(self.table, 4, 0, 1, 2)

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

                # localSymbol = self.localSymbolComboBox.currentText().split(
                #     "-"
                # )[0]

                localSymbol = self.localSymbolComboBox.currentText()

                self.getHistData(self.timeframe, localSymbol)
        finally:
            self.lock.release()

    @pyqtSlot(str)
    def localSymbolComboBoxChanged(self, value: str):
        # localSymbol = value.split("-")[0]
        self.getHistData(self.timeframe, value)

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
