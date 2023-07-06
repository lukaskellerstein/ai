import logging
from typing import List
import pandas as pd
from business.model.asset import AssetType
from business.modules.stocks_watchlist_bl import StocksWatchlistBL
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from ui.base.base_page import BasePage
from ui.services.realtime_data_service import RealtimeDataService
from ui.state.realtime_data import RealtimeDataItem
from ui.windows.main.pages.watchlists.stocks.table.table import StockTable
from typing import Dict
from rx.core.typing import Disposable
from helpers import constructKey
from ui.windows.asset_detail.stocks.stock_detail_window import (
    StockDetailWindow,
)

# create logger
log = logging.getLogger("CellarLogger")


class StocksWatchlistPage(BasePage):

    detailWindow = None

    tableSignal = pyqtSignal(dict)

    subscriptions: Dict[str, Disposable] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log.info("Running ...")

        self.bl = StocksWatchlistBL()
        self.realtimeService = RealtimeDataService()

        # load template
        uic.loadUi(
            "ui/windows/main/pages/watchlists/stocks/stocks_page.ui", self
        )

        # load styles
        with open(
            "ui/windows/main/pages/watchlists/stocks/stocks_page.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # Buttons
        self.startRealtime1Button.clicked.connect(self.addStockClick)
        self.loadSavedLayoutButton.clicked.connect(self.loadTableLayout)
        self.logButton.clicked.connect(self.log)

        # tableView
        self.table = StockTable()
        self.table.on_remove.connect(self.removeStock)
        self.table.on_open.connect(self.openStock)
        self.table.on_order_changed.connect(self.updateWatchlist)
        self.tableBox1.addWidget(self.table)

        # SIGNALS
        self.tableSignal.connect(self.table.tableModel.on_update_model)

        self.loadTableLayout()

    def __startRealtime(self, ticker):
        realtimeDataObjects: Dict[
            str, RealtimeDataItem
        ] = self.realtimeService.startRealtime(AssetType.STOCK, ticker)

        for key, rdi in realtimeDataObjects.items():
            subscriptionTemp = rdi.ticks.pipe().subscribe(
                self.tableSignal.emit
            )
            self.subscriptions[key] = subscriptionTemp

    def __stopRealtime(self, symbol):

        # self.realtimeService.stopRealtime(AssetType.STOCK, ticker, ticker)

        assetKey = constructKey(symbol, symbol)
        # Unsubscribe ticker
        for key, sub in self.subscriptions.items():
            if key == assetKey:
                log.info(f"Unsubscribing from {key}")
                sub.dispose()

        # remove the subscription
        self.subscriptions.pop(assetKey)

    def addStockClick(self):
        ticker: str = self.ticker1Input.text().upper()
        self.bl.addToWatchlist(ticker)
        self.__startRealtime(ticker)

    def removeStock(self, data):
        ticker = data.name
        self.__stopRealtime(ticker)
        self.table.tableModel.removeStock(ticker)
        self.bl.remove(ticker)

    def openStock(self, data: pd.Series):
        asset = self.bl.getAsset(AssetType.STOCK, data.name)
        self.detailWindow = StockDetailWindow(asset)
        self.detailWindow.show()

    def updateWatchlist(self, data):
        self.bl.updateStockWatchlist(data)

    def loadTableLayout(self):
        self.table.tableModel.reset()

        tickers = self.bl.getWatchlist()
        for ticker in tickers:
            self.__startRealtime(ticker)

    def log(self):
        # subscriptions
        for key, _ in self.subscriptions.items():
            log.info(key)

        # realtime data state
        wholeState = self.realtimeService.state.stocks_realtime_data.getState()
        for key, rdi in wholeState.items():
            log.info(key)
            log.info(f"{rdi.symbol},{rdi.localSymbol}")
            log.info(rdi.status)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # Unsubscribe everything
        for key, sub in self.subscriptions.items():
            log.info(f"Unsubscribing from {key}")
            sub.dispose()

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
