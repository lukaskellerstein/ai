import logging
from typing import Any, Dict, List, Tuple
from ui.windows.asset_detail.futures.future_detail_window import (
    FutureDetailWindow,
)

import pandas as pd
from business.model.asset import AssetType
from business.modules.asset_bl import AssetBL
from business.modules.futures_watchlist_bl import FuturesWatchlistBL
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from ui.base.base_page import BasePage
from ui.services.realtime_data_service import RealtimeDataService
from ui.state.realtime_data import RealtimeDataItem
from ui.windows.main.pages.watchlists.futures.table.tree import FuturesTree
from ui.windows.main.pages.watchlists.futures.table.tree_model import (
    FuturesTreeNode,
)
from rx.core.typing import Disposable
from helpers import constructKey

# create logger
log = logging.getLogger("CellarLogger")


class FuturesWatchlistPage(BasePage):

    detailWindow = None

    treeSignal = pyqtSignal(dict)

    subscriptions: Dict[str, Disposable] = {}

    def __init__(self, *args: Tuple[str, Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        log.info("Running ...")

        self.bl = FuturesWatchlistBL()
        self.realtimeService = RealtimeDataService()
        self.assetBL = AssetBL()

        # load template
        uic.loadUi(
            "ui/windows/main/pages/watchlists/futures/futures_page.ui", self
        )

        # load styles
        with open(
            "ui/windows/main/pages/watchlists/futures/futures_page.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # Buttons
        self.startRealtime1Button.clicked.connect(self.addFutureClick)
        self.loadSavedLayoutButton.clicked.connect(self.loadTableLayout)

        # treeView
        self.tree = FuturesTree()
        self.tree.on_remove.connect(self.removeFuture)
        self.tree.on_open.connect(self.openFuture)
        self.tree.on_order_changed.connect(self.updateWatchlist)
        self.tableBox1.addWidget(self.tree)

        # SIGNALS
        # self.treeSignal.connect(self.tree.tree_model.on_update_model)

        self.loadTableLayout()

    def __startRealtime(self, ticker: str):
        cds = self.assetBL.getLatestContractDetails(
            AssetType.FUTURE, ticker, 3
        )

        if not cds:
            log.warn(
                f"Asset with symbol: {ticker} has no valid ContractDetails in Asset DB"
            )
        else:
            # Update table
            self.tree.tree_model.addGroup(cds)

            # Start realtime data
            # realtimeDataObjects: Dict[
            #     str, RealtimeDataItem
            # ] = self.realtimeService.startRealtime(AssetType.FUTURE, ticker, 3)

            # for key, rdi in realtimeDataObjects.items():
            #     subscriptionTemp = rdi.ticks.subscribe(self.treeSignal.emit)
            #     self.subscriptions[key] = subscriptionTemp

    def __stopRealtime(self, symbol):
        cds = self.assetBL.getLatestContractDetails(
            AssetType.FUTURE, symbol, 3
        )

        if not cds:
            log.warn(
                f"Asset with symbol: {symbol} has no valid ContractDetails in Asset DB"
            )
        else:
            for cd in cds:

                # self.realtimeService.stopRealtime(AssetType.FUTURE, symbol, localSymbol)

                assetKey = constructKey(
                    cd.contract.symbol, cd.contract.localSymbol
                )
                # Unsubscribe ticker
                for key, sub in self.subscriptions.items():
                    if key == assetKey:
                        log.info(f"Unsubscribing from {key}")
                        sub.dispose()

                # remove the subscription
                self.subscriptions.pop(assetKey)

    def addFutureClick(self):
        ticker: str = self.ticker1Input.text().upper()
        self.bl.addToWatchlist(ticker)
        self.__startRealtime(ticker)

    def removeFuture(self, node: FuturesTreeNode):
        symbol: str = node.data.index.values[0][0]
        # localSymbol: str = node.data.index.values[0][1]
        self.__stopRealtime(symbol)
        self.tree.tree_model.removeFuture(symbol)
        self.bl.remove(symbol)

    def openFuture(self, data: Any):
        symbol = data.data.index[0][0]
        asset = self.bl.getAsset(AssetType.FUTURE, symbol)
        self.detailWindow = FutureDetailWindow(asset)
        self.detailWindow.show()

    def updateWatchlist(self, data):
        log.info(data)
        self.bl.updateWatchlist(data)

    def loadTableLayout(self):
        # self.tree.tree_model.reset()

        tickers: List[str] = self.bl.getWatchlist()
        for ticker in tickers:
            self.__startRealtime(ticker)

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
