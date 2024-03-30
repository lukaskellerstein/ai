from ui.windows.main.pages.debug.realtime_data.realtime_data import (
    RealtimeDataDebugPage,
)
from ui.windows.main.pages.options.manual_calc.manual_calc import (
    ManualCalcPage,
)
from ui.windows.main.pages.debug.threads.threads import ThreadsDebugPage
from business.model.asset import AssetType
import logging
import logging.config
import sys
from typing import Any, Callable, Dict, Tuple, Type

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow

import helpers as helpers
import resources  # RESOURCES FOR APP
from ui.base.base_page import BasePage
from ui.windows.main.pages.assets.asset_page import AssetPage
from ui.windows.main.pages.home.home_page import HomePage
from ui.windows.main.pages.watchlists.futures.futures_page import (
    FuturesWatchlistPage,
)
from ui.windows.main.pages.watchlists.options.options_page import (
    OptionsWatchlistPage,
)
from ui.windows.main.pages.watchlists.stocks.stocks_page import (
    StocksWatchlistPage,
)

# set logging from config file
logging.config.fileConfig("logging.conf")

# create logger
log = logging.getLogger("CellarLogger")


class MainWindow(QMainWindow):

    currentPage: BasePage

    def __init__(self, *args: Tuple[str, Any], **kwargs: Dict[str, Any]):
        super().__init__()
        log.info("Running ...")

        uic.loadUi("./ui/windows/main/main_window.ui", self)

        # load styles
        with open("./ui/windows/main/main_window.qss", "r") as fh:
            self.setStyleSheet(fh.read())

        # MenuBar actions
        self.actionHomePage.triggered.connect(self.setCurrentPage(HomePage))
        self.actionStocksWatchlist.triggered.connect(
            self.setCurrentPage(StocksWatchlistPage)
        )
        self.actionFuturesWatchlist.triggered.connect(
            self.setCurrentPage(FuturesWatchlistPage)
        )
        self.actionOptionsWatchlist.triggered.connect(
            self.setCurrentPage(OptionsWatchlistPage)
        )
        self.actionStocksAsset.triggered.connect(
            self.setCurrentPage(AssetPage, assetType=AssetType.STOCK)
        )
        self.actionFuturesAsset.triggered.connect(
            self.setCurrentPage(AssetPage, assetType=AssetType.FUTURE)
        )
        self.actionThreads.triggered.connect(self.openThreadsDebugWindow)
        self.actionRealtimeData.triggered.connect(
            self.openRealtimeDataDebugWindow
        )
        self.actionManual_Calc.triggered.connect(
            self.setCurrentPage(ManualCalcPage)
        )

        # Stacket Widget
        self.pageBox.removeWidget(self.pageBox.widget(0))
        self.pageBox.removeWidget(self.pageBox.widget(0))

        self.currentPage = None
        self.setCurrentPage(HomePage)()

    # --------------
    # HOF - High Ordered Function -> returns function
    # --------------
    def setCurrentPage(
        self, page: Type[BasePage], **kwargs: Any
    ) -> Callable[[], None]:
        def setPage():
            if self.currentPage is not None:
                self.pageBox.removeWidget(self.currentPage)
                self.currentPage.onDestroy()

            if kwargs is not None:
                self.currentPage = page(**kwargs)
            else:
                self.currentPage = page()
            self.pageBox.addWidget(self.currentPage)
            self.pageBox.setCurrentIndex(0)

        return setPage

    # --------------
    # Others
    # --------------
    def openThreadsDebugWindow(self):
        self.threadsDebugWindowInstance = ThreadsDebugPage()
        self.threadsDebugWindowInstance.show()

    def openRealtimeDataDebugWindow(self):
        self.realtimeDataDebugWindowInstance = RealtimeDataDebugPage()
        self.realtimeDataDebugWindowInstance.show()

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # Qt destroy -----------------------------------------
    def closeEvent(self, event: Any):
        log.info("Running ...")
        helpers.logThreads()
        self.currentPage.onDestroy()

    # Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
        helpers.logThreads()
