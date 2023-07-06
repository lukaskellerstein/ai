import logging
import threading
from typing import Any, Dict

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot

from ui.base.base_page import BasePage
from ui.services.realtime_data_service import RealtimeDataService
from ui.state.realtime_data import RealtimeDataItem
from business.model.asset import AssetType
from rx.core.typing import Disposable

# create logger
log = logging.getLogger("CellarLogger")


class RealtimeDataDebugPage(BasePage):
    subscriptions: Dict[str, Disposable] = {}

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/main/pages/debug/realtime_data/realtime_data.ui", self
        )

        # load styles
        with open(
            "ui/windows/main/pages/debug/realtime_data/realtime_data.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # apply styles
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.addWindow = None
        self.addButton.clicked.connect(self.add)
        self.logButton.clicked.connect(self.log)
        self.killallButton.clicked.connect(self.killall)

        self.realtimeService = RealtimeDataService()

    def add(self):
        ticker = self.ticker1Input.text().upper()
        self.__startRealtime(ticker)

    def __startRealtime(self, ticker):
        realtimeDataObjects: Dict[
            str, RealtimeDataItem
        ] = self.realtimeService.startRealtime(AssetType.STOCK, ticker)

        for key, rdo in realtimeDataObjects.items():
            subscriptionTemp = rdo.ticks.subscribe(self.logSubscribe)
            self.subscriptions[key] = subscriptionTemp

    @pyqtSlot()
    def logSubscribe(self, data):
        self.logTextEdit.append(str(data))

    @pyqtSlot()
    def log(self):
        self.logTextEdit.setText("")
        # realtime data state
        wholeState = self.realtimeService.state.stocks_realtime_data.getState()
        self.logTextEdit.append("----------LOG STATE --------")
        for key, rdi in wholeState.items():
            self.logTextEdit.append(key)
            self.logTextEdit.append(f"{rdi.symbol},{rdi.localSymbol}")
            self.logTextEdit.append(rdi.status.value)
        self.logTextEdit.append("----------------------------")

    def killall(self):
        self.realtimeService.state.stocks_realtime_data.stopAll()

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
