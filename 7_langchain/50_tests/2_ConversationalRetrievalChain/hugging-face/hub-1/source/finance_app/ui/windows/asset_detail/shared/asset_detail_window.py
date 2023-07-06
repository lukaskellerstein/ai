import logging
import time
from typing import Any, Callable, Type
from ui.windows.asset_detail.shared.pages.contract_details.contract_details import (
    ContractDetailsPage,
)

import rx.operators as ops
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from business.model.asset import Asset, AssetType
from business.model.contracts import IBStockContract
from business.modules.asset_bl import AssetBL
from ui.base.base_page import BasePage
from ui.windows.asset_detail.shared.pages.basic_info.basic_info import (
    BasicInfoPage,
)
from ui.windows.asset_detail.shared.pages.history_chart.history_chart import (
    HistoryChartPage,
)
from ui.windows.asset_detail.shared.pages.history_table.history_table import (
    HistoryTablePage,
)

# create logger
log = logging.getLogger("CellarLogger")


class AssetDetailWindow(QMainWindow):
    asset: Asset

    currentPage: BasePage

    on_update = pyqtSignal()

    def __init__(self, asset: Asset):
        super().__init__()

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/shared/asset_detail_window.ui", self
        )

        # load styles
        with open(
            "ui/windows/asset_detail/shared/asset_detail_window.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # apply styles
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.asset = asset

        self.fillHeaderBar()

        # MenuBar actions
        self.actionBasicInfo.triggered.connect(
            self.setCurrentPage(BasicInfoPage, data=self.asset)
        )
        self.actionTable.triggered.connect(
            self.setCurrentPage(HistoryTablePage, asset=self.asset)
        )
        self.actionChart.triggered.connect(
            self.setCurrentPage(HistoryChartPage, asset=self.asset)
        )
        self.actionContractDetails.triggered.connect(
            self.setCurrentPage(ContractDetailsPage, asset=self.asset)
        )

        # Stacket Widget
        self.pageBox.removeWidget(self.pageBox.widget(0))
        self.pageBox.removeWidget(self.pageBox.widget(0))

        self.currentPage = None
        self.setCurrentPage(HistoryChartPage, asset=self.asset)()

    def fillHeaderBar(self):
        self.secTypeLabel.setText(self.asset.type)
        self.setWindowTitle(
            f"{self.asset.symbol} - {self.asset.shortDescription}"
        )

    # def __pageOnUpdate(self):
    #     self.on_update.emit()

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

            self.currentPage.on_update.connect(self.on_update.emit)
            self.pageBox.addWidget(self.currentPage)
            self.pageBox.setCurrentIndex(0)

        return setPage

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # Qt destroy -----------------------------------------
    def closeEvent(self, event: Any):
        log.info("Running ...")
        self.currentPage.onDestroy()
        # self.bl.onDestroy()

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        self.currentPage.onDestroy()

        # self.bl.onDestroy()

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
