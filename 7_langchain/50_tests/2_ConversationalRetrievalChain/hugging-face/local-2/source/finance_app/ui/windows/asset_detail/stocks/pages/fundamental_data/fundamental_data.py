from business.modules.asset_bl import AssetBL
from ibapi.contract import Contract
import logging
from typing import Any

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal
from ui.base.base_page import BasePage
from business.model.asset import Asset

# create logger
log = logging.getLogger("CellarLogger")


class FundamentalDataPage(BasePage):
    asset: Asset

    on_fundamentals = pyqtSignal(str)

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/stocks/pages/fundamental_data/fundamental_data.ui",
            self,
        )

        # # load styles
        # with open(
        #     "ui/windows/asset_detail/shared/pages/basic_info/basic_info.qss",
        #     "r",
        # ) as fh:
        #     self.setStyleSheet(fh.read())

        # apply styles
        # self.setAttribute(Qt.WA_StyledBackground, True)

        # INPUT data
        self.asset = kwargs["data"]
        self.mainContractDetail = self.asset.contractDetails[0]

        # BL
        self.bl = AssetBL()

        self.on_fundamentals.connect(self.__setText)

        self.pushButton.clicked.connect(self.getFundamentals)

    def getFundamentals(self):
        self.bl.getFundamentals(self.mainContractDetail.contract).subscribe(
            self.__subscribe
        )

    def __subscribe(self, x):
        if isinstance(x, str):
            self.on_fundamentals.emit(x)

    def __setText(self, text):
        self.textEdit.setText(text)

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
