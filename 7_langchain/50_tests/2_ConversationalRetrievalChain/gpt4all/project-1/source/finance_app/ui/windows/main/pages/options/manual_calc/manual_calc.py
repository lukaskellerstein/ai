import logging
from typing import Any, Dict, Tuple

from PyQt5 import uic
from PyQt5.QtCore import Qt

from ui.base.base_page import BasePage

# create logger
log = logging.getLogger("CellarLogger")


class ManualCalcPage(BasePage):
    def __init__(self, *args: Tuple[str, Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/main/pages/options/manual_calc/manual_calc.ui", self
        )

        # # load styles
        # with open("ui/windows/main/pages/options/manual_calc/manual_calc.qss", "r") as fh:
        #     self.setStyleSheet(fh.read())

        # # apply styles
        # self.setAttribute(Qt.WA_StyledBackground, True)

        # load styles
        # with open(
        #     "ui/pages/futures_watchlist/futures_watchlist_page.qss", "r"
        # ) as fh:
        #     self.setStyleSheet(fh.read())

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # # 1. CUSTOM destroy -----------------------------------------
    # def onDestroy(self):
    #     log.info("Destroying ...")

    # # 2. Python destroy -----------------------------------------
    # def __del__(self):
    #     log.info("Running ...")
