import logging
import threading
from typing import Any

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot

from ui.base.base_page import BasePage

# create logger
log = logging.getLogger("CellarLogger")


class ThreadsDebugPage(BasePage):
    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi("ui/windows/main/pages/debug/threads/threads.ui", self)

        # load styles
        with open(
            "ui/windows/main/pages/debug/threads/threads.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # apply styles
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.addWindow = None
        self.logButton.clicked.connect(self.log)

    @pyqtSlot()
    def log(self):
        self.logTextEdit.setText("")

        self.logTextEdit.append(f"Threads count: {threading.active_count()}")
        self.logTextEdit.append("Thread names:")
        for thread in threading.enumerate():
            self.logTextEdit.append(f"- {thread.getName()}")

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
