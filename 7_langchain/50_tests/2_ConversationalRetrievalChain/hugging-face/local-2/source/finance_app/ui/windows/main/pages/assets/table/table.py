import logging
from typing import Tuple

import pandas as pd
from PyQt5.QtCore import QModelIndex, pyqtSignal
from PyQt5.QtWidgets import QHeaderView, QTableView

from ui.windows.main.pages.assets.table.table_model import AssetTableModel

# create logger
log = logging.getLogger("CellarLogger")


class AssetTable(QTableView):

    on_open = pyqtSignal(object, name="on_open")
    on_remove = pyqtSignal(object, name="on_remove")

    def __init__(self):
        super(QTableView, self).__init__()
        self.tableModel = AssetTableModel([])
        self.setModel(self.tableModel)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        self.setSortingEnabled(True)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setSelectionBehavior(QTableView.SelectRows)

        self.clicked.connect(self.myclick)
        self.doubleClicked.connect(self.mydoubleclick)

    def myclick(self, index):
        # delete column
        if index.column() == self.tableModel.columnCount(0) - 1:
            row = self.tableModel._data.iloc[index.row()]
            self.tableModel.removeAsset(index.row())
            self.on_remove.emit((row, index))

    def mydoubleclick(self, index):
        row = self.tableModel._data.iloc[index.row()]
        self.on_open.emit((row, index))
