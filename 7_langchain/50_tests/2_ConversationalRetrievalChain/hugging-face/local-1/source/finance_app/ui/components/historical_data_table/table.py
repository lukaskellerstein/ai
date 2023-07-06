import logging
from typing import Tuple

import pandas as pd
from PyQt5.QtWidgets import QHeaderView, QTableView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt

# create logger
log = logging.getLogger("CellarLogger")


class HistoricalDataTable(QTableView):
    def __init__(self, data: pd.DataFrame):
        super(QTableView, self).__init__()
        self.setSortingEnabled(True)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setSelectionBehavior(QTableView.SelectRows)

        self.tableModel = QStandardItemModel()
        self.setData(data)

        self.setModel(self.tableModel)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

    def setData(self, data: pd.DataFrame):
        self.tableModel.clear()
        if data is not None:
            # add columns
            resultColumns = []
            resultColumns.append(data.index.name)
            for col in data.columns:
                resultColumns.append(col)

            self.tableModel.setHorizontalHeaderLabels(resultColumns)

            # add rows
            for _, row in data.iterrows():
                resultRow = []
                resultRow.append(
                    QStandardItem(str(row.name.strftime("%Y%m%d %H:%M:%S")))
                )
                for _, cell in row.items():
                    resultRow.append(QStandardItem(str(cell)))

                self.tableModel.appendRow(resultRow)

            self.tableModel.sort(0, Qt.DescendingOrder)
