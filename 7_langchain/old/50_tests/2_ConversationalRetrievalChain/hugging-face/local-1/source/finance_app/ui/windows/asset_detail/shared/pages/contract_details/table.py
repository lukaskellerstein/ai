from business.model.contract_details import IBContractDetails
import logging
from typing import Tuple, List

import pandas as pd
from PyQt5.QtWidgets import QHeaderView, QTableView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from business.model.asset import Asset

# create logger
log = logging.getLogger("CellarLogger")


class CDTable(QTableView):
    def __init__(self, asset: Asset):
        super(QTableView, self).__init__()
        self.setSortingEnabled(True)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setSelectionBehavior(QTableView.SelectRows)

        self.tableModel = QStandardItemModel()
        self.setData(asset.contractDetails)

        self.setModel(self.tableModel)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

    def setData(self, data: List[IBContractDetails]):
        self.tableModel.clear()

        if len(data) > 0:

            # add columns
            resultColumns = [
                "symbol",
                "localSymbol",
                "lastTradeDateOrContractMonth",
            ]
            self.tableModel.setHorizontalHeaderLabels(resultColumns)

            # add rows
            for cd in data:
                resultRow = []
                resultRow.append(QStandardItem(str(cd.contract.symbol)))
                resultRow.append(QStandardItem(str(cd.contract.localSymbol)))
                resultRow.append(
                    QStandardItem(
                        str(cd.contract.lastTradeDateOrContractMonth)
                    )
                )

                self.tableModel.appendRow(resultRow)

            self.tableModel.sort(2, Qt.DescendingOrder)

