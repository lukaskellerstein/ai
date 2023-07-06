from business.model.contract_details import IBContractDetails
from PyQt5 import uic
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon

from typing import DefaultDict, List, Tuple
import numpy as np
import pandas as pd

import logging

import time

from helpers import getColorByYieldValue

from business.model.asset import Asset

# create logger
log = logging.getLogger("CellarLogger")


def defaultValue(data):
    df = pd.DataFrame(
        data=data,
        columns=["symbol", "shortDescription", "contractDetails", "delete"],
    )
    return df


class AssetTableModel(QAbstractTableModel):

    _dataOrig: pd.DataFrame
    _data: pd.DataFrame

    def __init__(self, data: List[Asset]):
        super(QAbstractTableModel, self).__init__()
        self._data = defaultValue(data)
        self._dataOrig = defaultValue(data)
        self.__columnCount = self._data.shape[1]

    # ----------------------------------------------------------
    # Custom methods
    # ----------------------------------------------------------
    def setData(self, data: List[Asset]):
        self.beginResetModel()

        aaa = list(
            map(
                lambda x: (
                    x.symbol,
                    x.shortDescription,
                    len(x.contractDetails),
                    None,  # delete
                ),
                data,
            )
        )

        self._data = defaultValue(aaa)
        self._dataOrig = defaultValue(aaa)
        self.endResetModel()

    def removeAsset(self, index):
        self.beginRemoveRows(
            QModelIndex(), self.rowCount(0) - 1, self.rowCount(0) - 1
        )
        self._data = self._data.drop(index, axis=0)
        self.endRemoveRows()

    def filterData(self, searchText: str):
        self.beginResetModel()

        if searchText != "":
            self._data = self._dataOrig[
                (self._dataOrig["symbol"].str.contains(searchText, case=False))
                | (
                    self._dataOrig["shortDescription"].str.contains(
                        searchText, case=False
                    )
                )
            ]
        else:
            self._data = self._dataOrig

        self.endResetModel()

    # ----------------------------------------------------------
    # Minimum methods to override
    # ----------------------------------------------------------
    def data(self, index, role):
        # log.debug("Running...")
        # log.debug(locals())

        if not index.isValid():
            print("index invalid - return None")
            return None

        if role == Qt.DisplayRole:
            # print("--")
            # print("index")
            # print(index)
            # print(index.row())
            # print(index.column())
            # print(index.model())
            # print("--")
            columnIndex = index.column()

            if columnIndex == self.__columnCount - 1:  # delete
                return None
            else:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)

        if role == Qt.DecorationRole:
            columnIndex = index.column()

            # "delete" column
            if columnIndex == self.__columnCount - 1:  # delete
                return QIcon(":/assets/delete-icon")

    def rowCount(self, index):
        # log.debug("Running...")
        # log.debug(locals())
        # print(traceback.print_stack(file=sys.stdout))
        # print("--")
        # print("index")
        # print(index.isValid())
        # print(index.row())
        # print(index.column())
        # print(index.model())
        # print("--")
        return self._data.shape[0]

    def columnCount(self, index):
        # log.debug("Running...")
        # log.debug(locals())
        # print(traceback.print_stack(file=sys.stdout))
        # print("--")
        # print("index")
        # print(index.isValid())
        # print(index.row())
        # print(index.column())
        # print(index.model())
        # print("--")
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # print("------------------headerData--------------")
        # print(self._data)
        # print("--------------------------------------------")
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

    def sort(self, Ncol, order):
        """Sort table by given column number.
        """
        try:
            self.layoutAboutToBeChanged.emit()
            self._data = self._data.sort_values(
                self._data.columns[Ncol], ascending=order
            )
            self.layoutChanged.emit()
        except Exception as e:
            print(e)
