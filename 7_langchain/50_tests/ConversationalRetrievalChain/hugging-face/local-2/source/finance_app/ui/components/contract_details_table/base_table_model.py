import logging
from typing import List, Tuple

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt


# create logger
log = logging.getLogger("CellarLogger")


class BaseContractDetailsTableModel(QAbstractTableModel):

    _data: pd.DataFrame

    def __init__(self):
        super(QAbstractTableModel, self).__init__()

    # ----------------------------------------------------------
    # Minimum methods to override
    # ----------------------------------------------------------
    def data(self, index, role):
        if not index.isValid():
            print("index invalid - return None")
            return None

        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
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
