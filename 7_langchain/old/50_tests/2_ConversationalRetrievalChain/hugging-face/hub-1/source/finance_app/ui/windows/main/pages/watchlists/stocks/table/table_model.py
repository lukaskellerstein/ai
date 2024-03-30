from PyQt5 import uic
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon


import numpy as np
import pandas as pd

import logging

import time

from helpers import getColorByYieldValue

# create logger
log = logging.getLogger("CellarLogger")


def defaultValue(data):
    df = pd.DataFrame(
        data=data,
        columns=[
            "symbol",
            "bid_size",
            "bid",
            "last",
            "ask",
            "ask_size",
            "open",
            "high",
            "low",
            "close",
            "change",
            "volume",
            "avg_volume",
            "option_historical_vol",
            "option_implied_vol",
            "delete",
        ],
    )
    df.set_index("symbol", inplace=True)
    return df


class StockTableModel(QAbstractTableModel):

    # signal_update_models = pyqtSignal()

    def __init__(self, data):
        super(QAbstractTableModel, self).__init__()
        self._data = defaultValue(data)

    # ----------------------------------------------------------
    # Custom methods
    # ----------------------------------------------------------
    def __indexColumnsCount(self):
        return len(self._data.index.names)

    def __dataColumnsCount(self):
        return self._data.shape[1]

    def __allColumnsCount(self):
        return self._data.shape[1] + self.__indexColumnsCount()

    def reset(self):
        self.beginResetModel()
        self._data = defaultValue([])
        self.endResetModel()

    def getStocks(self):
        return self._data.index

    def addStock(self, ticker):
        self.beginInsertRows(
            QModelIndex(), self.rowCount(0) - 1, self.rowCount(0) - 1
        )
        self._data.loc[ticker] = np.zeros(self.__dataColumnsCount())
        self.endInsertRows()

    def insertStock(self, to_index, row):
        self.beginResetModel()

        # print("--------------------")
        # print("-------BEFORE---------")
        # print(self._data)
        # print("--------------------")
        # print("--------------------")

        self._data = self.__insert_row(to_index, row)

        # print("--------------------")
        # print("-------AFTER---------")
        # print(self._data)
        # print("--------------------")
        # print("--------------------")

        self.endResetModel()

    def __insert_row(self, idx, df_insert):
        log.debug(f"INSERT ROW - {idx}, {df_insert}")
        # print(df_insert)
        # print(type(df_insert))

        idx += 1
        dfA = self._data.iloc[
            :idx,
        ]
        dfB = self._data.iloc[
            idx:,
        ]

        # print(idx)

        # print("----dfA----")
        # print(dfA)
        # print("----dfB----")
        # print(dfB)

        idx_name = df_insert.name
        # print("-------Name---------")
        # print(idx_name)
        # print("--------------------")

        if idx_name in dfA.index:
            dfA = dfA.drop(idx_name, axis=0)

        if idx_name in dfB.index:
            dfB = dfB.drop(idx_name, axis=0)

        # print("----dfA----")
        # print(dfA)
        # print("----dfB----")
        # print(dfB)

        df2 = dfA.append(df_insert, ignore_index=False).append(dfB)
        # print("-------DF2---------")
        # print(df2)
        # print("--------------------")

        return df2

    def removeStock(self, ticker):
        self.beginRemoveRows(
            QModelIndex(), self.rowCount(0) - 1, self.rowCount(0) - 1
        )
        self._data = self._data.drop(ticker, axis=0)
        self.endRemoveRows()

    # ----------------------------------------------------------
    # Signals / Slots
    # ----------------------------------------------------------
    @pyqtSlot(dict, name="on_update_model")
    def on_update_model(self, obj):
        if obj != {}:

            # start = time.time()

            atype = obj["type"]
            aticker = obj["ticker"]
            aprice = obj["price"]

            # if (atype in self._data.columns) == False:
            #     print("_______________________")
            #     print(atype)
            #     print("_______________________")

            if (aticker in self._data.index) == False:
                self.addStock(aticker)

            if (atype in defaultValue([]).columns) == True:
                self._data.loc[aticker, atype] = aprice

                if atype == "close" or atype == "last":
                    # change column - calculation
                    closeP = self._data.loc[aticker, "close"]
                    lastP = self._data.loc[aticker, "last"]
                    if closeP > 0 and lastP > 0:
                        self._data.loc[aticker, "change"] = round(
                            ((lastP - closeP) / closeP) * 100, 1
                        )
                    else:
                        self._data.loc[aticker, "change"] = 0

                # VAR 1. -----------------------------------
                # bbb = QModelIndex()
                # bbb.row = self._data.index.get_loc(aticker)
                # bbb.column = self._data.columns.get_loc(atype)
                # self.dataChanged.emit(bbb, bbb)  # <---

                bbb = QModelIndex()
                bbb.row = 1
                bbb.column = 1
                self.dataChanged.emit(bbb, bbb)

                # VAR 2. ----------------------------------

                # rowCount = self._data.shape[0] - 1
                # columnCount = self._data.shape[1] - 1

                # aaa = QModelIndex()
                # aaa.row = 0
                # aaa.column = 0

                # bbb = QModelIndex()
                # bbb.row = rowCount - 1
                # bbb.column = columnCount - 1

                # self.dataChanged.emit(aaa, bbb)

            # ------------------------------------------

            # end = time.time()
            # print(f"on_update_model - {aticker} - {(end - start) * 10000}sec.")

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

            indexColumnsCount = self.__indexColumnsCount()

            if index.column() < indexColumnsCount:

                # print(self._data.index)
                # print(self._data.index.values[index.row()])
                value = self._data.index.values[index.row()]
                # print(value)
                # print("--")
                return str(value)
            else:
                value = self._data.iloc[
                    index.row(), index.column() - indexColumnsCount
                ]
                columnIndex = index.column()

                if columnIndex == 1:  # bid-size
                    return f"{value:.0f}"
                elif columnIndex == 5:  # ask-size
                    return f"{value:.0f}"
                elif columnIndex == 10:  # change
                    return f"{value:.1f} %"
                elif columnIndex == 11:  # volume
                    return f"{value:.0f}"
                elif columnIndex == 12:  # avg volume
                    return f"{value:.0f}"
                elif columnIndex == 13:  # hist volatility
                    return f"{value:.2f}"
                elif columnIndex == 14:  # implied volatility
                    return f"{value:.2f}"
                elif columnIndex == 15:  # delete
                    return None
                else:
                    return str(value)

        if role == Qt.TextAlignmentRole:
            columnIndex = index.column()

            if columnIndex == 3:  # Last
                return Qt.AlignVCenter + Qt.AlignHCenter

        if role == Qt.FontRole:
            columnIndex = index.column()

            if columnIndex == 3:  # Last
                return QFont("Bold")

        if role == Qt.BackgroundRole:
            indexColumnsCount = self.__indexColumnsCount()
            columnIndex = index.column()

            if columnIndex == 3:  # Last
                return QColor("blue")

            # "change" column
            elif columnIndex == 10:  # change
                value = self._data.iloc[
                    index.row(), columnIndex - indexColumnsCount
                ]
                color = getColorByYieldValue(value)
                return QColor(color)

                # if value > 0:
                #     return QColor("green")
                #     # QWidget.setProperty("cssClass", [ "profit-color" ])
                # elif value == 0:
                #     return QColor("blue")
                # else:
                #     return QColor("red")

        if role == Qt.ForegroundRole:
            columnIndex = index.column()

            if columnIndex == 3:  # Last
                return QColor("white")

        if role == Qt.DecorationRole:
            columnIndex = index.column()

            # "delete" column
            if columnIndex == 15:  # delete
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
        return self.__allColumnsCount()

    def headerData(self, section, orientation, role):
        # print("------------------headerData--------------")
        # print(self._data)
        # print("--------------------------------------------")
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:

                indexColumnsCount = self.__indexColumnsCount()

                if section > indexColumnsCount - 1:
                    return str(self._data.columns[section - indexColumnsCount])
                else:
                    # print(section)
                    # print(self._data.index.names)
                    # print(self._data.index.names[section])
                    return str(self._data.index.names[section])

            # if orientation == Qt.Vertical:
            #     return str(self._data.index[section])

    def flags(self, index):
        return (
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )

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
