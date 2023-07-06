from __future__ import annotations
from business.model.contract_details import (
    IBContractDetails,
)  # allow return same type as class ..... -> FuturesTreeNode
import math
import logging
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from PyQt5.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    pyqtSlot,
    QVariant,
)

from ibapi.contract import ContractDetails

# create logger
log = logging.getLogger("CellarLogger")


def defaultValue(data: Union[List[List[Any]], None]) -> pd.DataFrame:
    df = pd.DataFrame(
        data=data,
        columns=[
            "order",
            "symbol",
            "localSymbol",
            "contractMonth",
            "contractEndDate",
            "parentId",
            "diff",
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
        ],
    )
    # df.set_index(["symbol", "localSymbol"], inplace=True)
    return df


# header: List[str] = [
#     "symbol",
#     "localSymbol",
#     "contractMonth",
#     "contractEndDate",
#     "diff",
#     "bid_size",
#     "bid",
#     "last",
#     "ask",
#     "ask_size",
#     "open",
#     "high",
#     "low",
#     "close",
#     "change",
#     "volume",
#     "avg_volume",
#     "option_historical_vol",
#     "option_implied_vol",
#     "delete",
# ]


class FuturesTreeModel(QAbstractItemModel):
    def __init__(self):
        super().__init__()
        self._data = defaultValue([])
        self.firstLevel = defaultValue([])

        # header
        # for index, value in enumerate(header):
        #     self.setHeaderData(index, Qt.Horizontal, value)

    # ----------------------------------------------------------
    # Custom methods
    # ----------------------------------------------------------

    # def __indexColumnsCount(self):
    #     return len(self.root.data.index.names)

    # def __dataColumnsCount(self) -> int:
    #     return self._root.data.shape[1]

    # def __allColumnsCount(self) -> int:
    #     return self._root.data.shape[1] + self.__indexColumnsCount()

    def addGroup(
        self, data: Union[List[ContractDetails], List[IBContractDetails]]
    ):
        # log.debug("Running...")
        # log.debug(locals())

        # ITEMS ----------------------------------
        firstIndex: int = -1
        for row in data:

            # add new row to the tree
            self.beginInsertRows(QModelIndex(), 0, 0)

            # find last "order" number
            lastOrderNumber = self._data["order"].max()
            if math.isnan(lastOrderNumber):
                lastOrderNumber = 0

            vals: List[Any] = [
                lastOrderNumber + 1,
                row.contract.symbol,
                row.contract.localSymbol,
                row.contractMonth,
                row.contract.lastTradeDateOrContractMonth,
                firstIndex,
                *np.zeros(15),
            ]
            vals_series = pd.Series(vals, index=self._data.columns)

            self._data = self._data.append(vals_series, ignore_index=True)

            # if this is parent of the group
            if firstIndex == -1:
                parentRow = self._data[
                    (self._data["symbol"] == row.contract.symbol)
                    & (self._data["localSymbol"] == row.contract.localSymbol)
                ]
                if parentRow.empty:
                    raise Exception("THIS SHOULD NOT HAPPENED")
                else:
                    firstIndex = parentRow.index[0]
                    self.firstLevel = self.firstLevel.append(
                        vals_series, ignore_index=True
                    )

            self.endInsertRows()

    # def reset(self):
    #     self.beginResetModel()
    #     self.root = FuturesTreeNode(None)
    #     self.endResetModel()

    def removeFuture(self, ticker: str):
        self.beginRemoveRows(QModelIndex(), 0, 0)
        self._data = self._data[self._data["symbol"] != ticker]
        self.endRemoveRows()

    # ----------------------------------------------------------
    # Signals / Slots
    # ----------------------------------------------------------
    # @pyqtSlot(dict, name="on_update_model")
    # def on_update_model(self, obj: Dict[str, Any]):
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     log.debug(
    #         f"ticker={obj['ticker']}|localSymbol={obj['localSymbol']}|type={obj['type']}|price={obj['price']}"
    #     )

    #     if obj == {}:
    #         log.info("EMPTY")
    #         return

    #     self.root.updateData(
    #         obj["ticker"], obj["localSymbol"], obj["type"], obj["price"]
    #     )

    #     bbb = QModelIndex()
    #     bbb.row = 1
    #     bbb.column = 1

    #     # self.dataChanged.emit(QModelIndex(), QModelIndex())
    #     self.dataChanged.emit(bbb, bbb)

    # ----------------------------------------------------------
    # Minimum methods to override
    # ----------------------------------------------------------
    def data(self, index: QModelIndex, role: Any) -> Union[str, None]:
        # log.debug("Running...")
        # log.debug(locals())

        # if not index.isValid():
        #     log.error("index invalid - return None")
        #     return None

        if role == Qt.DisplayRole:

            if index.parent().row() != -1:
                log.info(
                    f"index.row: {index.row()}, index.col: {index.column()}, index.parent.row: {index.parent().row()}, index.parent.col: {index.parent().column()}"
                )

            columnIndex: int = index.column()
            rowIndex: int = index.row()

            value = self.firstLevel.iloc[rowIndex, columnIndex]

            # value = self._data.iloc[rowIndex, columnIndex]

            if columnIndex == 3:  # contract-end-date
                return value
            elif columnIndex == 6:  # diff
                return f"{value:.0f}"
            elif columnIndex == 7:  # bid-size
                return f"{value:.0f}"
            elif columnIndex == 8:  # bid
                return f"{value:.2f}"
            if columnIndex == 9:  # last
                return f"{value:.2f}"
            elif columnIndex == 10:  # ask
                return f"{value:.2f}"
            elif columnIndex == 11:  # ask-size
                return f"{value:.0f}"
            elif columnIndex == 12:  # open
                return f"{value:.2f}"
            elif columnIndex == 13:  # high
                return f"{value:.2f}"
            elif columnIndex == 14:  # low
                return f"{value:.2f}"
            elif columnIndex == 15:  # close
                return f"{value:.2f}"
            # elif columnIndex == 14:  # change
            #     return f"{value:.1f}"
            elif columnIndex == 17:  # volume
                return f"{value:.0f}"
            elif columnIndex == 18:  # avg volume
                return f"{value:.0f}"
            elif columnIndex == 19:  # hist volatility
                return f"{value:.2f}"
            elif columnIndex == 20:  # implied volatility
                return f"{value:.2f}"

            return str(value)

        return None

    def rowCount(self, parent: QModelIndex) -> int:
        # log.debug("Running...")
        # log.debug(locals())

        # log.info(
        #     f"isValid: {parent.isValid()}, parent.row: {parent.row()}, parent.col: {parent.column()}"
        # )

        if self._data.empty:
            return 0

        if parent.isValid():
            # number of children in parent
            childRowsCount = self._data[
                self._data["parentId"] == parent.row()
            ].shape[0]
            return childRowsCount
        else:
            # all data
            # return self._data.shape[0]
            # # number of parents
            return self._data[self._data["parentId"] == -1].shape[0]

    def columnCount(self, parent: QModelIndex) -> int:
        # log.debug("Running...")
        # log.debug(locals())

        if self._data.empty:
            return 0

        return self._data.shape[1]

        # if parent.isValid():

        #     print(parent)

        #     return 0
        # else:
        #     return self._data[self._data["parentId"] == -1].shape[1]

    def index(self, row: int, col: int, parent: QModelIndex) -> QModelIndex:
        # log.info("Running...")
        # # log.info(locals())
        # log.info(
        #     f"row: {row}, col: {col}, parent.row: {parent.row()}, parent.col: {parent.column()}"
        # )

        # if col > 0:
        #     return QModelIndex()

        # return self.createIndex(row, col, None)

        # item = self._data.iloc[
        #     row,
        # ]
        # return self.createIndex(row, col, item)

        if not parent.isValid():
            # log.info("not parent.isValid()")
            item = self.firstLevel.iloc[
                row,
            ]
            return self.createIndex(row, col, item["order"])
        else:
            log.info("is parent.isValid()")
            # if row == 0 and parent.row() == 0:
            #     i = 6
            parentDf = self.firstLevel.iloc[parent.row()]
            item = self._data[self._data["parentId"] == parentDf.name].iloc[
                row, col
            ]

            return self.createIndex(row, col, item["order"])

        # if not self.hasIndex(row, col, parent):
        #     return QModelIndex()

        # if parent.column() != 0:
        #     return QModelIndex()

        # if parent.isValid() and parent.column() != 0:
        #     return QModelIndex()

        onlyParents = self._data[self._data["parentId"] == -1]

        item = onlyParents.iloc[
            row,
        ]
        return self.createIndex(row, col, item)

        if not parent.isValid():
            item = onlyParents.iloc[
                row,
            ]
            return self.createIndex(row, col, item)
        else:
            item = self._data[self._data["parentId"] == parent.row()].loc[
                row,
            ]
            return self.createIndex(row, col, item)

        # OLD
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()

        if not self.hasIndex(row, col, parent):
            return QModelIndex()

        if not parent or not parent.isValid():

            # index = (
            #     self._data[self._data["parentId"] == -1]
            #     .reset_index()
            #     .iloc[row,]
            #     .name
            # )

            index = self._data[self._data["parentId"] == -1].iloc[row,].name

            return QAbstractItemModel.createIndex(self, index, col, None)

        else:

            # index = (
            #     self._data[self._data["parentId"] == parent.row()]
            #     .reset_index()
            #     .iloc[row,]
            #     .name
            # )

            index = (
                self._data[self._data["parentId"] == parent.row()]
                .reset_index()
                .iloc[row,]["index"]
            )
            return QAbstractItemModel.createIndex(self, index, col, None)

            return QModelIndex()

        dfRow = self._data.loc[row]
        return QAbstractItemModel.createIndex(self, row, col, dfRow)

        # if not _parent or not _parent.isValid():
        #     parent = self.root
        # else:
        #     parent = _parent.internalPointer()

        # if not QAbstractItemModel.hasIndex(self, row, col, _parent):
        #     return QModelIndex()

        # child = parent.child(row)
        # if child:
        #     return QAbstractItemModel.createIndex(self, row, col, child)
        # else:
        #     return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        # log.debug("Running...")
        # log.debug(locals())

        # log.info(f"index.row: {index.row()}, index.col: {index.column()}")

        # parentId = self._data.iloc[index.row(),]["parentId"]
        parentId = self.firstLevel.iloc[index.row(),]["parentId"]

        if parentId == -1:
            return QModelIndex()
        else:
            parent = self._data.iloc[
                parentId,
            ]
            return self.createIndex(parentId, 0, parent)

        # # OLD 1
        # if not index.isValid():
        #     return QModelIndex()

        # onlyParents = self._data[self._data["parentId"] == -1]

        # parentId = onlyParents.iloc[index.row(),]["parentId"]

        # if parentId == -1:
        #     return QModelIndex()
        # else:
        #     parent = onlyParents.iloc[
        #         parentId,
        #     ]
        #     return self.createIndex(parentId, 0, parent)

        # # OLD
        # if index.isValid():

        #     parentId = self._data.iloc[index.row(),]["parentId"]

        #     if parentId == -1:
        #         return QAbstractItemModel.createIndex(self, -1, -1, None)

        #     parent = self._data.iloc[
        #         parentId,
        #     ]

        #     return QAbstractItemModel.createIndex(self, parentId, 0, parent)

        # else:
        #     return QModelIndex()
        # raise Exception("??????????")
        # return QAbstractItemModel.createIndex(self, -1, -1, None)

    # ----------------------------------------------------------
    # Override - headers
    # ----------------------------------------------------------

    # def setHeaderData(
    #     self,
    #     section: int,
    #     orientation: Qt.Orientation,
    #     value: QVariant,
    #     role=Qt.DisplayRole,
    # ) -> bool:
    #     # log.debug("Running...")
    #     # log.debug(locals())

    #     if orientation == Qt.Horizontal:

    #         # print("Setting value: " + value)
    #         self.header_data[section] = value

    #         return True

    #     else:

    #         aaa = super(FuturesTreeModel, self).setHeaderData(
    #             section, orientation, value, role
    #         )
    #         # print(aaa)
    #         return aaa

    def headerData(
        self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole
    ) -> object:
        # log.debug("Running...")
        # log.debug(locals())

        # print(section)

        if role == Qt.DisplayRole:
            aaa = self._data.columns[section]
            # print(aaa)
            return aaa

    def flags(self, index):
        return (
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )

    # # ----------------------------------------------------------
    # # AbstractItemModel - methods
    # # ----------------------------------------------------------

    # def beginInsertColumns(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginInsertColumns(parent, first, last)

    # def beginInsertRows(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginInsertRows(parent, first, last)

    # def beginMoveColumns(self, sourceParent, sourceFirst, sourceLast, destinationParent, destinationColumn):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginMoveColumns(sourceParent, sourceFirst, sourceLast, destinationParent, destinationColumn)

    # def beginMoveRows(self, sourceParent, sourceFirst, sourceLast, destinationParent, destinationRow):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginMoveRows(sourceParent, sourceFirst, sourceLast, destinationParent, destinationRow)

    # def beginRemoveColumns(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginRemoveColumns(parent, first, last)

    # def beginRemoveRows(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginRemoveRows(parent, first, last)

    # def beginResetModel(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).beginResetModel()

    # def changePersistentIndex(self, a_from, to):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).changePersistentIndex(a_from, to)

    # def changePersistentIndexList(self, a_from, to):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).changePersistentIndexList(a_from, to)

    # def checkIndex(self, index, options):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).checkIndex(index, options)

    # def createIndex(self, row, column, ptr):
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).createIndex(row, column, ptr)

    #     # log.info(f"row: {row}, col: {column}")
    #     # log.info(ptr)

    #     # log.debug("S - -----------------------")
    #     # print(type(aaa))
    #     # print(aaa)
    #     # log.debug("E - -----------------------")
    #     return aaa

    # def createIndex(self, row, column, id):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).createIndex(row, column, id)

    #     # log.debug("-----------------------")
    #     # print(type(aaa))
    #     # print(aaa)
    #     # log.debug("-----------------------")
    #     return aaa

    # def decodeData(self, row, column, parent, stream):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).decodeData(row, column, parent, stream)

    # def encodeData(self, indexes, stream):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).encodeData(indexes, stream)

    # def endInsertColumns(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endInsertColumns()

    # def endInsertRows(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endInsertRows()

    # def endMoveColumns(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endMoveColumns()

    # def endMoveRows(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endMoveRows()

    # def endRemoveColumns(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endRemoveColumns()

    # def endRemoveRows(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endRemoveRows()

    # def endResetModel(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).endResetModel()

    # def hasIndex(self, row, column, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).hasIndex(row, column, parent)
    #     # print("######")
    #     # print(aaa)
    #     # print("######")
    #     return aaa

    # def insertColumn(self, column, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).insertColumn(column, parent)

    # def insertRow(self, row, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).insertRow(row, parent)

    # def moveColumn(self, sourceParent, sourceColumn, destinationParent, destinationChild):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).moveColumn(sourceParent, sourceColumn, destinationParent, destinationChild)

    # def moveRow(self, sourceParent, sourceRow, destinationParent, destinationChild):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).moveRow(sourceParent, sourceRow, destinationParent, destinationChild)

    # def persistentIndexList(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).persistentIndexList()

    # def removeColumn(self, column, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).removeColumn(column, parent)

    # def removeRow(self, row, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).removeRow(row, parent)

    # # ----------------------------------------------------------
    # # AbstractItemModel - virtual methods
    # # ----------------------------------------------------------

    # def buddy(self, index) -> QModelIndex:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).buddy(index)
    #     # print(aaa)
    #     return aaa

    # def canDropMimeData(self, data, action, row, column, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).canDropMimeData(data, action, row, column, parent)
    #     # print(aaa)
    #     return aaa

    # def canFetchMore(self, parent) -> bool:
    #     # log.debug("Running...")
    #     # log.debug(locals())

    #     log.info(f"parent.row: {parent.row()}, parent.col: {parent.column()}")

    #     # result = self.hasChildren(parent)
    #     if parent.row() < self.firstLevel.shape[0]:
    #         return True
    #     else:
    #         return False

    # print(result)
    # return result

    # aaa = super(FuturesTreeModel, self).canFetchMore(parent)
    # print(aaa)
    # return aaa

    # # def columnCount(self, parent) -> int:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     print(parent.row())
    # #     print(parent.column())

    # #     aaa = super(FuturesTreeModel, self).columnCount(parent)
    # #     print(aaa)
    # #     return aaa

    # # def data(self, index, role) -> object:
    # #     # log.debug("Running...")
    # #     # log.debug(locals())

    # #     aaa = super(FuturesTreeModel, self).data(index, role)
    # #     # print(aaa)
    # #     return aaa

    # def dropMimeData(self, data, action, row, column, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).dropMimeData(data, action, row, column, parent)
    #     # print(aaa)
    #     return aaa

    # def fetchMore(self, parent):
    #     # log.debug("Running...")
    #     # log.debug(locals())

    #     log.info(f"parent.row: {parent.row()}, parent.col: {parent.column()}")

    #     # super(FuturesTreeModel, self).fetchMore(parent)

    # def flags(self, index) -> Qt.ItemFlags:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).flags(index)
    #     # print(aaa)
    #     return aaa

    def hasChildren(self, parent) -> bool:
        # log.debug("Running...")
        # log.debug(locals())
        # print(traceback.print_stack(file=sys.stdout))
        # print("--")
        # print("parent")
        # print(parent.isValid())
        # print(parent.row())
        # print(parent.column())
        # print(parent.model())
        # print(parent.internalPointer())
        # if parent.internalPointer() is not None:
        #     print(parent.internalPointer().data)

        # log.info(f"parent.row: {parent.row()}, parent.col: {parent.column()}")

        # if not parent.isValid():
        #     return True

        if parent.row() < self.firstLevel.shape[0]:
            return True
        else:
            return False

        itemId = self.firstLevel.iloc[
            parent.row(),
        ]

        childrenCount = self._data[self._data["parentId"] == itemId.name]

        result = True if childrenCount.shape[0] > 0 else False

        # aaa = super(FuturesTreeModel, self).hasChildren(parent)
        # print("--")
        # print(aaa)
        # print("--")
        return result

    # # def headerData(self, section, orientation, role=Qt.DisplayRole) -> object:
    # #     # log.debug("Running...")
    # #     # log.debug(locals())
    # #     aaa = super(FuturesTreeModel, self).headerData(section, orientation, role)
    # #     # print(aaa)
    # #     return aaa

    # # def index(self, row: int, col: int, _parent=QModelIndex()) -> QModelIndex:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     print(row)
    # #     print(col)
    # #     print(_parent.row())
    # #     print(_parent.column())

    # #     aaa = super(FuturesTreeModel, self).index(row, col, _parent)
    # #     print(aaa)
    # #     return aaa

    # def insertColumns(self, column, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).insertColumns(column, count, parent)
    #     print(aaa)
    #     return aaa

    # def insertRows(self, row, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).insertRows(row, count, parent)
    #     print(aaa)
    #     return aaa

    # def itemData(self, index):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).itemData(index)
    #     print(aaa)
    #     return aaa

    # def match(self, start, role, value, hits, flags):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).match(start, role, value, hits, flags)
    #     print(aaa)
    #     return aaa

    # def mimeData(self, indexes) -> QMimeData:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).mimeData(indexes)
    #     print(aaa)
    #     return aaa

    # def mimeTypes(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).mimeTypes()
    #     print(aaa)
    #     return aaa

    # def moveColumns(self, sourceParent, sourceColumn, count, destinationParent, destinationChild) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).moveColumns(sourceParent, sourceColumn, count, destinationParent, destinationChild)
    #     print(aaa)
    #     return aaa

    # def moveRows(self, sourceParent, sourceRow, count, destinationParent, destinationChild) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).moveRows(sourceParent, sourceRow, count, destinationParent, destinationChild)
    #     print(aaa)
    #     return aaa

    # # def parent(self, child) -> QModelIndex:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     print(traceback.print_stack(file=sys.stdout))
    # #     print(child.row())
    # #     print(child.column())

    # #     aaa = super(FuturesTreeModel, self).parent(child)
    # #     print(aaa)
    # #     return aaa

    # def removeColumns(self, column, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).removeColumns(column, count, parent)
    #     print(aaa)
    #     return aaa

    # def removeRows(self, row, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).removeRows(row, count, parent)
    #     print(aaa)
    #     return aaa

    # def revert(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).revert()

    # def roleNames(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).roleNames()
    #     print(aaa)
    #     return aaa

    # # def rowCount(self, parent) -> int:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     aaa = super(FuturesTreeModel, self).rowCount(parent)
    # #     print(aaa)
    # #     return aaa

    # def setData(self, index, value, role) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).setData(index, value, role)
    #     print(aaa)
    #     return aaa

    # # def setHeaderData(self, section, orientation, value, role) -> bool:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     aaa = super(FuturesTreeModel, self).setHeaderData(section, orientation, value, role)
    # #     print(aaa)
    # #     return aaa

    # def setItemData(self, index, roles) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).setItemData(index, roles)
    #     print(aaa)
    #     return aaa

    # def sibling(self, row, column, idx) -> QModelIndex:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).sibling(row, column, idx)
    #     # print(aaa)
    #     return aaa

    # def sort(self, column, order):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).sort(column, order)

    # def span(self, index) -> QSize:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).span(index)
    #     print(aaa)
    #     return aaa

    # def submit(self) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).submit()
    #     print(aaa)
    #     return aaa

    # def supportedDragActions(self) -> Qt.DropActions:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).supportedDragActions()
    #     print(aaa)
    #     return aaa

    # def supportedDropActions(self) -> Qt.DropActions:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).supportedDropActions()
    #     print(aaa)
    #     return aaa
