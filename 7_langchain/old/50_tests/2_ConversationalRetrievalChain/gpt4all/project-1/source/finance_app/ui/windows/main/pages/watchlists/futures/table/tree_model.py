from __future__ import annotations
from business.model.contract_details import (
    IBContractDetails,
)  # allow return same type as class ..... -> FuturesTreeNode

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


def defaultValue(data: Union[List[Any], None]) -> pd.DataFrame:
    df = pd.DataFrame(
        data=data,
        columns=[
            "symbol",
            "localSymbol",
            "contractMonth",
            "contractEndDate",
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
            "delete",
        ],
    )
    df.set_index(["symbol", "localSymbol"], inplace=True)
    return df


class FuturesTreeNode(object):
    def __init__(self, data: Union[List[Any], None]):
        self.data: pd.DataFrame = defaultValue(data)
        log.info("------------------")
        log.info(data)
        log.info(self.data)
        log.info("------------------")

        self._children: List[FuturesTreeNode] = []
        self._parent: Union[FuturesTreeNode, Any] = None

    # def data(self, column_index: int) -> Any:
    #     if column_index >= 0 and column_index < self.columnCount():
    #         return self.data.iloc[[], column_index]
    #     else:
    #         return None

    def columnCount(self) -> int:
        return int(self.data.shape[1]) + len(self.data.index.names)

    def childCount(self) -> int:
        return len(self._children)

    def child(self, row_index: int) -> Union[FuturesTreeNode, None]:
        if row_index >= 0 and row_index < self.childCount():
            return self._children[row_index]
        else:
            return None

    def parent(self):
        return self._parent

    def row(self):
        return 0

    def updateData(
        self, index1: str, index2: str, columnName: str, value: Any
    ) -> bool:
        # log.debug("Running...")
        # log.debug(locals())
        isExist = (index1, index2) in self.data.index

        if isExist:
            self.data.loc[(index1, index2), columnName] = value

            if columnName == "close" or columnName == "last":
                # change column - calculation
                closeP: float = self.data.loc[(index1, index2), "close"]
                lastP: float = self.data.loc[(index1, index2), "last"]
                if closeP > 0 and lastP > 0:
                    self.data.loc[(index1, index2), "change"] = round(
                        ((lastP - closeP) / closeP) * 100, 1
                    )
                else:
                    self.data.loc[(index1, index2), "change"] = 0

            return True
        else:
            # print("SEARCH IN CHILDREN")
            result = False

            for child in self._children:
                childResult = child.updateData(
                    index1, index2, columnName, value
                )
                if childResult:
                    result = childResult
                    continue

            return result

    def addChild(self, child: FuturesTreeNode):
        child._parent = self

        log.info("-addChild-1----------------")
        log.info(self.data)
        log.info("------------------")

        self._children.append(child)

        log.info("-addChild-2----------------")
        log.info(self.data)
        log.info("------------------")

    def removeChild(self, ticker: str):
        resIndex = -1
        curIndex = 0
        for child in self._children:

            if child.data.index[0][0] == ticker:
                resIndex = curIndex
            curIndex += 1

        if resIndex != -1:
            self._children.pop(resIndex)

    def moveChild(self, fromIndex: int, toIndex: int):
        if fromIndex < 0 or toIndex < 0:
            raise Exception("THIS SHOULD NOT HAPPENED")

        fromItem = self._children.pop(fromIndex)
        self._children.insert(toIndex, fromItem)

        log.info("CHILDRENCHILDRENCHILDRENCHILDRENCHILDREN ---")
        for a in self._children:
            log.info(a.data)
        log.info("CHILDRENCHILDRENCHILDRENCHILDRENCHILDREN ---")

        # data
        log.info("DATADATADATADATADATADATADATA ---")
        log.info(self.data)

        # self.data: pd.DataFrame = defaultValue(self._children)
        # log.info(self.data)
        log.info("DATADATADATADATADATADATADATA ---")


class FuturesTreeModel(QAbstractItemModel):
    def __init__(
        self,
        data: Union[List[FuturesTreeNode], None],
        headerData: List[str],
        parent=None,
    ):
        super(FuturesTreeModel, self).__init__(parent)
        self._data = defaultValue(data)

        # data
        self.root = FuturesTreeNode(None)
        if data is not None:
            for item in data:
                self.root.addChild(item)

        # header
        self.header_data = defaultdict()
        for index, value in enumerate(headerData):
            self.setHeaderData(index, Qt.Horizontal, value)

    # ----------------------------------------------------------
    # Custom methods
    # ----------------------------------------------------------

    def __indexColumnsCount(self):
        return len(self.root.data.index.names)

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
        helpIndex = 0
        rootItem = FuturesTreeNode([])
        for row in data:
            vals: List[Any] = [
                row.contract.symbol,
                row.contract.localSymbol,
                row.contractMonth,
                row.contract.lastTradeDateOrContractMonth,
                *np.zeros(16),
            ]
            item = FuturesTreeNode([vals])

            if helpIndex == 0:
                rootItem = item
            else:
                rootItem.addChild(item)

            helpIndex += 1

        self.beginInsertRows(QModelIndex(), 0, 0)
        self.root.addChild(rootItem)
        self.endInsertRows()

    def reset(self):
        self.beginResetModel()
        self.root = FuturesTreeNode(None)
        self.endResetModel()

    def removeFuture(self, ticker: str):
        self.beginRemoveRows(QModelIndex(), 0, 0)
        self.root.removeChild(ticker)
        self.endRemoveRows()

    # ----------------------------------------------------------
    # Signals / Slots
    # ----------------------------------------------------------
    @pyqtSlot(dict, name="on_update_model")
    def on_update_model(self, obj: Dict[str, Any]):
        # log.debug("Running...")
        # log.debug(locals())
        log.debug(
            f"ticker={obj['ticker']}|localSymbol={obj['localSymbol']}|type={obj['type']}|price={obj['price']}"
        )

        if obj == {}:
            log.info("EMPTY")
            return

        self.root.updateData(
            obj["ticker"], obj["localSymbol"], obj["type"], obj["price"]
        )

        bbb = QModelIndex()
        bbb.row = 1
        bbb.column = 1

        # self.dataChanged.emit(QModelIndex(), QModelIndex())
        self.dataChanged.emit(bbb, bbb)

    # ----------------------------------------------------------
    # Minimum methods to override
    # ----------------------------------------------------------
    def data(self, index: QModelIndex, role: Any) -> Union[str, None]:
        # log.debug("Running...")
        # log.debug(locals())

        if not index.isValid():
            log.error("index invalid - return None")
            return None

        if index.internalPointer().data.empty:
            # log.error("index data are empty")
            return None

        if role == Qt.DisplayRole:
            columnIndex: int = index.column()
            if columnIndex <= 1:
                return str(
                    index.internalPointer().data.index.values[0][columnIndex]
                )
            else:
                value: str = index.internalPointer().data.iloc[
                    0, columnIndex - self.__indexColumnsCount()
                ]

                if columnIndex == 3:  # contract-end-date
                    return value
                elif columnIndex == 4:  # diff
                    return f"{value:.0f}"
                elif columnIndex == 5:  # bid-size
                    return f"{value:.0f}"
                elif columnIndex == 6:  # bid
                    return f"{value:.2f}"
                if columnIndex == 7:  # last
                    return f"{value:.2f}"
                elif columnIndex == 8:  # ask
                    return f"{value:.2f}"
                elif columnIndex == 9:  # ask-size
                    return f"{value:.0f}"
                elif columnIndex == 10:  # open
                    return f"{value:.2f}"
                elif columnIndex == 11:  # high
                    return f"{value:.2f}"
                elif columnIndex == 12:  # low
                    return f"{value:.2f}"
                elif columnIndex == 13:  # close
                    return f"{value:.2f}"
                # elif columnIndex == 14:  # change
                #     return f"{value:.1f}"
                elif columnIndex == 15:  # volume
                    return f"{value:.0f}"
                elif columnIndex == 16:  # avg volume
                    return f"{value:.0f}"
                elif columnIndex == 17:  # hist volatility
                    return f"{value:.2f}"
                elif columnIndex == 18:  # implied volatility
                    return f"{value:.2f}"

                return str(value)

        return None

    def rowCount(self, parent: QModelIndex) -> int:
        # log.debug("Running...")
        # log.debug(locals())

        if parent is None:
            return 0
        elif parent.isValid():
            return int(parent.internalPointer().childCount())
        else:
            return self.root.childCount()

    def columnCount(self, parent: QModelIndex) -> int:
        # log.debug("Running...")
        # log.debug(locals())

        if parent.isValid():
            return int(parent.internalPointer().columnCount())
        else:
            return self.root.columnCount()

    def index(self, row: int, col: int, _parent=QModelIndex()) -> QModelIndex:
        # log.debug("Running...")
        # log.debug(locals())

        if not self.hasIndex(self, row, col, _parent):
            return QModelIndex()

        if not _parent or not _parent.isValid():
            parent = self.root
        else:
            parent = _parent.internalPointer()

        child = parent.child(row)
        if child:
            return self.createIndex(self, row, col, child)
        else:
            return QModelIndex()

    def parent(self, child: QModelIndex) -> QModelIndex:
        # log.debug("Running...")
        # log.debug(locals())

        if child.isValid():
            p = child.internalPointer().parent()
            if p:
                return self.createIndex(self, p.row(), 0, p)

        return QModelIndex()

    # ----------------------------------------------------------
    # Override - headers
    # ----------------------------------------------------------

    def setHeaderData(
        self,
        section: int,
        orientation: Qt.Orientation,
        value: QVariant,
        role=Qt.DisplayRole,
    ) -> bool:
        # log.debug("Running...")
        # log.debug(locals())

        if orientation == Qt.Horizontal:

            # print("Setting value: " + value)
            self.header_data[section] = value

            return True

        else:

            aaa = super(FuturesTreeModel, self).setHeaderData(
                section, orientation, value, role
            )
            # print(aaa)
            return aaa

    def headerData(
        self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole
    ) -> object:
        # log.debug("Running...")
        # log.debug(locals())

        if role == Qt.DisplayRole:
            aaa = self.header_data[section]
            # print(aaa)
            return aaa
        else:
            aaa = super(FuturesTreeModel, self).headerData(
                section, orientation, role
            )
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
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).createIndex(row, column, ptr)

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
    #     aaa = super(FuturesTreeModel, self).canFetchMore(parent)
    #     # print(aaa)
    #     return aaa

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
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(FuturesTreeModel, self).fetchMore(parent)

    # def flags(self, index) -> Qt.ItemFlags:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(FuturesTreeModel, self).flags(index)
    #     # print(aaa)
    #     return aaa

    # def hasChildren(self, parent) -> bool:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     # print(traceback.print_stack(file=sys.stdout))
    #     # print("--")
    #     # print("parent")
    #     # print(parent.isValid())
    #     # print(parent.row())
    #     # print(parent.column())
    #     # print(parent.model())
    #     # print(parent.internalPointer())
    #     # if parent.internalPointer() is not None:
    #     #     print(parent.internalPointer().data)
    #     aaa = super(FuturesTreeModel, self).hasChildren(parent)
    #     # print("--")
    #     # print(aaa)
    #     # print("--")
    #     return aaa

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
