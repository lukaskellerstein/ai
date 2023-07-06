import logging
import sys
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, pyqtSlot

# create logger
log = logging.getLogger("CellarLogger")


def defaultValue(data):
    df = pd.DataFrame(
        data=data,
        columns=[
            "expiration",
            "strike",
            "bid",
            "ask",
            "last",
            "optPrice",
            "undPrice",
            "pvDividend",
            "delta",
            "gamma",
            "vega",
            "theta",
        ],
    )
    df.set_index(["expiration", "strike"], inplace=True)
    return df


class OptionsTreeNode(object):
    def __init__(self, data):
        self._data = defaultValue(data)
        self._columncount = self._data.shape[1]
        self._children = []
        self._parent = None
        self._row = 0

    def data(self, column_index: int):
        if column_index >= 0 and column_index < self.columnCount():
            return self._data.iloc[[], column_index]

    def columnCount(self):
        return self._data.shape[1] + len(self._data.index.names)

    def childCount(self):
        return len(self._children)

    def child(self, row_index: int):
        if row_index >= 0 and row_index < self.childCount():
            return self._children[row_index]

    def parent(self):
        return self._parent

    def row(self):
        return self._row

    def updateData(self, index1, index2, columnName, value) -> bool:
        # log.debug("Running...")
        # log.debug(locals())
        isExist = (index1, index2) in self._data.index

        if isExist:
            self._data.loc[(index1, index2), columnName] = value
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

    def addChild(self, child):
        child._parent = self
        child._row = len(self._children)
        self._children.append(child)
        self._columncount = max(child.columnCount(), self._columncount)

    def removeChild(self, ticker):
        resIndex = -1
        curIndex = 0
        for child in self._children:

            if child._data.index[0][0] == ticker:
                resIndex = curIndex
            curIndex += 1

        if resIndex != -1:
            self._children.pop(resIndex)


class OptionsTreeModel(QAbstractItemModel):
    def __init__(self, data, headerData, parent=None):
        super(OptionsTreeModel, self).__init__(parent)

        # data
        self._root = OptionsTreeNode(None)
        if data is not None:
            for item in data:
                self._root.addChild(item)

        # header
        self.header_data = defaultdict()
        for index, value in enumerate(headerData):
            self.setHeaderData(index, Qt.Horizontal, value)

    # ----------------------------------------------------------
    # Custom methods
    # ----------------------------------------------------------

    def __indexColumnsCount(self):
        return len(self._root._data.index.names)

    def __dataColumnsCount(self):
        return self._root._data.shape[1]

    def __allColumnsCount(self):
        return self._root._data.shape[1] + self.__indexColumnsCount()

    def setStructure(self, expirations: list, strikes: list):
        # log.debug("Running...")
        # log.debug(locals())

        # sort expirations and strikes
        expirations.sort()
        strikes.sort()

        for expiration in expirations:

            vals = [expiration, *np.zeros(11)]
            item = OptionsTreeNode([vals])

            for strike in strikes:
                vals = [expiration, strike, *np.zeros(10)]
                itemInner = OptionsTreeNode([vals])
                item.addChild(itemInner)

            self.beginInsertRows(QModelIndex(), 0, 0)
            self._root.addChild(item)
            self.endInsertRows()

    def reset(self):
        # log.debug("Running...")
        # log.debug(locals())

        self.beginResetModel()
        self._root = OptionsTreeNode(None)
        self.endResetModel()

    def removeFuture(self, ticker):
        # log.debug("Running...")
        # log.debug(locals())

        self.beginRemoveRows(QModelIndex(), 0, 0)
        self._root.removeChild(ticker)
        self.endRemoveRows()

    # ----------------------------------------------------------
    # Signals / Slots
    # ----------------------------------------------------------
    @pyqtSlot(dict, name="on_update_model")
    def on_update_model(self, obj):
        # log.debug("Running...")
        # log.debug(locals())
        # log.debug(f"ticker={obj['ticker']}|localSymbol={obj['localSymbol']}|type={obj['type']}|price={obj['price']}")

        # self._root.updateData(obj["ticker"], obj["localSymbol"], obj["type"], obj["price"])

        # bbb = QModelIndex()
        # bbb.row = 1
        # bbb.column = 1

        # # self.dataChanged.emit(QModelIndex(), QModelIndex())
        # self.dataChanged.emit(bbb, bbb)
        pass

    # ----------------------------------------------------------
    # Minimum methods to override
    # ----------------------------------------------------------
    def data(self, index, role) -> object:
        # log.debug("Running...")
        # log.debug(locals())

        if not index.isValid():
            print("index invalid - return None")
            return None

        if role == Qt.DisplayRole:
            columnIndex = index.column()
            if columnIndex <= 1:
                return index.internalPointer()._data.index.values[0][
                    columnIndex
                ]
            else:
                value = index.internalPointer()._data.iloc[
                    0, columnIndex - self.__indexColumnsCount()
                ]

                # if columnIndex == 3:  # contract-end-date
                #     return value
                # elif columnIndex == 4:  # diff
                #     return f"{value:.0f}"
                # elif columnIndex == 5:  # bid-size
                #     return f"{value:.0f}"
                # elif columnIndex == 6:  # bid
                #     return f"{value:.2f}"
                # if columnIndex == 7: # last
                #     return f"{value:.2f}"
                # elif columnIndex == 8:  # ask
                #     return f"{value:.2f}"
                # elif columnIndex == 9:  # ask-size
                #     return f"{value:.0f}"
                # elif columnIndex == 10:  # open
                #     return f"{value:.2f}"
                # elif columnIndex == 11:  # high
                #     return f"{value:.2f}"
                # elif columnIndex == 12:  # low
                #     return f"{value:.2f}"
                # elif columnIndex == 13:  # close
                #     return f"{value:.2f}"
                # # elif columnIndex == 14:  # change
                # #     return f"{value:.1f}"
                # elif columnIndex == 15:  # volume
                #     return f"{value:.0f}"
                # elif columnIndex == 16:  # avg volume
                #     return f"{value:.0f}"
                # elif columnIndex == 17:  # hist volatility
                #     return f"{value:.2f}"
                # elif columnIndex == 18:  # implied volatility
                #     return f"{value:.2f}"

                return str(value)

        return None

    def rowCount(self, parent) -> int:
        # log.debug("Running...")
        # log.debug(locals())

        if parent is None:
            return 0
        elif parent.isValid():
            return parent.internalPointer().childCount()
        else:
            return self._root.childCount()

    def columnCount(self, parent) -> int:
        # log.debug("Running...")
        # log.debug(locals())

        if parent.isValid():
            return parent.internalPointer().columnCount()
        else:
            return self._root.columnCount()

    def index(self, row: int, col: int, _parent=QModelIndex()) -> QModelIndex:
        # log.debug("Running...")
        # log.debug(locals())

        if not _parent or not _parent.isValid():
            parent = self._root
        else:
            parent = _parent.internalPointer()

        if not QAbstractItemModel.hasIndex(self, row, col, _parent):
            return QModelIndex()

        child = parent.child(row)
        if child:
            return QAbstractItemModel.createIndex(self, row, col, child)
        else:
            return QModelIndex()

    def parent(self, child) -> QModelIndex:
        # log.debug("Running...")
        # log.debug(locals())

        if child.isValid():
            p = child.internalPointer().parent()
            if p:
                return QAbstractItemModel.createIndex(self, p.row(), 0, p)

        return QModelIndex()

    # ----------------------------------------------------------
    # Override - headers
    # ----------------------------------------------------------

    def setHeaderData(
        self, section, orientation, value, role=Qt.DisplayRole
    ) -> bool:
        # log.debug("Running...")
        # log.debug(locals())

        if orientation == Qt.Horizontal:

            # print("Setting value: " + value)
            self.header_data[section] = value

            return True

        else:

            aaa = super(OptionsTreeModel, self).setHeaderData(
                section, orientation, value, role
            )
            # print(aaa)
            return aaa

    def headerData(self, section, orientation, role=Qt.DisplayRole) -> object:
        # log.debug("Running...")
        # log.debug(locals())

        if role == Qt.DisplayRole:
            aaa = self.header_data[section]
            # print(aaa)
            return aaa
        else:
            aaa = super(OptionsTreeModel, self).headerData(
                section, orientation, role
            )
            # print(aaa)
            return aaa

    # # ----------------------------------------------------------
    # # AbstractItemModel - methods
    # # ----------------------------------------------------------

    # def beginInsertColumns(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginInsertColumns(parent, first, last)

    # def beginInsertRows(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginInsertRows(parent, first, last)

    # def beginMoveColumns(self, sourceParent, sourceFirst, sourceLast, destinationParent, destinationColumn):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginMoveColumns(sourceParent, sourceFirst, sourceLast, destinationParent, destinationColumn)

    # def beginMoveRows(self, sourceParent, sourceFirst, sourceLast, destinationParent, destinationRow):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginMoveRows(sourceParent, sourceFirst, sourceLast, destinationParent, destinationRow)

    # def beginRemoveColumns(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginRemoveColumns(parent, first, last)

    # def beginRemoveRows(self, parent, first, last):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginRemoveRows(parent, first, last)

    # def beginResetModel(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).beginResetModel()

    # def changePersistentIndex(self, a_from, to):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).changePersistentIndex(a_from, to)

    # def changePersistentIndexList(self, a_from, to):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).changePersistentIndexList(a_from, to)

    # def checkIndex(self, index, options):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).checkIndex(index, options)

    # def createIndex(self, row, column, ptr):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).createIndex(row, column, ptr)

    #     # log.debug("S - -----------------------")
    #     # print(type(aaa))
    #     # print(aaa)
    #     # log.debug("E - -----------------------")
    #     return aaa

    # def createIndex(self, row, column, id):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).createIndex(row, column, id)

    #     # log.debug("-----------------------")
    #     # print(type(aaa))
    #     # print(aaa)
    #     # log.debug("-----------------------")
    #     return aaa

    # def decodeData(self, row, column, parent, stream):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).decodeData(row, column, parent, stream)

    # def encodeData(self, indexes, stream):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).encodeData(indexes, stream)

    # def endInsertColumns(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endInsertColumns()

    # def endInsertRows(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endInsertRows()

    # def endMoveColumns(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endMoveColumns()

    # def endMoveRows(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endMoveRows()

    # def endRemoveColumns(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endRemoveColumns()

    # def endRemoveRows(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endRemoveRows()

    # def endResetModel(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).endResetModel()

    # def hasIndex(self, row, column, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).hasIndex(row, column, parent)
    #     # print("######")
    #     # print(aaa)
    #     # print("######")
    #     return aaa

    # def insertColumn(self, column, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).insertColumn(column, parent)

    # def insertRow(self, row, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).insertRow(row, parent)

    # def moveColumn(self, sourceParent, sourceColumn, destinationParent, destinationChild):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).moveColumn(sourceParent, sourceColumn, destinationParent, destinationChild)

    # def moveRow(self, sourceParent, sourceRow, destinationParent, destinationChild):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).moveRow(sourceParent, sourceRow, destinationParent, destinationChild)

    # def persistentIndexList(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).persistentIndexList()

    # def removeColumn(self, column, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).removeColumn(column, parent)

    # def removeRow(self, row, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).removeRow(row, parent)

    # # ----------------------------------------------------------
    # # AbstractItemModel - virtual methods
    # # ----------------------------------------------------------

    # def buddy(self, index) -> QModelIndex:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).buddy(index)
    #     # print(aaa)
    #     return aaa

    # def canDropMimeData(self, data, action, row, column, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).canDropMimeData(data, action, row, column, parent)
    #     # print(aaa)
    #     return aaa

    # def canFetchMore(self, parent) -> bool:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).canFetchMore(parent)
    #     # print(aaa)
    #     return aaa

    # # def columnCount(self, parent) -> int:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     print(parent.row())
    # #     print(parent.column())

    # #     aaa = super(OptionsTreeModel, self).columnCount(parent)
    # #     print(aaa)
    # #     return aaa

    # # def data(self, index, role) -> object:
    # #     # log.debug("Running...")
    # #     # log.debug(locals())

    # #     aaa = super(OptionsTreeModel, self).data(index, role)
    # #     # print(aaa)
    # #     return aaa

    # def dropMimeData(self, data, action, row, column, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).dropMimeData(data, action, row, column, parent)
    #     # print(aaa)
    #     return aaa

    # def fetchMore(self, parent):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).fetchMore(parent)

    # def flags(self, index) -> Qt.ItemFlags:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).flags(index)
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
    #     #     print(parent.internalPointer()._data)
    #     aaa = super(OptionsTreeModel, self).hasChildren(parent)
    #     # print("--")
    #     # print(aaa)
    #     # print("--")
    #     return aaa

    # # def headerData(self, section, orientation, role=Qt.DisplayRole) -> object:
    # #     # log.debug("Running...")
    # #     # log.debug(locals())
    # #     aaa = super(OptionsTreeModel, self).headerData(section, orientation, role)
    # #     # print(aaa)
    # #     return aaa

    # # def index(self, row: int, col: int, _parent=QModelIndex()) -> QModelIndex:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     print(row)
    # #     print(col)
    # #     print(_parent.row())
    # #     print(_parent.column())

    # #     aaa = super(OptionsTreeModel, self).index(row, col, _parent)
    # #     print(aaa)
    # #     return aaa

    # def insertColumns(self, column, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).insertColumns(column, count, parent)
    #     print(aaa)
    #     return aaa

    # def insertRows(self, row, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).insertRows(row, count, parent)
    #     print(aaa)
    #     return aaa

    # def itemData(self, index):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).itemData(index)
    #     print(aaa)
    #     return aaa

    # def match(self, start, role, value, hits, flags):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).match(start, role, value, hits, flags)
    #     print(aaa)
    #     return aaa

    # def mimeData(self, indexes) -> QMimeData:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).mimeData(indexes)
    #     print(aaa)
    #     return aaa

    # def mimeTypes(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).mimeTypes()
    #     print(aaa)
    #     return aaa

    # def moveColumns(self, sourceParent, sourceColumn, count, destinationParent, destinationChild) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).moveColumns(sourceParent, sourceColumn, count, destinationParent, destinationChild)
    #     print(aaa)
    #     return aaa

    # def moveRows(self, sourceParent, sourceRow, count, destinationParent, destinationChild) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).moveRows(sourceParent, sourceRow, count, destinationParent, destinationChild)
    #     print(aaa)
    #     return aaa

    # # def parent(self, child) -> QModelIndex:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     print(traceback.print_stack(file=sys.stdout))
    # #     print(child.row())
    # #     print(child.column())

    # #     aaa = super(OptionsTreeModel, self).parent(child)
    # #     print(aaa)
    # #     return aaa

    # def removeColumns(self, column, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).removeColumns(column, count, parent)
    #     print(aaa)
    #     return aaa

    # def removeRows(self, row, count, parent) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).removeRows(row, count, parent)
    #     print(aaa)
    #     return aaa

    # def revert(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).revert()

    # def roleNames(self):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).roleNames()
    #     print(aaa)
    #     return aaa

    # # def rowCount(self, parent) -> int:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     aaa = super(OptionsTreeModel, self).rowCount(parent)
    # #     print(aaa)
    # #     return aaa

    # def setData(self, index, value, role) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).setData(index, value, role)
    #     print(aaa)
    #     return aaa

    # # def setHeaderData(self, section, orientation, value, role) -> bool:
    # #     log.debug("Running...")
    # #     log.debug(locals())
    # #     aaa = super(OptionsTreeModel, self).setHeaderData(section, orientation, value, role)
    # #     print(aaa)
    # #     return aaa

    # def setItemData(self, index, roles) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).setItemData(index, roles)
    #     print(aaa)
    #     return aaa

    # def sibling(self, row, column, idx) -> QModelIndex:
    #     # log.debug("Running...")
    #     # log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).sibling(row, column, idx)
    #     # print(aaa)
    #     return aaa

    # def sort(self, column, order):
    #     log.debug("Running...")
    #     log.debug(locals())
    #     super(OptionsTreeModel, self).sort(column, order)

    # def span(self, index) -> QSize:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).span(index)
    #     print(aaa)
    #     return aaa

    # def submit(self) -> bool:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).submit()
    #     print(aaa)
    #     return aaa

    # def supportedDragActions(self) -> Qt.DropActions:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).supportedDragActions()
    #     print(aaa)
    #     return aaa

    # def supportedDropActions(self) -> Qt.DropActions:
    #     log.debug("Running...")
    #     log.debug(locals())
    #     aaa = super(OptionsTreeModel, self).supportedDropActions()
    #     print(aaa)
    #     return aaa
