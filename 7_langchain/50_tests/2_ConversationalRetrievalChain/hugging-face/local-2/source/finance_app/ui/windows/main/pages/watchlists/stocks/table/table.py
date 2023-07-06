import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHeaderView, QTableView

from ui.windows.main.pages.watchlists.stocks.table.table_model import (
    StockTableModel,
)

# create logger
log = logging.getLogger("CellarLogger")


class StockTable(QTableView):

    on_open = pyqtSignal(object, name="on_open")
    on_remove = pyqtSignal(object, name="on_remove")
    on_order_changed = pyqtSignal(object, name="on_order_changed")

    draggedItem = None

    def __init__(self):
        super(QTableView, self).__init__()
        self.tableModel = StockTableModel([])
        self.setModel(self.tableModel)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        # header.setSectionResizeMode(5, QHeaderView.Stretch)

        # verHeader = self.verticalHeader()
        # verHeader.setSectionsMovable(True)
        # verHeader.setDragEnabled(True)
        # verHeader.setDragDropMode(self.InternalMove)

        self.setSortingEnabled(True)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setSelectionBehavior(QTableView.SelectRows)

        # self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.clicked.connect(self.myclick)
        self.doubleClicked.connect(self.mydoubleclick)

        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTableView.DragDrop)
        # self.setDragDropMode(self.InternalMove)
        self.setDragDropOverwriteMode(False)

    def myclick(self, index):
        # delete column
        if index.column() == self.tableModel.columnCount(0) - 1:
            row = self.tableModel._data.iloc[index.row()]
            self.on_remove.emit(row)

    def mydoubleclick(self, index):
        row = self.tableModel._data.iloc[index.row()]
        self.on_open.emit(row)

    # def dragEnterEvent(self, event):

    #     from_index = self.indexAt(event.pos()).row()

    #     # log.info(from_index)
    #     # log.info(from_index)

    #     # from_index = self.rowAt(event.pos().y())
    #     # log.info(event.pos().y())

    #     # log.info(from_index)

    #     self.draggedItem = self.tableModel._data.iloc[from_index]
    #     log.info(
    #         f"from index: {from_index}; from name: {self.draggedItem.name}"
    #     )
    #     event.accept()

    def dropEvent(self, event):
        # dropPosition = self.dropIndicatorPosition()
        # to_index = self.rowAt(event.pos().y())

        # if dropPosition == QTableView.AboveItem:
        #     print("INSERT ABOVE")
        #     to_index -= 1
        # elif dropPosition == QTableView.BelowItem:
        #     print("INSERT BELOW")
        #     pass
        # elif dropPosition == QTableView.OnItem:
        #     print("INSERT")
        #     pass

        # print(f"to index: {to_index}")

        selection = self.selectedIndexes()
        from_index = selection[0].row() if selection else -1
        fromItem = self.tableModel._data.iloc[from_index]
        log.info(f"from index: {from_index}; from name: {fromItem.name}")

        to_index = self.indexAt(event.pos()).row()
        toItem = self.tableModel._data.iloc[to_index]

        log.info(f"to index: {to_index}; to name: {toItem.name}")

        self.tableModel.insertStock(to_index, fromItem)
        self.on_order_changed.emit(self.tableModel.getStocks().tolist())
