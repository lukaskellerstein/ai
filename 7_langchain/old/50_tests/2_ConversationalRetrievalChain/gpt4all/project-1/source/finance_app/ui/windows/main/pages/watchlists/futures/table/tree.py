import logging
from PyQt5.QtCore import pyqtSignal, QModelIndex
from PyQt5.QtWidgets import QTreeView, QHeaderView

from ui.windows.main.pages.watchlists.futures.table.item_delegate import (
    MyRenderDelegate,
)
from ui.windows.main.pages.watchlists.futures.table.tree_model_new import (
    FuturesTreeModel,
)

# create logger
log = logging.getLogger("CellarLogger")


class FuturesTree(QTreeView):

    on_open = pyqtSignal(object, name="on_open")
    on_remove = pyqtSignal(object, name="on_remove")
    on_order_changed = pyqtSignal(object, name="on_order_changed")

    def __init__(self):
        super(QTreeView, self).__init__()

        # load styles
        with open(
            "ui/windows/main/pages/watchlists/futures/table/tree.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # self.tree_header_data = [
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
        # self.tree_model = FuturesTreeModel([], self.tree_header_data)
        self.tree_model = FuturesTreeModel()
        self.setModel(self.tree_model)

        header = self.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        # header.setSectionResizeMode(5, QHeaderView.Stretch)

        # self.setItemDelegate(MyRenderDelegate(self))
        self.collapseAll()

        self.clicked.connect(self.myclick)
        self.doubleClicked.connect(self.mydoubleclick)

        # self.setDragEnabled(True)
        # self.setDropIndicatorShown(True)
        # self.setAcceptDrops(True)
        # self.setDragDropMode(QTreeView.DragDrop)
        # # self.setDragDropMode(self.InternalMove)
        # self.setDragDropOverwriteMode(False)

    def myclick(self, index: QModelIndex):
        pass
        # if index.column() == 19:
        #     row = self.tree_model.root.child(index.row())
        #     self.on_remove.emit(row)

    def mydoubleclick(self, index):
        pass
        # row = self.tree_model.root.child(index.row())
        # self.on_open.emit(row)

    # def dragEnterEvent(self, event):
    #     # to_index = self.rowAt(event.pos().y())

    #     i = event

    #     log.info(type(event))
    #     log.info(event)
    #     log.info(type(event.pos()))
    #     log.info(event.pos())
    #     log.info(type(event.pos().y()))
    #     log.info(event.pos().y())

    #     to_index = self.indexAt(event.pos())

    #     log.info(type(to_index))
    #     log.info(to_index)

    #     log.info(type(to_index.row()))
    #     log.info(to_index.row())

    #     log.info(self.tree_model)

    #     self.draggedItem = self.tree_model._data.iloc[to_index]
    #     print(self.draggedItem.name)
    #     event.accept()

    # def dropEvent(self, event):
    #     dropPosition = self.dropIndicatorPosition()
    #     to_index = self.rowAt(event.pos().y())

    #     if dropPosition == QTreeView.AboveItem:
    #         print("INSERT ABOVE")
    #         to_index -= 1
    #     elif dropPosition == QTreeView.BelowItem:
    #         print("INSERT BELOW")
    #         pass
    #     elif dropPosition == QTreeView.OnItem:
    #         print("INSERT")
    #         pass

    #     self.tableModel.insertStock(to_index, self.draggedItem)
    #     self.on_order_changed.emit(self.tableModel.getStocks().tolist())

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
        log.info(selection)
        log.info(selection[0])
        log.info(selection[0].row())
        from_index = selection[0].row() if selection else -1
        log.info(from_index)
        log.info(self.tree_model._data)

        fromItem1 = self.tree_model.root.child(from_index)
        # fromItem = self.tree_model._data.iloc[from_index]
        log.info(fromItem1.data)
        log.info(fromItem1.data.index[0])
        log.info(fromItem1.data.index[0][0])
        log.info(
            f"from index: {from_index}; from name: {fromItem1.data.index[0][0]}"
        )

        to_index = self.indexAt(event.pos()).row()
        toItem1 = self.tree_model.root.child(to_index)
        # toItem = self.tree_model._data.iloc[to_index]

        log.info(f"to index: {to_index}; to name: {toItem1.data.index[0][0]}")

        self.tree_model.root.moveChild(from_index, to_index)
        # self.tree_model.insertStock(to_index, fromItem)
        # self.on_order_changed.emit(self.tree_model.getStocks().tolist())
