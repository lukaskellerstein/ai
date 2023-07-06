from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QTreeView, QHeaderView

from ui.windows.main.pages.watchlists.options.table.tree_model import (
    OptionsTreeModel,
)


class OptionsTree(QTreeView):

    on_remove = pyqtSignal(object, name="on_remove")

    def __init__(self):
        super(QTreeView, self).__init__()

        # load styles
        with open(
            "ui/windows/main/pages/watchlists/options/table/tree.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        self.tree_header_data = [
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
        ]
        self.tree_model = OptionsTreeModel([], self.tree_header_data)
        self.setModel(self.tree_model)

        header = self.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        # header.setSectionResizeMode(5, QHeaderView.Stretch)

        self.clicked.connect(self.myclick)

    def myclick(self, index):
        if index.column() == 19:
            row = self.tree_model._root.child(index.row())
            self.on_remove.emit(row)
