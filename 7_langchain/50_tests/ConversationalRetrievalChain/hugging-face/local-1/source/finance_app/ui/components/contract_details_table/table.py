from business.model.asset import AssetType
import logging

from PyQt5.QtWidgets import QHeaderView, QTableView

from ui.components.contract_details_table.table_model_factory import (
    ContractDetailsTableModelFactory,
)

# create logger
log = logging.getLogger("CellarLogger")


class AssetContractDetailsTable(QTableView):
    def __init__(self, assetType: AssetType):
        super(QTableView, self).__init__()

        self.assetType = assetType
        self.tableModel = ContractDetailsTableModelFactory.create(
            self.assetType.value, []
        )
        self.setModel(self.tableModel)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        self.setSortingEnabled(True)
