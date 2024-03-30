from business.modules.asset_bl import AssetBL
import logging
from typing import Any

from PyQt5 import uic

from typing import List
from business.model.asset import Asset, AssetType
from ui.base.base_page import BasePage
from business.model.contract_details import IBContractDetails
from ui.windows.asset_detail.shared.pages.contract_details.table import CDTable
from ui.components.contract_details_table.table import (
    AssetContractDetailsTable,
)
from rx import operators as ops
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from business.model.factory.contract_factory import ContractFactory, SecType

# create logger
log = logging.getLogger("CellarLogger")


class ContractDetailsPage(BasePage):
    asset: Asset

    on_fillTable = pyqtSignal(list)

    def __init__(self, **kwargs: Any):
        super().__init__()
        log.info("Running ...")

        # load template
        uic.loadUi(
            "ui/windows/asset_detail/shared/pages/contract_details/contract_details.ui",
            self,
        )

        # INPUT data
        self.asset: Asset = kwargs["asset"]
        self.bl: AssetBL = AssetBL()

        # Business object factory
        self.contractFactory = ContractFactory()

        # signals
        self.updateButton.clicked.connect(self.updateCD)

        self.on_fillTable.connect(self.__fillTable)

        # table
        self.table = AssetContractDetailsTable(
            AssetType.from_str(self.asset.type)
        )
        self.table.tableModel.setData(self.asset.contractDetails)
        self.gridLayout.addWidget(self.table, 1, 0, 1, 1)

    def updateCD(self):
        exchange = self.asset.contractDetails[0].contract.exchange

        contract = self.contractFactory.createNewIBContract(
            SecType.from_str(self.asset.type),
            self.asset.symbol,
            exchange=exchange,
        )

        assetType = AssetType.from_str(self.asset.type)

        self.bl.getContractDetails(assetType, contract).pipe(
            ops.do_action(lambda x: log.info(x)),
            # We have to use pyqtSignal, otherwise Qt will complain about threads
            # --> Cannot set parent, new parent is in a different thread
            ops.do_action(lambda x: self.on_fillTable.emit(x)),
        ).subscribe(self.__updateAsset)

    def __updateAsset(self, data: List[IBContractDetails]):
        # log.info(self.asset.contractDetails)
        # log.info(self.asset.contractDetails[0].contract.symbol)
        self.asset.contractDetails.clear()

        [self.asset.contractDetails.append(item) for item in data]
        # log.info(self.asset.contractDetails)
        # log.info(self.asset.contractDetails[0].contract.symbol)
        self.bl.remove(AssetType.from_str(self.asset.type), self.asset.symbol)
        self.bl.save(self.asset)

    @pyqtSlot(list)
    def __fillTable(self, data: List[IBContractDetails]):

        self.table.tableModel.setData(data)
        self.on_update.emit()
