import logging
from typing import Any, List, Union

from business.model.asset import AssetType
from business.model.contracts import IBContract
from business.model.factory.contract_detail_factory import (
    ContractDetailsFactory,
)
from business.model.factory.contract_factory import ContractFactory
from business.services.ibclient.my_ib_client import MyIBClient
from db.services.file_watchlist_service import FileWatchlistService
from business.model.asset import Asset
from business.modules.asset_bl import AssetBL

# create logger
log = logging.getLogger("CellarLogger")


allowExchanges = ["GLOBEX", "ECBOT", "NYMEX", "NYBOT"]


class FuturesWatchlistBL(object):
    def __init__(self):
        log.info("Running ...")

        # # connect to IB
        # self.ibClient = MyIBClient()

        # # start thread
        # self.ibClient_thread = threading.Thread(
        #     name="FuturesWatchlistBL-ibClient-thread",
        #     target=lambda: self.ibClient.myStart(),
        #     daemon=True,
        # )
        # self.ibClient_thread.start()

        # DB
        self.db = FileWatchlistService()
        self.assetBL = AssetBL()

        # Business object factory
        # self.contractFactory = ContractFactory()

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # BUSINESS LOGIC
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    def getWatchlist(self) -> List[str]:
        return self.db.getWatchlist(AssetType.FUTURE.value)

    def addToWatchlist(self, symbol: str):
        self.db.addSymbol(AssetType.FUTURE.value, symbol)

    def remove(self, symbol: str):
        self.db.removeSymbol(AssetType.FUTURE.value, symbol)

        # stop receiving data from Broker
        # contract = self.contractFactory.createNewIBContract(
        #     SecType.FUTURE, symbol, localSymbol
        # )
        # self.ibClient.stopRealtimeData(contract)

    def updateWatchlist(self, arr: List[Any]):
        self.db.updateWatchlist(AssetType.STOCK.value, arr)

    # ----------------------------------------------------------
    # ASSET
    # ----------------------------------------------------------

    def getAsset(
        self, assetType: AssetType, symbol: str
    ) -> Union[None, Asset]:
        return self.assetBL.get(assetType, symbol)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. - CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # Close IB
        # self.ibClient.connectionClosed()  # close the EWrapper
        # self.ibClient.disconnect()  # close the EClient

    # 2. - Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
