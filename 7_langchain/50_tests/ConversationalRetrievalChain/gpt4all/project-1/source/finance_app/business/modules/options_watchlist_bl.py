from business.modules.asset_bl import AssetBL
from db.services.mongo_asset_service import MongoAssetService
import logging
import threading

from typing import Any

from rx import of
from rx import operators as ops
from rx.core.typing import Observable

from business.model.contracts import (
    IBContract,
    IBOptionContract,
)
from business.services.ibclient.my_ib_client import MyIBClient

# from db.services.mongo_service import MongoService

import pandas as pd
from business.model.factory.contract_factory import ContractFactory
from business.model.factory.contract_detail_factory import (
    ContractDetailsFactory,
)
from business.model.asset import AssetType

# create logger
log = logging.getLogger("CellarLogger")


class OptionsWatchlistBL(object):
    def __init__(self):
        log.info("Running ...")

        # connect to IB
        self.ibClient = MyIBClient()

        # start thread
        self.ibClient_thread = threading.Thread(
            name="OptionsWatchlistBL-ibClient-thread",
            target=lambda: self.ibClient.myStart(),
            daemon=True,
        )
        self.ibClient_thread.start()

        # DB
        # self.dbService = MongoService()

        # Asset BL
        self.assetBl = AssetBL()

        # Business object factory
        self.contractFactory = ContractFactory()
        self.contractDetailsFactory = ContractDetailsFactory()

    def getOptionChain(self, symbol: str) -> Observable[Any]:
        log.debug("Running...")
        log.debug(locals())

        # contractDetailDict = self.dbService.getStockContractDetail(
        #     symbol, symbol
        # )

        # contractDetail = self.contractDetailsFactory.createIBContractDetails(
        #     contractDetailDict
        # )

        # if contractDetail is not None:

        #     contractFull = self.contractFactory.createIBContract(
        #         contractDetail.contract
        #     )
        #     return self.ibClient.getOptionChain(contractFull).pipe(
        #         ops.filter(lambda x: x is not None),  # filter empty
        #         ops.filter(
        #             lambda x: self.__filterExchange(x, contractFull.exchange)
        #         ),
        #     )
        # else:
        #     print("-----------------------------------------------")
        #     print("-----------------------------------------------")
        #     print("-----------------------------------------------")
        #     print("????????")
        #     print("-----------------------------------------------")
        #     print("-----------------------------------------------")
        #     print("-----------------------------------------------")
        #     return of(None)

    # region "getOptionChain" operators

    def __filterExchange(self, data: pd.DataFrame, exchange: str) -> bool:
        return True if data["exchange"] == exchange else False

    # endregion

    def getOptionPrice(
        self,
        contract: IBOptionContract,
        volatility: float,
        underPrice: float,
        timeout: int,
    ) -> Observable[Any]:
        return self.ibClient.getOptionPrice(
            contract, volatility, underPrice, timeout
        )

    # def getPrice(self, contract) -> Observable:
    #     return self.ibClient.startRealtimeData(contract).pipe(
    #         ops.do_action(lambda x: print(x))
    #     )

    # def getVolatility(self) -> Observable:
    #     return of(None)

    # def addToWatchlist(self, contract):
    #     self.dbService.addToFuturesWatchlist_IfNotExists(contract.symbol)

    # def startRealtime(self, contract, fn):
    #     # print(contract.symbol)
    #     # print(contract.localSymbol)
    #     self.ibClient.startRealtimeData(contract, fn)

    # def remove(self, ticker, updateDB=True):
    #     if updateDB:
    #         self.dbService.removeFromFuturesWatchlist(ticker)

    #     self.ibClient.stopRealtimeData(ticker)

    # def updateWatchlist(self, arr):
    #     self.dbService.futures_watchlist_table.drop()
    #     for item in arr:
    #         self.dbService.addToFuturesWatchlist(item)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. - CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # Close DB
        # self.dbService.client.close()
        # self.dbService.db.logout()

        # Close IB
        self.ibClient.connectionClosed()  # close the EWrapper
        self.ibClient.disconnect()  # close the EClient

    # 2. - Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
