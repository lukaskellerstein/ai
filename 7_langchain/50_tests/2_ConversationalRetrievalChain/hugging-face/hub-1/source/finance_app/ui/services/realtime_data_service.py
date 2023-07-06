from ui.state.realtime_data import RealtimeDataItem
from business.model.contract_details import IBContractDetails
from business.model.asset import Asset, AssetType
from business.modules.asset_bl import AssetBL
import logging
import random
import threading
from typing import Any, Dict, List, Union

import pandas as pd
from rx import Observable, of
from rx import operators as ops

from business.modules.stocks_watchlist_bl import StocksWatchlistBL
from business.model.contracts import (
    IBStockContract,
    IBContract,
)
from ui.state.main import State
from ui.state.realtime_data import RealtimeDataItemStatus
from business.model.factory.contract_factory import ContractFactory
from helpers import constructKey

# create logger
log = logging.getLogger("CellarLogger")


class RealtimeDataService(object):
    """ Service managing Realtime data. Creates the Realtime subscription if necessary, ortherwise using the UI State.
    """

    def __init__(self):
        log.info("Running ...")

        # State
        self.state = State.getInstance()

        # BL
        self.assetBL = AssetBL()

    def __isRunningRealtime(
        self, assetType: AssetType, symbol: str, localSymbol: str
    ) -> Union[None, RealtimeDataItem]:
        data = None
        if assetType == AssetType.STOCK:
            data = self.state.stocks_realtime_data.get(symbol, localSymbol)
        elif assetType == AssetType.FUTURE:
            data = self.state.futures_realtime_data.get(symbol, localSymbol)
        else:
            raise Exception("THIS SHOULD NOT HAPPENED")

        if data is None or data.status != RealtimeDataItemStatus.RUNNING:
            return None
        else:
            return data

    def startRealtime(
        self, assetType: AssetType, symbol: str, latestContractDetails: int = 1
    ) -> Dict[str, RealtimeDataItem]:

        # Asset
        asset = self.assetBL.get(assetType, symbol)
        if asset is None:
            log.warn(f"Asset with symbol: {symbol} is not found in Asset DB")
            return {}

        # Choose the right contract
        contractDetails = asset.latestContractDetails(latestContractDetails)
        if not contractDetails:
            log.warn(
                f"Asset with symbol: {symbol} has empty ContractDetails in Asset DB"
            )
            return {}

        # State management
        result: Dict[str, RealtimeDataItem] = {}
        for contractDetail in contractDetails:
            # get existing (running)
            stateObject = self.__isRunningRealtime(
                assetType,
                contractDetail.contract.symbol,
                contractDetail.contract.localSymbol,
            )
            if stateObject is None:
                # create or get existing (not running)
                stateObject = self.state.stocks_realtime_data.create(
                    contractDetail.contract.symbol,
                    contractDetail.contract.localSymbol,
                )
                # start if not running
                if stateObject.status != RealtimeDataItemStatus.RUNNING:
                    stateObject.start(
                        self.assetBL.startRealtime(contractDetail.contract)
                    )

            result[
                constructKey(
                    contractDetail.contract.symbol,
                    contractDetail.contract.localSymbol,
                )
            ] = stateObject

        return result

    # def stopRealtime(
    #     self, assetType: AssetType, symbol: str, localSymbol: str
    # ) -> None:
    #     realtimeDataItem = self.__isRunningRealtime(
    #         assetType, symbol, localSymbol
    #     )

    #     if realtimeDataItem is not None:
    #         realtimeDataItem.ticks.dispose()

    #     self.asstBL.stopRealtime(contract)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # destroy BL
        # self.bl.onDestroy()

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
