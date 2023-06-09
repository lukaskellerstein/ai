from __future__ import (
    annotations,
)  # allow return same type as class ..... -> State

import logging
from typing import Dict, Union, Any
from enum import Enum
from ibapi.contract import Contract

from business.model.contracts import (
    IBFutureContract,
    IBStockContract,
    IBContract,
)

# create logger
log = logging.getLogger("CellarLogger")


class SecType(Enum):
    NONE = "none"
    STOCK = "stock"
    FUTURE = "future"

    @staticmethod
    def from_str(value: str) -> SecType:
        if value.lower() == SecType.STOCK.value:
            return SecType.STOCK
        elif value.lower() == SecType.FUTURE.value:
            return SecType.FUTURE
        else:
            return SecType.NONE


class ContractFactory(object):
    def __init__(self):
        log.debug("Running ...")

    def createNewIBContract(
        self,
        secType: SecType,
        symbol: str,
        localSymbol: str = "",
        exchange: str = "",
    ) -> Union[IBContract, IBStockContract, IBFutureContract]:
        # contract -------------
        resultContract: IBContract

        if secType == SecType.STOCK:
            resultContract = IBStockContract()
        elif secType == SecType.FUTURE:
            resultContract = IBFutureContract()
        else:
            resultContract = IBContract()

        resultContract.symbol = symbol
        if localSymbol != "":
            resultContract.localSymbol = localSymbol
        if exchange != "":
            resultContract.exchange = exchange

        return resultContract

    def createIBContract(
        self, contract: Union[Contract, Dict[str, Any]]
    ) -> IBContract:

        if type(contract) is Contract:
            return self._createIBContract_fromContract(contract)
        elif type(contract) is dict:
            return self._createIBContract_fromDict(contract)
        else:
            raise Exception("THIS SHOULD NOT HAPPENED")

    def _createIBContract_fromContract(
        self, contract: Contract,
    ) -> Union[IBContract, IBStockContract, IBFutureContract]:

        resultContract: IBContract

        if contract.secType == "STK":
            resultContract = IBStockContract()
        elif contract.secType == "FUT":
            resultContract = IBFutureContract()
        else:
            resultContract = IBContract()

        if resultContract is not None:
            resultContract.conId = contract.conId
            resultContract.symbol = contract.symbol
            resultContract.secType = contract.secType
            resultContract.lastTradeDateOrContractMonth = (
                contract.lastTradeDateOrContractMonth
            )
            resultContract.strike = contract.strike
            resultContract.right = contract.right
            resultContract.multiplier = contract.multiplier
            resultContract.exchange = contract.exchange
            resultContract.primaryExchange = contract.primaryExchange
            resultContract.currency = contract.currency
            resultContract.localSymbol = contract.localSymbol
            resultContract.tradingClass = contract.tradingClass
            resultContract.includeExpired = contract.includeExpired
            resultContract.secIdType = contract.secIdType
            resultContract.secId = contract.secId
            resultContract.comboLegsDescrip = contract.comboLegsDescrip
            resultContract.comboLegs = contract.comboLegs
            # resultContract.deltaNeutralContract = contract.deltaNeutralContract
        # ---------------------

        return resultContract

    def _createIBContract_fromDict(
        self, contract: Dict[str, Any],
    ) -> Union[IBContract, IBStockContract, IBFutureContract]:

        resultContract: IBContract

        if contract["secType"] == "STK":
            resultContract = IBStockContract()
        elif contract["secType"] == "FUT":
            resultContract = IBFutureContract()
        else:
            resultContract = IBContract()

        if resultContract is not None:
            resultContract.conId = contract["conId"]
            resultContract.symbol = contract["symbol"]
            resultContract.secType = contract["secType"]
            resultContract.lastTradeDateOrContractMonth = contract[
                "lastTradeDateOrContractMonth"
            ]
            resultContract.strike = contract["strike"]
            resultContract.right = contract["right"]
            resultContract.multiplier = contract["multiplier"]
            resultContract.exchange = contract["exchange"]
            resultContract.primaryExchange = contract["primaryExchange"]
            resultContract.currency = contract["currency"]
            resultContract.localSymbol = contract["localSymbol"]
            resultContract.tradingClass = contract["tradingClass"]
            resultContract.includeExpired = contract["includeExpired"]
            resultContract.secIdType = contract["secIdType"]
            resultContract.secId = contract["secId"]
            resultContract.comboLegsDescrip = contract["comboLegsDescrip"]
            resultContract.comboLegs = contract["comboLegs"]
            # resultContract.deltaNeutralContract = contract["deltaNeutralContract"]
        # ---------------------

        return resultContract

    def createContract(self, contract: IBContract) -> Contract:

        resultContract: Contract = Contract()

        resultContract.conId = contract.conId
        resultContract.symbol = contract.symbol
        resultContract.secType = contract.secType
        resultContract.lastTradeDateOrContractMonth = (
            contract.lastTradeDateOrContractMonth
        )
        resultContract.strike = contract.strike
        resultContract.right = contract.right
        resultContract.multiplier = contract.multiplier
        resultContract.exchange = contract.exchange
        resultContract.primaryExchange = contract.primaryExchange
        resultContract.currency = contract.currency
        resultContract.localSymbol = contract.localSymbol
        resultContract.tradingClass = contract.tradingClass
        resultContract.includeExpired = contract.includeExpired
        resultContract.secIdType = contract.secIdType
        resultContract.secId = contract.secId
        resultContract.comboLegsDescrip = contract.comboLegsDescrip
        resultContract.comboLegs = contract.comboLegs
        # resultContract.deltaNeutralContract = contract.deltaNeutralContract

        return resultContract

    def createDict(
        self, contract: Union[Contract, IBContract]
    ) -> Dict[str, Any]:

        resultContract: Dict[str, Any] = {}

        resultContract["conId"] = contract.conId
        resultContract["symbol"] = contract.symbol
        resultContract["secType"] = contract.secType
        resultContract[
            "lastTradeDateOrContractMonth"
        ] = contract.lastTradeDateOrContractMonth
        resultContract["strike"] = contract.strike
        resultContract["right"] = contract.right
        resultContract["multiplier"] = contract.multiplier
        resultContract["exchange"] = contract.exchange
        resultContract["primaryExchange"] = contract.primaryExchange
        resultContract["currency"] = contract.currency
        resultContract["localSymbol"] = contract.localSymbol
        resultContract["tradingClass"] = contract.tradingClass
        resultContract["includeExpired"] = contract.includeExpired
        resultContract["secIdType"] = contract.secIdType
        resultContract["secId"] = contract.secId
        resultContract["comboLegsDescrip"] = contract.comboLegsDescrip
        resultContract["comboLegs"] = contract.comboLegs
        # resultContract["deltaNeutralContract"] = contract.deltaNeutralContract

        return resultContract
