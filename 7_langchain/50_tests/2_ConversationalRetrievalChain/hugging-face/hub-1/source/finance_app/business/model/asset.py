from __future__ import (
    annotations,
)  # allow return same type as class ..... -> State

from business.model.contract_details import IBContractDetails
from typing import List, Union
from db.model.base import DBObject
from enum import Enum
from datetime import date, datetime


class AssetType(Enum):
    NONE = "none"
    STOCK = "stock"
    FUTURE = "future"

    @staticmethod
    def from_str(value: str) -> AssetType:
        if value.lower() == AssetType.STOCK.value:
            return AssetType.STOCK
        elif value.lower() == AssetType.FUTURE.value:
            return AssetType.FUTURE
        else:
            return AssetType.NONE


class Asset(DBObject):
    def __init__(self):
        DBObject.__init__(self, self.__module__, type(self).__name__)

        self.symbol: str = ""
        self.shortDescription: str = ""
        self.type: str = ""
        self.contractDetails: List[IBContractDetails] = []

    def latestContractDetails(
        self, count: int = 1
    ) -> Union[None, List[IBContractDetails]]:
        if self.contractDetails.count == 0:
            return None
        elif self.type == AssetType.STOCK.value:
            self.contractDetails.sort(
                key=lambda x: x.contract.lastTradeDateOrContractMonth
            )
            return self.contractDetails[:count]
        elif self.type == AssetType.FUTURE.value:
            result = list(
                filter(self.__filterOlderThanToday, self.contractDetails)
            )
            return self.__chooseContractMonths(result, count)

    def __chooseContractMonths(
        self, data: List[IBContractDetails], count: int
    ) -> List[IBContractDetails]:
        data.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
        return data[:count]

    def __filterOlderThanToday(self, cd: IBContractDetails) -> bool:
        lastDate = datetime.strptime(
            cd.contract.lastTradeDateOrContractMonth, "%Y%m%d"
        ).date()
        nowDate = date.today()
        if lastDate < nowDate:
            return False
        else:
            return True
