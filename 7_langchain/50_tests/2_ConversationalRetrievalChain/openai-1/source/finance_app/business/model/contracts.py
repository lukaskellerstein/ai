import logging

from ibapi.contract import Contract
from db.model.base import DBObject
import traceback

# create logger
log = logging.getLogger("CellarLogger")


class IBContract(Contract, DBObject):
    def __init__(self):
        Contract.__init__(self)
        DBObject.__init__(self, self.__module__, type(self).__name__)


class IBStockContract(IBContract):
    def __init__(self):
        super().__init__()

        self.secType = "STK"
        self.exchange = "SMART"
        self.currency = "USD"
        self.primaryExchange = "NASDAQ"


class IBFutureContract(IBContract):
    def __init__(self,):
        super().__init__()

        self.secType = "FUT"
        self.includeExpired = True


class IBOptionContract(IBContract):
    def __init__(self):
        super().__init__()

        self.secType = "OPT"
        self.currency = "USD"

