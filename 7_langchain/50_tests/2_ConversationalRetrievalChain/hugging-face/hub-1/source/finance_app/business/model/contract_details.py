import logging
import traceback
from typing import Any, Dict

from business.model.contracts import IBContract, IBStockContract
from db.model.base import DBObject
from ibapi.contract import ContractDetails

# create logger
log = logging.getLogger("CellarLogger")


class IBContractDetails(ContractDetails, DBObject):
    def __init__(self):
        ContractDetails.__init__(self)
        DBObject.__init__(self, self.__module__, type(self).__name__)

        self.contract = IBContract()

