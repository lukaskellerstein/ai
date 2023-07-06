from __future__ import (
    annotations,
)  # allow return same type as class ..... -> State

import logging
from typing import Dict, List, Any
from enum import Enum
from ibapi.contract import Contract

from business.model.asset import Asset, AssetType
from business.model.factory.contract_detail_factory import (
    ContractDetailsFactory,
)
from business.model.contract_details import IBContractDetails

# create logger
log = logging.getLogger("CellarLogger")


class AssetFactory(object):
    def __init__(self):
        log.debug("Running ...")
        self.contractDetailsFactory = ContractDetailsFactory()

    def createNewAsset(self, assetType: AssetType, symbol: str,) -> Asset:
        result: Asset = Asset()
        result.type = assetType.value
        result.symbol = symbol
        return result

    def createAsset(self, asset: Dict[str, Any]) -> Asset:
        return self._createAsset_fromDict(asset)

    def _createAsset_fromDict(self, asset: Dict[str, Any],) -> Asset:

        result: Asset = Asset()

        result.symbol = asset["symbol"]
        result.shortDescription = asset["shortDescription"]
        result.type = asset["type"]

        # contract details
        cds: List[IBContractDetails] = []
        for cd in asset["contractDetails"]:
            cdDict = self.contractDetailsFactory.createIBContractDetails(cd)
            cds.append(cdDict)

        result.contractDetails = cds

        return result

    def createDict(self, asset: Asset) -> Dict[str, Any]:

        result: Dict[str, Any] = {}

        result["symbol"] = asset.symbol
        result["shortDescription"] = asset.shortDescription
        result["type"] = asset.type

        # contract details
        cds: List[Dict[str, Any]] = []
        for cd in asset.contractDetails:
            cdDict = self.contractDetailsFactory.createDict(cd)
            cds.append(cdDict)

        result["contractDetails"] = cds

        return result
