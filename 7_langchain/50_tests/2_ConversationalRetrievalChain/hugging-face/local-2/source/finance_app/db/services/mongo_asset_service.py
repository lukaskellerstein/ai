import logging
from typing import Any, List, Dict

import pandas as pd
import pymongo

from business.model.asset import Asset, AssetType


# create logger
log = logging.getLogger("CellarLogger")


class MongoAssetService:
    def __init__(self):
        self.client: pymongo.MongoClient = pymongo.MongoClient(
            "mongodb://localhost:27017/"
        )
        self.db: Any = self.client["cellarstone-app"]
        # self.stocks_assets_table = self.db["stock-assets"]
        # self.future_assets_table = self.db["futures-assets"]

    # -----------------------------------------------------------------
    # Inner methods
    # -----------------------------------------------------------------
    def __find(
        self, assetType: AssetType, findObject: Dict[str, str] = {}
    ) -> List[Dict[str, str]]:
        return self.db[f"{assetType.value}-assets"].find(findObject)

    def __findOne(
        self, assetType: AssetType, findObject: Dict[str, str] = {}
    ) -> Dict[str, str]:
        return self.db[f"{assetType.value}-assets"].find_one(findObject)

    def __add(self, assetType: AssetType, obj: Dict[str, str] = {}):
        self.db[f"{assetType.value}-assets"].insert_one(obj)

    def __remove(self, assetType: AssetType, findObject: Dict[str, str]):
        self.db[f"{assetType.value}-assets"].delete_one(findObject)

    # -----------------------------------------------------------------
    # Specific
    # -----------------------------------------------------------------

    def add(self, assetType: AssetType, asset: Dict[str, Any]):
        self.__add(assetType, asset)

    def findOne(self, assetType: AssetType, findObject: Dict[str, str]):
        return self.__findOne(assetType, findObject)

    def getAll(self, assetType: AssetType) -> List[Dict[str, Any]]:
        return list(map(lambda x: x, self.__find(assetType)))

    def remove(self, assetType: AssetType, findObject: Dict[str, str]):
        self.__remove(assetType, findObject)

    # def add(self, assetType: AssetType, asset: Asset):
    #     self.__add(assetType, obj_to_dict(asset))

    # def findOne(self, assetType: AssetType, findObject: Dict[str, str]):
    #     return self.__findOne(assetType, findObject)

    # def getAll(self, assetType: AssetType) -> List[Asset]:
    #     return list(map(lambda x: dict_to_obj(x), self.__find(assetType)))

    # def remove(self, assetType: AssetType, findObject: Dict[str, str]):
    #     self.__remove(assetType, findObject)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
