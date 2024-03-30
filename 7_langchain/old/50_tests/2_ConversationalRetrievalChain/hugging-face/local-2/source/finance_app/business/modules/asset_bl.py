from business.model.factory.asset_factory import AssetFactory
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

from rx import operators as ops
from rx.core.typing import Observable
from rx.subject.behaviorsubject import BehaviorSubject

from business.helpers import getTimeBlocks
from business.model.asset import Asset, AssetType
from business.model.contract_details import IBContractDetails
from business.model.contracts import IBContract
from business.model.timeframe import TimeFrame
from business.services.ibclient.my_ib_client import MyIBClient
from business.tasks.download_hist_data_task import DownloadHistDataTask
from db.services.mongo_asset_service import MongoAssetService
from db.services.pystore_hist_service import PyStoreHistService
from business.model.factory.contract_factory import ContractFactory
from business.model.factory.contract_detail_factory import (
    ContractDetailsFactory,
)

# create logger
log = logging.getLogger("CellarLogger")


class AssetBL(object):
    """ Service integrates DB and IB
    """

    def __init__(self):
        log.info("Running ...")

        # connect to IB
        self.__ibClient = MyIBClient()

        # start thread
        self.__ibClient_thread = threading.Thread(
            name=f"AssetBL-ibClient-{self.__ibClient.uid}-thread",
            target=lambda: self.__ibClient.myStart(),
            daemon=True,
        )
        self.__ibClient_thread.start()

        self.__currentThread = None

        # DB
        self.__assetDbService = MongoAssetService()
        self.__histDataDbService = PyStoreHistService()

        # Business object factory
        self.__assetFactory = AssetFactory()
        self.__contractFactory = ContractFactory()
        self.__contractDetailsFactory = ContractDetailsFactory()

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # BUSINESS LOGIC
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    # ASSET - DB ---------------------------
    def get(self, assetType: AssetType, symbol: str) -> Union[None, Asset]:
        resultDict = self.__assetDbService.findOne(
            assetType, {"symbol": symbol}
        )

        if resultDict == None:
            log.warn(f"Asset with symbol: {symbol} is not found in Asset DB")
            return None

        result = self.__assetFactory.createAsset(resultDict)
        return result

    def isExist(self, assetType: AssetType, symbol: str) -> bool:
        isExist = self.__assetDbService.findOne(assetType, {"symbol": symbol})
        return True if isExist is not None else False

    def save(self, asset: Asset):
        dbobject = self.__assetFactory.createDict(asset)
        self.__assetDbService.add(AssetType.from_str(asset.type), dbobject)

    def getAll(self, assetType: AssetType) -> List[Asset]:
        dbobjects = self.__assetDbService.getAll(assetType)
        objects: List[Asset] = [
            self.__assetFactory.createAsset(dbobject) for dbobject in dbobjects
        ]
        return objects

    def remove(self, assetType: AssetType, symbol: str):
        self.__assetDbService.remove(assetType, {"symbol": symbol})

    # CONTRACT DETAILS - IB ---------------------------
    def getContractDetails(
        self, assetType: AssetType, contract: IBContract
    ) -> Observable[IBContractDetails]:
        log.info(contract)
        if assetType == AssetType.FUTURE:
            return self.__ibClient.getContractDetail(contract).pipe(
                ops.filter(lambda x: x is not None),
                ops.buffer_with_time(2),
                ops.take(1),
            )
        elif assetType == AssetType.STOCK:
            return self.__ibClient.getContractDetail(contract).pipe(
                ops.filter(lambda x: x is not None),
                ops.buffer_with_time(1),
                ops.take(1),
            )

    def getLatestContractDetails(
        self, assetType: AssetType, symbol: str, latestContractDetails: int = 1
    ) -> List[IBContractDetails]:
        # Asset
        asset = self.get(assetType, symbol)
        if asset is None:
            log.warn(f"Asset with symbol: {symbol} is not found in Asset DB")
            return []

        # Choose the right contract
        contractDetails = asset.latestContractDetails(latestContractDetails)
        return contractDetails if contractDetails is not None else []

    # REALTIME DATA - IB ---------------------------
    def startRealtime(self, contract: IBContract) -> Observable[Any]:
        return self.__ibClient.startRealtimeData(contract).pipe(
            ops.filter(lambda x: x is not None),
        )

    def stopRealtime(self, contract: IBContract) -> None:
        return self.__ibClient.stopRealtimeData(contract)

    # HISTORICAL DATA - IB + DB ---------------------------
    def downloadHistoricalData(
        self,
        assets: List[Asset],
        timeframe: TimeFrame = TimeFrame.day1,
        maxBlockSize: int = 365,  # in days
    ) -> Observable[Any]:
        progressResult0000 = BehaviorSubject(0)

        contractsAndTimeBlocks = []

        for asset in assets:
            if AssetType.from_str(asset.type) == AssetType.STOCK:
                contractsAndTimeBlocks.extend(
                    self.__downloadStock(asset, maxBlockSize)
                )
            elif AssetType.from_str(asset.type) == AssetType.FUTURE:
                contractsAndTimeBlocks.extend(
                    self.__downloadFutures(asset, maxBlockSize)
                )

        self.__currentThread = DownloadHistDataTask(
            self.__ibClient,
            progressResult0000,
            contractsAndTimeBlocks,
            timeframe,
        )
        self.__currentThread.start()

        return progressResult0000

    def updateHistoricalData(
        self,
        assets: List[Asset],
        timeframe: TimeFrame = TimeFrame.day1,
        maxBlockSize: int = 365,  # in days
    ) -> Observable[Any]:
        progressResult0000 = BehaviorSubject(0)

        contractsAndTimeBlocks = []

        for asset in assets:
            if AssetType.from_str(asset.type) == AssetType.STOCK:
                contractsAndTimeBlocks.extend(
                    self.__updateStock(asset, timeframe, maxBlockSize)
                )

            elif AssetType.from_str(asset.type) == AssetType.FUTURE:
                contractsAndTimeBlocks.extend(
                    self.__updateFutures(asset, timeframe, maxBlockSize)
                )

        log.info(contractsAndTimeBlocks)

        self.__currentThread = DownloadHistDataTask(
            self.__ibClient,
            progressResult0000,
            contractsAndTimeBlocks,
            timeframe,
        )
        self.__currentThread.start()

        return progressResult0000

    def getHistoricalDataFromDB(self, symbol: str, timeframe: TimeFrame):
        return self.__histDataDbService.getAll(symbol, timeframe)

    # FUNDAMENTALS DATA - IB -----------------------------------
    def getFundamentals(self, contract: IBContract) -> Observable[Any]:
        return self.__ibClient.getFundamentalData(contract)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # HELPER METHODS
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    def __updateFutures(
        self, asset: Asset, timeframe: TimeFrame, maxBlockSize: int
    ) -> List[Dict]:
        result = []

        for cd in asset.contractDetails:

            lastTradeDateTime = datetime.strptime(
                cd.contract.lastTradeDateOrContractMonth, "%Y%m%d"
            )
            now = datetime.now()

            if lastTradeDateTime > now:

                localSymbol = f"{cd.contract.localSymbol}-{cd.contract.lastTradeDateOrContractMonth}"
                symbolData = self.getHistoricalDataFromDB(
                    localSymbol, timeframe
                )

                if symbolData is not None:

                    # WE HAVE SOME DATA IN DB -> UPDATE
                    lastDateTime = symbolData.tail(1).index[0]

                    if now > lastDateTime:

                        timeBlocks = getTimeBlocks(
                            lastDateTime, now, maxBlockSize,
                        )

                        for timeBlock in timeBlocks:
                            result.append(
                                {
                                    "contract": cd.contract,
                                    "from": timeBlock[0],
                                    "to": timeBlock[1],
                                }
                            )
                else:

                    # OTHERWISE DOWNLOAD FULL
                    timeBlocks = getTimeBlocks(
                        datetime.strptime("19860101", "%Y%m%d"),
                        now,
                        maxBlockSize,
                    )

                    for timeBlock in timeBlocks:
                        result.append(
                            {
                                "contract": cd.contract,
                                "from": timeBlock[0],
                                "to": timeBlock[1],
                            }
                        )

            else:
                log.info(
                    f" SKIPPED - {cd.contract.localSymbol} - {cd.contract.lastTradeDateOrContractMonth}"
                )

        return result

    def __downloadFutures(self, asset: Asset, maxBlockSize: int) -> List[Dict]:
        result = []

        for cd in asset.contractDetails:

            lastTradeDateTime = datetime.strptime(
                cd.contract.lastTradeDateOrContractMonth, "%Y%m%d"
            )
            now = datetime.now()

            if (
                lastTradeDateTime > datetime.strptime("19860101", "%Y%m%d")
                and lastTradeDateTime < now
            ):
                result.append(
                    {
                        "contract": cd.contract,
                        "from": lastTradeDateTime
                        - timedelta(days=maxBlockSize),
                        "to": lastTradeDateTime,
                    }
                )
            elif lastTradeDateTime >= now:
                result.append(
                    {
                        "contract": cd.contract,
                        "from": now - timedelta(days=maxBlockSize),
                        "to": now,
                    }
                )

        return result

    def __updateStock(
        self, asset: Asset, timeframe: TimeFrame, maxBlockSize: int
    ) -> List[Dict]:
        result = []

        contract = asset.contractDetails[0].contract
        symbolData = self.getHistoricalDataFromDB(asset.symbol, timeframe)

        now = datetime.now()

        if symbolData is not None:

            # WE HAVE SOME DATA IN DB -> UPDATE
            lastDateTime = symbolData.tail(1).index[0]

            if now > lastDateTime:

                timeBlocks = getTimeBlocks(lastDateTime, now, maxBlockSize,)

                for timeBlock in timeBlocks:
                    result.append(
                        {
                            "contract": contract,
                            "from": timeBlock[0],
                            "to": timeBlock[1],
                        }
                    )

        else:

            # OTHERWISE DOWNLOAD FULL
            timeBlocks = getTimeBlocks(
                datetime.strptime("19860101", "%Y%m%d"), now, maxBlockSize,
            )

            for timeBlock in timeBlocks:
                result.append(
                    {
                        "contract": contract,
                        "from": timeBlock[0],
                        "to": timeBlock[1],
                    }
                )

        return result

    def __downloadStock(self, asset: Asset, maxBlockSize: int) -> List[Dict]:
        result = []

        contract = asset.contractDetails[0].contract

        timeBlocks = getTimeBlocks(
            datetime.strptime("19860101", "%Y%m%d"),
            datetime.now(),
            maxBlockSize,
        )

        for timeBlock in timeBlocks:
            result.append(
                {
                    "contract": contract,
                    "from": timeBlock[0],
                    "to": timeBlock[1],
                }
            )

        return result

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. - CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        if self.__currentThread is not None:
            self.__currentThread.terminate()

        # Close DB
        self.__assetDbService.client.close()
        self.__assetDbService.db.logout()

        # Close IB
        self.__ibClient.connectionClosed()  # close the EWrapper
        self.__ibClient.disconnect()  # close the EClient

    # 2. - Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
