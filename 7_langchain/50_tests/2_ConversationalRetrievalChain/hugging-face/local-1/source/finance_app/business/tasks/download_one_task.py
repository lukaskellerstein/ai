import logging
import random
import time
from datetime import datetime
from threading import Thread
from typing import Any, List, Tuple

from rx import operators as ops
from rx.subject.behaviorsubject import BehaviorSubject

from business.model.contracts import (
    IBContract,
    IBFutureContract,
    IBStockContract,
)
from business.model.timeframe import TimeFrame
from business.services.ibclient.my_ib_client import MyIBClient
from db.services.pystore_hist_service import PyStoreHistService
from helpers import logThreads

# create logger
log = logging.getLogger("CellarLogger")


class DownloadOneDateTask(Thread):
    def __init__(
        self,
        client: MyIBClient,
        progress: BehaviorSubject,
        date,
        contract,
        timeframe: TimeFrame = TimeFrame.day1,
    ):
        super().__init__()
        self.daemon = True

        self.uid = random.randint(1000, 10000)

        self._running = True
        self.ibClient = client
        self.progress = progress
        self.histDataDbService = PyStoreHistService()

        self.date = date
        self.contract = contract
        self.timeframe = timeframe

        self.subscriptions = []

    def run(self):
        log.info(f"{self.uid}.... Thread is Starting ....")

        (startTemp, endTemp) = self.date

        blockSizeCalculated = (endTemp - startTemp).days

        logText = f"requested historical data from {startTemp.strftime('%Y%m%d %H:%M:%S')} to {endTemp.strftime('%Y%m%d %H:%M:%S')} with {blockSizeCalculated}D block size "
        log.info(logText)
        # self.histDataLog.on_next(logText)

        subscriptionTemp = (
            self.ibClient.getHistoricalData(
                self.contract,
                endTemp.strftime("%Y%m%d %H:%M:%S"),
                f"{blockSizeCalculated} D",
                self.timeframe.value,
                "TRADES",
            )
            .pipe(
                ops.filter(lambda x: x is not None),
                ops.do_action(
                    lambda x: self.__saveToDB(self.contract, self.timeframe, x)
                ),
            )
            .subscribe(self.__histDataSubscribe)
        )

        self.subscriptions.append(subscriptionTemp)

        # ----------------------
        # wait until END
        # ----------------------
        while self._running:
            # log.info(f"{self.uid}.... waiting...0.2 sec.")
            time.sleep(0.2)

        # ----------------------
        # END of Thread
        # ----------------------
        log.info(f"{self.uid}.... Thread Ended ....")

    def __histDataSubscribe(
        self, data: List[Tuple[datetime, float, float, float, float, float]]
    ):
        # self.histDataCurrentCount += 1
        # progress = (self.histDataCurrentCount / self.histDataOriginCount) * 100
        self.progress.on_next(1)

        if len(data) > 0:
            first = data[0][0]
            last = data[len(data) - 1][0]

            logText = f"downloaded historical data from {first.strftime('%Y%m%d %H:%M:%S')} to {last.strftime('%Y%m%d %H:%M:%S')}"
        else:
            logText = "no data"

        log.info(f"{self.uid}.... {logText}")
        # self.histDataLog.on_next(logText)

        self.terminate()

    def __saveToDB(
        self,
        contract: IBContract,
        timeframe: TimeFrame,
        data: List[Tuple[datetime, float, float, float, float, float]],
    ):
        fullSymbolName = f"{contract.localSymbol}"
        if isinstance(contract, IBFutureContract):
            fullSymbolName = f"{contract.localSymbol}-{contract.lastTradeDateOrContractMonth}"

        log.info(f"saveToDb - {fullSymbolName}")
        # log.info(data)

        self.histDataDbService.add(fullSymbolName, timeframe, data)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # STOP & DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    def terminate(self):
        log.info(f"{self.uid}.... Terminate one task")
        # print(self.subscriptions)

        # stop the Thread
        self._running = False

        # Unsubscribe everything
        for sub in self.subscriptions:
            sub.dispose()

        self.histDataDbService = None
