from business.model.timeframe import TimeFrame
from business.services.ibclient.my_ib_client import MyIBClient
import logging
from threading import Thread
from typing import Any, Dict, List, Tuple
from db.services.pystore_hist_service import PyStoreHistService
import time
from rx import operators as ops
from rx.subject.behaviorsubject import BehaviorSubject
from helpers import logThreads
from datetime import datetime

from business.model.contracts import IBContract

from business.tasks.download_one_task import DownloadOneDateTask

# create logger
log = logging.getLogger("CellarLogger")


class DownloadHistDataTask(Thread):
    subscriptions = []

    currentThreads: List[DownloadOneDateTask] = []
    maximumThreads = 1

    histDataOriginCount: int = 0
    histDataCurrentCount: int = 0
    itemsProgress: BehaviorSubject
    groupProgress: BehaviorSubject

    def __init__(
        self,
        client: MyIBClient,
        progress: BehaviorSubject,
        contractsAndDates: List[Dict],
        timeframe: TimeFrame = TimeFrame.day1,
    ):
        Thread.__init__(self)
        self.daemon = True

        self._running = True
        self.ibClient = client
        self.groupProgress = progress
        self.itemsProgress = BehaviorSubject(0)
        self.contractsAndDates = contractsAndDates
        self.timeframe = timeframe

        # db
        self.histDataDbService = PyStoreHistService()

        self.histDataOriginCount = len(contractsAndDates)

        subscriptionTemp = self.itemsProgress.subscribe(
            lambda x: self.recalculateProgress(x)
        )
        self.subscriptions.append(subscriptionTemp)

    def run(self):
        log.info(".... Thread is Starting ....")
        start = time.time()

        currentIndex = 0
        for obj in self.contractsAndDates:
            if self._running is False:
                break

            date = (obj["from"], obj["to"])
            contract = obj["contract"]

            aaa = DownloadOneDateTask(
                self.ibClient,
                self.itemsProgress,
                date,
                contract,
                self.timeframe,
            )
            log.info(
                f"run thread for {contract.localSymbol} download data - date - {date}"
            )
            aaa.start()

            self.currentThreads.append(aaa)

            currentIndex += 1
            if currentIndex >= self.maximumThreads:
                currentIndex = 0

                while True:
                    # log.info("waiting...1 sec.")
                    time.sleep(1)

                    if self._running is False:
                        break

                    areAllDead = True
                    for t in self.currentThreads:
                        # print(f"{t} - {t.isAlive()}")
                        if t.isAlive() is True:
                            areAllDead = False

                    if areAllDead is True:
                        log.info("----------------------------------------")
                        log.info("------- bunch Threads is DONE ----------")
                        log.info("----------------------------------------")
                        logThreads()

                        # terminate all threads
                        [th.terminate() for th in self.currentThreads]
                        self.currentThreads = []

                        break

        # ----------------------
        # END of Thread
        # ----------------------
        log.info(".... Thread is Ending ....")
        end = time.time()
        log.info(f"Download HistData Task takes: {end - start} sec.")

    def recalculateProgress(self, value):
        self.histDataCurrentCount += value
        progress = (self.histDataCurrentCount / self.histDataOriginCount) * 100

        self.groupProgress.on_next(progress)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # STOP & DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    def terminate(self):
        # Unsubscribe everything
        for sub in self.subscriptions:
            sub.dispose()

        self.itemsProgress = None
        self.groupProgress = None

        # terminate all threads
        [th.terminate() for th in self.currentThreads]
        self.currentThreads = []

        self.histDataDbService = None

        # stop the Thread
        self._running = False
