from __future__ import (
    annotations,
)  # allow return same type as class ..... -> State

import logging
import random
import threading
from collections import defaultdict
from typing import Tuple

import pandas as pd

from rx.core.typing import Observable, Subject
from rx.subject import BehaviorSubject

from business.model.contracts import IBContract

# from typings import ObservableType, PandasDataFrameType, PandasSerieType
from typing import List, Any

# create logger
log = logging.getLogger("CellarLogger")


dfColumns = ["Symbol", "LocalSymbol", "ReqId", "ObservableName"]


class State(object):
    __instance = None

    __data: pd.DataFrame = pd.DataFrame(columns=dfColumns)
    __observablesInstances = defaultdict(None)  # key: reqId, value: Observable

    __tempData = defaultdict(None)  # key: reqId, value: List[Any]

    reqId = 0

    @staticmethod
    def getInstance() -> State:
        """ Static access method. """
        if State.__instance == None:
            State.__instance = State()
        return State.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if State.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            State.__instance = self
            self.uid = random.randint(1000, 10000)
            self.lock = threading.Lock()

    def log(self):
        print(self.__data)

    # -----------------------------------------------------------------
    # OBSERVABLES
    # -----------------------------------------------------------------

    def getObservable(self, key: int) -> Observable:
        return self.__observablesInstances[key]

    def removeObservable(
        self, reqId: int, contract: IBContract, observableName: str
    ):

        # remove instance
        self.__observablesInstances.pop(reqId)

        # get data from DataFrame
        isExist: pd.DataFrame = self.__data[
            (self.__data["Symbol"] == contract.symbol)
            & (self.__data["LocalSymbol"] == contract.localSymbol)
            & (self.__data["ObservableName"] == observableName)
        ]

        # remove isExist from global DataFrame
        self.__data = self.__data.append(isExist).drop_duplicates(keep=False)

    def registerOnlyNewObservable(self) -> Tuple[int, Observable]:
        self.reqId += 1

        # -------------------------------------------------
        # ????? change type is ok
        obs = BehaviorSubject(None)
        # -------------------------------------------------

        self.__observablesInstances[self.reqId] = obs
        return (self.reqId, obs)

    def observableForContract(
        self, contract: IBContract, observableName: str
    ) -> Tuple[bool, int, Observable]:
        self.lock.acquire()
        try:
            # log.debug("LOCK - Acquired")

            (isExist, reqId, obs) = self.getObservableForContract(
                contract, observableName
            )

            if isExist == True:
                return (True, reqId, obs)
            else:
                # NEW observable
                (reqId, newObs) = self.registerOnlyNewObservable()

                # save information about observable and contract
                a_row: pd.Series = pd.Series(
                    [
                        contract.symbol,
                        contract.localSymbol,
                        reqId,
                        observableName,
                    ],
                    index=dfColumns,
                )
                row_df: pd.DataFrame = pd.DataFrame([a_row], columns=dfColumns)

                self.__data = self.__data.append(row_df, ignore_index=True)
                # log.debug(self.__data)

                return (False, reqId, newObs)
        finally:
            # log.debug("LOCK - released")
            self.lock.release()

    def getObservableForContract(
        self, contract: IBContract, observableName: str
    ) -> Tuple[bool, int, Observable]:
        isExist: pd.DataFrame = self.__data[
            (self.__data["Symbol"] == contract.symbol)
            & (self.__data["LocalSymbol"] == contract.localSymbol)
            & (self.__data["ObservableName"] == observableName)
        ]

        # IS ALREADY EXISTS ------------------------------------
        if isExist.shape[0] > 0:
            log.debug(
                "Observable already exist in IBClient ------------------------------------"
            )
            log.debug(isExist)
            reqId: int = isExist["ReqId"].item()
            obs: Observable = self.getObservable(reqId)
            return (True, reqId, obs)
        else:
            return (False, 0, None)

    def getObservableAndContract(
        self, reqId: int
    ) -> Tuple[Observable, str, str]:

        # observable
        obs: Observable = self.getObservable(reqId)

        # contract
        isExist: pd.DataFrame = self.__data[self.__data["ReqId"] == reqId]

        if isExist.shape[0] > 0:
            # print("IS ALREADY EXISTS - 2 ------------------------------------")
            # print(isExist.shape[0])
            # print(isExist)
            # print(isExist["Symbol"].item())
            # print(isExist["LocalSymbol"].item())
            return (
                obs,
                str(isExist["Symbol"].item()),
                str(isExist["LocalSymbol"].item()),
            )

        else:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("ERROR - OBSERVABLE DOESN'T EXIST - 2")
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            log.debug(reqId)
            log.debug(obs)
            log.debug(self.__data)
            return (None, "", "")

    # -----------------------------------------------------------------
    # TEMP-DATA
    # -----------------------------------------------------------------

    def registerTempData(self, reqId: int, data: List[Any]):
        self.__tempData[reqId] = data

    def addToTempData(self, reqId: int, data: Any):
        self.__tempData[reqId].append(data)

    def getTempData(self, reqId: int) -> List[Any]:
        return self.__tempData[reqId]

    def removeTempData(self, reqId: int):
        self.__tempData.pop(reqId)
