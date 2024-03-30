import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Union
from enum import Enum
import pandas as pd
from rx import operators as ops
from rx.core.typing import Observable, Observer
from rx.subject import BehaviorSubject

# create logger
log = logging.getLogger("CellarLogger")


class RealtimeDataItemStatus(Enum):
    NEW = "new"
    RUNNING = "runing"
    STOPED = "stoped"


class RealtimeDataItem(object):
    def __init__(self, symbol, localSymbol):

        self.symbol = symbol
        self.localSymbol = localSymbol

        self.status = RealtimeDataItemStatus.NEW

        # ----------------------------------
        # main observable
        # ----------------------------------
        self.ticks = BehaviorSubject({})

        # ----------------------------------
        # derived observables
        # ----------------------------------
        self.bid = BehaviorSubject(0.0)
        self.bidSize = BehaviorSubject(0)
        self.bidExch = BehaviorSubject(
            ""
        )  # Example: Z, KQ, NKTZ, BJQZ ...etc.

        self.last = BehaviorSubject(0.0)
        self.lastSize = BehaviorSubject(0)
        self.lastExch = BehaviorSubject(
            ""
        )  # Example: Z, KQ, NKTZ, BJQZ ...etc.
        self.lastTimestamp = BehaviorSubject(0)  # Example: 1587648950

        self.ask = BehaviorSubject(0.0)
        self.askSize = BehaviorSubject(0)
        self.askExch = BehaviorSubject(
            ""
        )  # Example: Z, KQ, NKTZ, BJQZ ...etc.

        self.open = BehaviorSubject(0.0)
        self.high = BehaviorSubject(0.0)
        self.low = BehaviorSubject(0.0)
        self.close = BehaviorSubject(0.0)

        self.volume = BehaviorSubject(0)

        self.optionHistoricalVolatility = BehaviorSubject(0.0)
        self.optionImpliedVolatility = BehaviorSubject(0.0)

        self.dividends = BehaviorSubject(
            ""
        )  # Example: 3.0284,3.1628,20200723,0.7907

    def start(self, observable: Observable[Any]) -> None:
        self.status = RealtimeDataItemStatus.NEW
        self.ticks = observable.pipe(ops.do_action(lambda x: self.route(x)),)

    def stop(self) -> None:
        self.status = RealtimeDataItemStatus.STOPED
        self.ticks = None

    def route(self, data: Dict[str, Any]) -> None:
        if data == {}:
            return

        self.status = RealtimeDataItemStatus.RUNNING

        if data["type"] == "ask":
            self.ask.on_next(data)
        elif data["type"] == "ask_size":
            self.askSize.on_next(data)
        elif data["type"] == "ask_exch":
            self.askExch.on_next(data)
        elif data["type"] == "last":
            self.last.on_next(data)
        elif data["type"] == "last_size":
            self.lastSize.on_next(data)
        elif data["type"] == "last_exch":
            self.lastExch.on_next(data)
        elif data["type"] == "last_timestamp":
            self.lastTimestamp.on_next(data)
        elif data["type"] == "bid":
            self.bid.on_next(data)
        elif data["type"] == "bid_size":
            self.bidSize.on_next(data)
        elif data["type"] == "bid_exch":
            self.bidExch.on_next(data)
        elif data["type"] == "open":
            self.open.on_next(data)
        elif data["type"] == "high":
            self.high.on_next(data)
        elif data["type"] == "low":
            self.low.on_next(data)
        elif data["type"] == "close":
            self.close.on_next(data)
        elif data["type"] == "volume":
            self.volume.on_next(data)
        elif data["type"] == "option_historical_vol":
            self.optionHistoricalVolatility.on_next(data)
        elif data["type"] == "option_implied_vol":
            self.optionImpliedVolatility.on_next(data)
        elif data["type"] == "ib_dividends":
            self.dividends.on_next(data)

    def log(self):
        log.info(f"{self.symbol}-{self.localSymbol}")
        # log.info(self.ticks.is_disposed)
        # log.info(self.ticks.is_stopped)
        # for obs in self.ticks.observers:
        #     log.info(obs.__slots__)


class RealtimeDataState(object):
    __data: DefaultDict[str, RealtimeDataItem]

    def __init__(self):
        self.__data = defaultdict(None)

    def __add(self, symbol: str, localSymbol: str) -> RealtimeDataItem:
        x = RealtimeDataItem(symbol, localSymbol)

        self.__data[self.__constructKey(symbol, localSymbol)] = x

        return x

    def getState(self) -> DefaultDict[str, RealtimeDataItem]:
        return self.__data

    def get(
        self, symbol: str, localSymbol: str
    ) -> Union[None, RealtimeDataItem]:
        return self.__data.get(self.__constructKey(symbol, localSymbol))

    def create(self, symbol: str, localSymbol: str) -> RealtimeDataItem:
        x = self.get(symbol, localSymbol)
        if x is None:
            return self.__add(symbol, localSymbol)
        else:
            log.info(
                "Observable already exist in State ------------------------------------"
            )
            return x

    def stop(self, symbol: str, localSymbol: str) -> None:
        rdi = self.get(symbol, localSymbol)
        rdi.stop()

        self.__data.pop(self.__constructKey(symbol, localSymbol))

    def stopAll(self) -> None:
        for key, rdi in self.__data.items():
            log.info(f"Stopping: {key}")
            rdi.stop()

        # reset dictionary
        self.__data = defaultdict(None)

    # ------------------------------------
    # HELEPRS
    # ------------------------------------
    def __constructKey(self, symbol, localSymbol) -> str:
        return f"{symbol}|{localSymbol}"
