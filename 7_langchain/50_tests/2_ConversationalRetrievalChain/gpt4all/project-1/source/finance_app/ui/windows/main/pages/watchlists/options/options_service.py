import logging
import random
import threading
from typing import List

import pandas as pd
from rx import of
from rx import operators as ops
from rx.core.typing import Observable, Subject

from business.modules.options_watchlist_bl import OptionsWatchlistBL
from business.model.contracts import (
    IBOptionContract,
    IBStockContract,
    IBContract,
)
from ui.state.main import State

# create logger
log = logging.getLogger("CellarLogger")


class OptionsWatchlistService(object):
    """ Service integrates BL and State management
    """

    def __init__(self):
        log.info("Running ...")

        # State
        self.state = State.getInstance()

        # BL
        self.bl = OptionsWatchlistBL()

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ACTIONS (setting state)
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    # def startRealtimeAction(self, ticker) -> StocksRealtimeDataItem:
    #     print(ticker)

    #     stateItem = self.state.stocks_realtime_data.get(ticker, ticker)

    #     stateItem.ticks = self.bl.getContractDetails(LlStock(ticker)).pipe(
    #         # ops.do_action(lambda x: log.info(x)),
    #         ops.flat_map(
    #             lambda x: self.bl.startRealtime(x.contract).pipe(
    #                 ops.filter(lambda x: x is not None),
    #             )
    #         ),
    #     )

    #     stateItem.ask = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "ask")
    #     )
    #     stateItem.askSize = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "ask_size")
    #     )
    #     stateItem.askExch = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "ask_exch")
    #     )

    #     stateItem.last = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "last")
    #     )
    #     stateItem.lastSize = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "last_size")
    #     )
    #     stateItem.lastExch = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "last_exch")
    #     )
    #     stateItem.lastTimestamp = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "last_timestamp")
    #     )

    #     stateItem.bid = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "bid")
    #     )
    #     stateItem.bidSize = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "bid_size")
    #     )
    #     stateItem.bidExch = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "bid_exch")
    #     )

    #     stateItem.open = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "open")
    #     )
    #     stateItem.high = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "high")
    #     )
    #     stateItem.low = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "low")
    #     )
    #     stateItem.close = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "close")
    #     )

    #     stateItem.volume = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "volume")
    #     )

    #     stateItem.optionHistoricalVolatility = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "option_historical_vol")
    #     )
    #     stateItem.optionImpliedVolatility = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "option_implied_vol")
    #     )

    #     stateItem.dividends = stateItem.ticks.pipe(
    #         ops.filter(lambda x: x["type"] == "ib_dividends")
    #     )

    #     return stateItem

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ACTIONS (getting state)
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    def getPriceAction(self, ticker: str) -> Observable[float]:
        return self.state.stocks_realtime_data.getOrCreate(ticker, ticker).last

    def getImpliedVolatilityAction(self, ticker: str) -> Observable[float]:
        return self.state.stocks_realtime_data.getOrCreate(
            ticker, ticker
        ).optionImpliedVolatility

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # BUSINESS LOGIC
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    def getOptionPrice(
        self,
        contract: IBOptionContract,
        volatility: float,
        underPrice: float,
        timeout: int,
    ) -> Observable:
        return self.bl.getOptionPrice(
            contract, volatility, underPrice, timeout
        ).pipe(ops.filter(lambda x: x is not None),)

    def getOptionPrice2(self):
        pass

    def getOptionChain(self, ticker: str) -> Observable[dict]:
        return self.bl.getOptionChain(ticker)

    # def getWatchlist(self) -> pd.DataFrame:
    #     return self.bl.getWatchlist()

    # def remove(self, ticker: str):
    #     self.bl.remove(ticker)

    # def updateStockWatchlist(self, arr: List[str]):
    #     self.bl.updateStockWatchlist(arr)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # destroy BL
        self.bl.onDestroy()

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
