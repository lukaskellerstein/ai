import logging
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from rx import Observable, merge
from rx import operators as ops

from typing import Any, Tuple, Dict

from business.model.contracts import IBOptionContract
from ui.windows.main.pages.watchlists.options.options_service import (
    OptionsWatchlistService,
)
from ui.windows.main.pages.watchlists.options.table.tree import OptionsTree
from ui.state.main import State

from ui.base.base_page import BasePage

# create logger
log = logging.getLogger("CellarLogger")


class OptionsWatchlistPage(BasePage):

    detailWindow = None

    treeSignal = pyqtSignal(dict)

    subscriptions = []

    def __init__(self, *args: Tuple[str, Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        log.info("Running ...")

        # self.bl = OptionsWatchlistBL()

        self.state = State.getInstance()
        self.service = OptionsWatchlistService()

        # load template
        uic.loadUi(
            "ui/windows/main/pages/watchlists/options/options_page.ui", self
        )

        # load styles
        with open(
            "ui/windows/main/pages/watchlists/options/options_page.qss", "r"
        ) as fh:
            self.setStyleSheet(fh.read())

        # self.destroyed.connect(self.myDestroy)

        # Buttons
        self.startRealtime1Button.clicked.connect(self.startRealtime)
        # self.loadSavedLayoutButton.clicked.connect(self.loadTableLayout)

        # treeView
        self.tree = OptionsTree()
        # self.tree.on_remove.connect(self.remove)
        # self.table.on_open.connect(self.open)
        # self.table.on_order_changed.connect(self.update_watchlist)
        self.tableBox1.addWidget(self.tree)

        # SIGNALS
        # self.treeSignal.connect(self.tree.tree_model.on_update_model)

        # self.loadTableLayout()

    def startRealtime(self):
        ticker = self.ticker1Input.text().upper()

        self.__startRealtime(ticker)

    # region "startRealtime" operators

    def __startRealtime(self, ticker: str):

        step1 = self.service.getOptionChain(ticker).pipe(
            ops.do_action(self.__updateTableModel),
            ops.do_action(lambda x: log.info(x)),
        )

        # price (from state)
        step2 = self.state.stocks_realtime_data.getOrCreate(
            ticker, ticker
        ).close.pipe(ops.do_action(lambda x: log.info(x)))

        # volatility (from state)
        step3 = self.state.stocks_realtime_data.getOrCreate(
            ticker, ticker
        ).optionImpliedVolatility.pipe(ops.do_action(lambda x: log.info(x)))

        # Rx - Combine Latest
        higherObs = (
            step1.pipe(ops.combine_latest(step2, step3))
            .pipe(
                ops.flat_map(self.__startRealtimeOption),
                # ops.do_action(lambda x: log.info(x)),
            )
            .subscribe(self.__subscribe)
        )

        self.subscriptions.append(higherObs)

    def __startRealtimeOption(self, data) -> Observable:

        (optionChain, lastPrice, lastImplVolatility) = data

        expirations = optionChain["expirations"]
        strikes = optionChain["strikes"]

        resultListObs = []

        count = 0
        timeout = 0

        for expiration in expirations:
            for strike in strikes:
                if count >= 50:
                    count = 0
                    timeout = 1  # sec
                else:
                    timeout = 0

                count += 1

                optionContract = IBOptionContract(symbol="AAPL")
                optionContract.exchange = optionChain["exchange"]
                optionContract.lastTradeDateOrContractMonth = expiration
                optionContract.strike = float(strike)
                optionContract.right = "C"  # CALL OPTION
                optionContract.multiplier = 100

                temp = self.service.getOptionPrice(
                    optionContract,
                    float(lastImplVolatility),
                    float(lastPrice),
                    timeout,
                )

                resultListObs.append(temp)

        return merge(*resultListObs)

        # (exchange, expirations, strikes) = x

        # # resultList = []

        # exp1 = list(expirations)[0]
        # strike1 = list(strikes)[0]

        # print(list(expirations))
        # print(exp1)
        # print(list(strikes))
        # print(strike1)

        # # for strikePrice in strikes:

        # #     cont1 = copy.deepcopy(contract)
        # #     cont1.
        # #     cont1.strike = strikePrice

        # # return merge(*resultList)

        # optionContract = LlOption(contract.symbol)
        # optionContract.exchange = contract.exchange
        # optionContract.lastTradeDateOrContractMonth = "20200501"
        # optionContract.strike = 310.0
        # optionContract.right = "C"  # CALL OPTION
        # optionContract.multiplier = 100

        # print(optionContract)

        # return self.bl.ibClient.testOption(optionContract)

    def __updateTableModel(self, x):
        expirations = x["expirations"]
        strikes = x["strikes"]
        self.tree.tree_model.setStructure(expirations, strikes)

    def __subscribe(self, data):
        log.info("Running ....")
        log.info(data)

    # def __updateTableModel(self, x):
    #     # print("---------------------__updateTableModel-------------------")
    #     # print(x)
    #     # print("----------------------------------------------------------")

    #     # self.tree.tree_model.addGroup(x)

    #     return x

    # def __addToWatchlist(self, x, contract):
    #     # self.bl.addToWatchlist(contract)

    #     # self.bl.startRealtime(contract, self.resendRealtime)

    #     log.debug(x)

    # endregion

    def resendRealtime(self, data):
        i = 5
        # print(data)
        # self.treeSignal.emit(data)

    # def remove(self, data):
    #     ticker = data._data.index.values[0][0]

    #     self.tree.tree_model.removeFuture(ticker)
    #     # BL
    #     self.bl.remove(ticker)

    # def open(self, data):
    #     self.detailWindow = StocksDetailPage(data)
    #     self.detailWindow.show()

    # def update_watchlist(self, data):
    #     self.bl.updateStockWatchlist(data)

    # --------------------------------------------------------
    # --------------------------------------------------------
    # DESTROY
    # --------------------------------------------------------
    # --------------------------------------------------------

    # 1. CUSTOM destroy -----------------------------------------
    def onDestroy(self):
        log.info("Destroying ...")

        # Unsubscribe everything
        for sub in self.subscriptions:
            sub.dispose()

        # destroy BL
        self.service.onDestroy()

    # 2. Python destroy -----------------------------------------
    def __del__(self):
        log.info("Running ...")
