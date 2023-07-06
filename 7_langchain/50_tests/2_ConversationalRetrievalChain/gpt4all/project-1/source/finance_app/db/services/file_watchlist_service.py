import logging
import threading
from datetime import datetime
from typing import Any, List, Tuple
from business.model.timeframe import TimeFrame
import pandas as pd
import pystore
import time
from business.model.asset import AssetType
from utils import files
import sys
import os

# create logger
log = logging.getLogger("CellarLogger")


class FileWatchlistService(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.basePath = files.get_full_path("finance_app/db/watchlists")

    def addSymbol(self, watchlistName: str, symbol: str):
        self.__appendSymbols(watchlistName, [symbol])

    def removeSymbol(self, watchlistName: str, symbol: str):
        try:
            current = self.getWatchlist(watchlistName)
            future = list(filter(lambda x: x != symbol, current))
            self.updateWatchlist(watchlistName, future)
        except:
            log.error("Unexpected error:", sys.exc_info()[0])
            raise

    def updateWatchlist(self, watchlistName: str, symbols: List[str]):
        try:
            self.removeWatchlist(watchlistName)
            self.__writeSymbols(watchlistName, symbols)
        except:
            log.error("Unexpected error:", sys.exc_info()[0])
            raise

    def getWatchlist(self, watchlistName: str) -> List[str]:

        if not os.path.exists(f"{self.basePath}/{watchlistName}.txt"):
            return []

        start = time.time()
        file = None
        result: List[str] = []

        self.lock.acquire()
        try:
            file = open(f"{self.basePath}/{watchlistName}.txt", "r")
            result = [line.rstrip("\n") for line in file.readlines()]
        except:
            log.error("Unexpected error:", sys.exc_info()[0])
            raise
        finally:
            file.close()
            self.lock.release()

        end = time.time()
        log.info(f"takes {end - start} sec.")

        return result

    def removeWatchlist(self, watchlistName: str):
        if os.path.exists(f"{self.basePath}/{watchlistName}.txt"):
            os.remove(f"{self.basePath}/{watchlistName}.txt")
        else:
            log.error("The watchlist file does not exist !!")

    def __writeSymbols(self, watchlistName: str, symbols: List[str]):
        self.__writeSymbolsToFile(watchlistName, symbols, "w+")

    def __appendSymbols(self, watchlistName: str, symbols: List[str]):
        self.__writeSymbolsToFile(watchlistName, symbols, "a")

    def __writeSymbolsToFile(
        self, watchlistName: str, symbols: List[str], mode: str
    ):
        start = time.time()
        file = None
        self.lock.acquire()
        try:
            file = open(f"{self.basePath}/{watchlistName}.txt", mode)
            lines = list(map(lambda x: x + "\n", symbols))
            file.writelines(lines)
        except:
            log.error("Unexpected error:", sys.exc_info()[0])
            raise
        finally:
            file.close()
            self.lock.release()

        end = time.time()
        log.info(f"takes {end - start} sec.")
