import logging
import threading
from datetime import datetime
from typing import Any, List, Tuple
from business.model.timeframe import TimeFrame
import pandas as pd
import pystore
import time

# create logger
log = logging.getLogger("CellarLogger")


class PyStoreHistService(object):
    def __init__(self):
        self.lock = threading.Lock()
        pystore.set_path("./db/pystore")
        self.store = pystore.store("cellarstone_db")

    def __getNewDf(
        self, data: List[Tuple[datetime, float, float, float, float, float]]
    ) -> pd.DataFrame:
        my_df = pd.DataFrame(
            data,
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        my_df["Datetime"] = pd.to_datetime(my_df["Datetime"])
        my_df.set_index(["Datetime"], inplace=True)
        return my_df

    def add(
        self,
        symbol: str,
        timeframe: TimeFrame,
        bars: List[Tuple[datetime, float, float, float, float, float]],
    ):
        if len(bars) > 0:
            self.lock.acquire()
            try:
                # log.info(bars)
                my_df = self.__getNewDf(bars)
                # log.info(my_df)

                t = timeframe.value.strip()

                collection = self.store.collection(t)
                if symbol in collection.list_items():
                    item = collection.item(symbol)
                    collection.append(
                        symbol, my_df, npartitions=item.data.npartitions
                    )
                else:
                    collection.write(
                        symbol,
                        my_df,
                        metadata={"source": "InteractiveBrokers"},
                    )
            finally:
                self.lock.release()

    def getAll(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        log.info(symbol)
        start = time.time()

        t = timeframe.value.strip()

        collection = self.store.collection(t)

        df = None
        if symbol in collection.list_items():
            item = collection.item(symbol)
            # data = item.data  # <-- Dask dataframe (see dask.pydata.org)
            # log.info(item)
            # log.info(data)
            # metadata = item.metadata
            # log.info(metadata)
            df = item.to_pandas()

        end = time.time()
        log.info(f"takes {end - start} sec.")
        return df

    def removeAll(self, symbol, timeframe: TimeFrame):
        t = timeframe.value.strip()
        collection = self.store.collection(t)
        collection.delete_item(symbol)
