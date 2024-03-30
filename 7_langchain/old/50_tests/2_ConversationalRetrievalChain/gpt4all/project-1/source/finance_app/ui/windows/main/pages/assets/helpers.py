import logging
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from business.helpers import getTimeBlocks
from business.model.asset import Asset
from business.model.timeframe import TimeFrame
from business.modules.asset_bl import AssetBL

# create logger
log = logging.getLogger("CellarLogger")


def updateFutures(
    asset: Asset, timeframe: TimeFrame, bl: AssetBL
) -> List[Dict]:
    result = []
    for cd in asset.contractDetails:

        lastTradeDateTime = datetime.strptime(
            cd.contract.lastTradeDateOrContractMonth, "%Y%m%d"
        )
        now = datetime.now()

        if lastTradeDateTime > now:

            # localSymbol = cd.contract.localSymbol
            localSymbol = f"{cd.contract.localSymbol}-{cd.contract.lastTradeDateOrContractMonth}"
            symbolData = bl.getHistoricalDataFromDB(localSymbol, timeframe)

            if symbolData is not None:

                lastDateTime = symbolData.tail(1).index[0]

                if now > lastDateTime:
                    result.append(
                        {
                            "contract": cd.contract,
                            "from": lastDateTime,
                            "to": now,
                        }
                    )
            else:
                result.append(
                    {
                        "contract": cd.contract,
                        "from": datetime.strptime("19860101", "%Y%m%d"),
                        "to": now,
                    }
                )

        else:
            log.info(
                f" SKIPPED - {cd.contract.localSymbol} - {cd.contract.lastTradeDateOrContractMonth}"
            )

    return result


def downloadFutures(asset: Asset, blockSize: int) -> List[Dict]:
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
                    "from": lastTradeDateTime - timedelta(days=blockSize),
                    "to": lastTradeDateTime,
                }
            )
        elif lastTradeDateTime >= now:
            result.append(
                {
                    "contract": cd.contract,
                    "from": now - timedelta(days=blockSize),
                    "to": now,
                }
            )

    return result


def updateStock(
    asset: Asset, timeframe: TimeFrame, histData: pd.DataFrame
) -> List[Dict]:
    result = []

    contract = asset.contractDetails[0].contract
    # symbolData = bl.getHistoricalDataFromDB(asset.symbol, timeframe)

    if histData is not None:

        lastDateTime = histData.tail(1).index[0]
        now = datetime.now()

        if now > lastDateTime:
            result.append(
                {"contract": contract, "from": lastDateTime, "to": now}
            )

    return result


def downloadStock(asset: Asset, blockSize: int) -> List[Dict]:
    result = []

    contract = asset.contractDetails[0].contract

    timeBlocks = getTimeBlocks(
        datetime.strptime("19860101", "%Y%m%d"), datetime.now(), blockSize
    )

    for timeBlock in timeBlocks:
        result.append(
            {"contract": contract, "from": timeBlock[0], "to": timeBlock[1],}
        )

    return result
