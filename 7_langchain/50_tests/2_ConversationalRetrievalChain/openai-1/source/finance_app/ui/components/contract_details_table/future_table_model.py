import logging
from typing import List, Tuple

import pandas as pd

from business.model.contract_details import IBContractDetails
from ui.components.contract_details_table.base_table_model import (
    BaseContractDetailsTableModel,
)

# create logger
log = logging.getLogger("CellarLogger")


def defaultValue(data: List[Tuple]):
    df = pd.DataFrame(
        data=data,
        columns=[
            "secType",
            "longName",
            "symbol",
            "localSymbol",
            "contractMonth",
            "lastDate",
            "exchange",
            "minTick",
            "mdSizeMultiplier",
        ],
    )
    return df


class FutureContractDetailsTableModel(BaseContractDetailsTableModel):
    def __init__(self, data: List[IBContractDetails]):
        super().__init__()
        self.setData(data)

    # ----------------------------------------------------------
    # Custom methods
    # ----------------------------------------------------------
    def setData(self, data: List[IBContractDetails]):
        self.beginResetModel()

        result = list(
            map(
                lambda x: (
                    x.contract.secType,
                    x.longName,
                    x.contract.symbol,
                    x.contract.localSymbol,
                    x.contractMonth,
                    x.contract.lastTradeDateOrContractMonth,
                    x.contract.exchange,
                    x.minTick,
                    x.mdSizeMultiplier,
                ),
                data,
            )
        )

        self._data = defaultValue(result)
        self.endResetModel()
