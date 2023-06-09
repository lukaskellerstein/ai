from ui.windows.asset_detail.stocks.pages.fundamental_data.fundamental_data import (
    FundamentalDataPage,
)
from business.model.asset import Asset
from ui.windows.asset_detail.shared.asset_detail_window import (
    AssetDetailWindow,
)


class StockDetailWindow(AssetDetailWindow):
    def __init__(self, asset: Asset):
        super().__init__(asset)

        # MenuBar actions
        self.actionFundamentals.triggered.connect(
            self.setCurrentPage(FundamentalDataPage, asset=self.asset)
        )
