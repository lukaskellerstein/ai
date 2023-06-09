
from business.model.contracts import IBContract
from typing import Any

class LlContractDetails(object):
    def __init__(self, contract: IBContract):
        # contract ---
        self.conId: int = contract.conId
        self.symbol: str = contract.symbol
        self.secType: str = contract.secType
        self.lastTradeDateOrContractMonth = contract.lastTradeDateOrContractMonth
        self.strike = contract.strike  # float !!
        self.right = contract.right
        self.multiplier = contract.multiplier
        self.exchange = contract.exchange
        self.primaryExchange = contract.primaryExchange # pick an actual (ie non-aggregate) exchange that the contract trades on.  DO NOT SET TO SMART.
        self.currency = contract.currency
        self.localSymbol = contract.localSymbol
        self.tradingClass = contract.tradingClass
        self.includeExpired = contract.includeExpired
        self.secIdType = contract.secIdType	  # CUSIP;SEDOL;ISIN;RIC
        self.secId = contract.secId

        #combos
        self.comboLegsDescrip = contract.comboLegsDescrip  
        # type: str; received in open order 14 and up for all combos
        self.comboLegs: Any = contract.comboLegs     
        # type: list<ComboLeg>
        self.deltaNeutralContract = contract.deltaNeutralContract
         # -----------

        self.marketName = ""
        self.minTick = 0.
        self.orderTypes = ""
        self.validExchanges = ""
        self.priceMagnifier = 0
        self.underConId = 0
        self.longName = ""
        self.contractMonth = ""
        self.industry = ""
        self.category = ""
        self.subcategory = ""
        self.timeZoneId = ""
        self.tradingHours = ""
        self.liquidHours = ""
        self.evRule = ""
        self.evMultiplier = 0
        self.mdSizeMultiplier = 0
        self.aggGroup = 0
        self.underSymbol = ""
        self.underSecType = ""
        self.marketRuleIds = ""
        self.secIdList = None
        self.realExpirationDate = ""
        self.lastTradeTime = ""
        # BOND values
        self.cusip = ""
        self.ratings = ""
        self.descAppend = ""
        self.bondType = ""
        self.couponType = ""
        self.callable = False
        self.putable = False
        self.coupon = 0
        self.convertible = False
        self.maturity = ""
        self.issueDate = ""
        self.nextOptionDate = ""
        self.nextOptionType = ""
        self.nextOptionPartial = False
        self.notes = ""

    # def __iter__(self):
    #     return vars(self).iteritems()

    # def __str__(self):
    #     s = ",".join((
    #         str(self.contract),
    #         str(self.marketName),
    #         str(self.minTick),
    #         str(self.orderTypes),
    #         str(self.validExchanges),
    #         str(self.priceMagnifier),
    #         str(self.underConId),
    #         str(self.longName),
    #         str(self.contractMonth),
    #         str(self.industry),
    #         str(self.category),
    #         str(self.subcategory),
    #         str(self.timeZoneId),
    #         str(self.tradingHours),
    #         str(self.liquidHours),
    #         str(self.evRule),
    #         str(self.evMultiplier),
    #         str(self.mdSizeMultiplier),
    #         str(self.underSymbol),
    #         str(self.underSecType),
    #         str(self.marketRuleIds),
    #         str(self.aggGroup),
    #         str(self.secIdList),
    #         str(self.realExpirationDate),
    #         str(self.cusip),
    #         str(self.ratings),
    #         str(self.descAppend),
    #         str(self.bondType),
    #         str(self.couponType),
    #         str(self.callable),
    #         str(self.putable),
    #         str(self.coupon),
    #         str(self.convertible),
    #         str(self.maturity),
    #         str(self.issueDate),
    #         str(self.nextOptionDate),
    #         str(self.nextOptionType),
    #         str(self.nextOptionPartial),
    #         str(self.notes)),
    #         str(self.conId),
    #         str(self.symbol),
    #         str(self.secType),
    #         str(self.lastTradeDateOrContractMonth),
    #         str(self.strike),
    #         str(self.right),
    #         str(self.multiplier),
    #         str(self.exchange),
    #         str(self.primaryExchange),
    #         str(self.currency),
    #         str(self.localSymbol),
    #         str(self.tradingClass),
    #         str(self.includeExpired),
    #         str(self.secIdType),
    #         str(self.secId))

    #     s += "combo:" + self.comboLegsDescrip

    #     if self.comboLegs:
    #         for leg in self.comboLegs:
    #             s += ";" + str(leg)

    #     if self.deltaNeutralContract:
    #         s += ";" + str(self.deltaNeutralContract)

    #     return s
