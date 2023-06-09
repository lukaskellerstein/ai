from business.model.contract_details import IBContractDetails
import logging
from typing import Dict, Union, Any

from ibapi.contract import ContractDetails

from business.model.contract_details import IBContractDetails
from business.model.factory.contract_factory import ContractFactory

# create logger
log = logging.getLogger("CellarLogger")


class ContractDetailsFactory(object):
    def __init__(self):
        log.debug("Running ...")
        self.contractFactory = ContractFactory()

    def createIBContractDetails(
        self, contractDetails: Union[ContractDetails, Dict[str, Any]]
    ) -> IBContractDetails:
        if type(contractDetails) is ContractDetails:
            return self._createIBContractDetails_fromContractDetails(
                contractDetails
            )
        elif type(contractDetails) is dict:
            return self._createIBContractDetails_fromDict(contractDetails)
        else:
            log.info(type(contractDetails))
            raise Exception("THIS SHOULD NOT HAPPENED")

    def _createIBContractDetails_fromContractDetails(
        self, contractDetails: ContractDetails,
    ) -> IBContractDetails:

        resultContractDetail: IBContractDetails = IBContractDetails()

        resultContractDetail.contract = contractDetails.contract
        resultContractDetail.marketName = contractDetails.marketName
        resultContractDetail.minTick = contractDetails.minTick
        resultContractDetail.orderTypes = contractDetails.orderTypes
        resultContractDetail.validExchanges = contractDetails.validExchanges
        resultContractDetail.priceMagnifier = contractDetails.priceMagnifier
        resultContractDetail.underConId = contractDetails.underConId
        resultContractDetail.longName = contractDetails.longName
        resultContractDetail.contractMonth = contractDetails.contractMonth
        resultContractDetail.industry = contractDetails.industry
        resultContractDetail.category = contractDetails.category
        resultContractDetail.subcategory = contractDetails.subcategory
        resultContractDetail.timeZoneId = contractDetails.timeZoneId
        resultContractDetail.tradingHours = contractDetails.tradingHours
        resultContractDetail.liquidHours = contractDetails.liquidHours
        resultContractDetail.evRule = contractDetails.evRule
        resultContractDetail.evMultiplier = contractDetails.evMultiplier
        resultContractDetail.mdSizeMultiplier = (
            contractDetails.mdSizeMultiplier
        )
        resultContractDetail.aggGroup = contractDetails.aggGroup
        resultContractDetail.underSymbol = contractDetails.underSymbol
        resultContractDetail.underSecType = contractDetails.underSecType
        resultContractDetail.marketRuleIds = contractDetails.marketRuleIds
        # CAUSING ERRORS
        # resultContractDetail.secIdList = contractDetails.secIdList
        resultContractDetail.realExpirationDate = (
            contractDetails.realExpirationDate
        )
        resultContractDetail.lastTradeTime = contractDetails.lastTradeTime
        # BOND values
        resultContractDetail.cusip = contractDetails.cusip
        resultContractDetail.ratings = contractDetails.ratings
        resultContractDetail.descAppend = contractDetails.descAppend
        resultContractDetail.bondType = contractDetails.bondType
        resultContractDetail.couponType = contractDetails.couponType
        resultContractDetail.callable = contractDetails.callable
        resultContractDetail.putable = contractDetails.putable
        resultContractDetail.coupon = contractDetails.coupon
        resultContractDetail.convertible = contractDetails.convertible
        resultContractDetail.maturity = contractDetails.maturity
        resultContractDetail.issueDate = contractDetails.issueDate
        resultContractDetail.nextOptionDate = contractDetails.nextOptionDate
        resultContractDetail.nextOptionType = contractDetails.nextOptionType
        resultContractDetail.nextOptionPartial = (
            contractDetails.nextOptionPartial
        )
        resultContractDetail.notes = contractDetails.notes

        return resultContractDetail

    def _createIBContractDetails_fromDict(
        self, contractDetails: Dict[str, Any]
    ) -> IBContractDetails:

        result: IBContractDetails = IBContractDetails()

        # CONTRACT
        c = contractDetails["contract"]
        cc = self.contractFactory.createIBContract(c)
        result.contract = cc

        # Contract details
        result.marketName = contractDetails["marketName"]
        result.minTick = float(contractDetails["minTick"])
        result.orderTypes = contractDetails["orderTypes"]
        result.validExchanges = contractDetails["validExchanges"]
        result.priceMagnifier = int(contractDetails["priceMagnifier"])
        result.underConId = int(contractDetails["underConId"])
        result.longName = contractDetails["longName"]
        result.contractMonth = contractDetails["contractMonth"]
        result.industry = contractDetails["industry"]
        result.category = contractDetails["category"]
        result.subcategory = contractDetails["subcategory"]
        result.timeZoneId = contractDetails["timeZoneId"]
        result.tradingHours = contractDetails["tradingHours"]
        result.liquidHours = contractDetails["liquidHours"]
        result.evRule = contractDetails["evRule"]
        result.evMultiplier = int(contractDetails["evMultiplier"])
        result.mdSizeMultiplier = int(contractDetails["mdSizeMultiplier"])
        result.aggGroup = int(contractDetails["aggGroup"])
        result.underSymbol = contractDetails["underSymbol"]
        result.underSecType = contractDetails["underSecType"]
        result.marketRuleIds = contractDetails["marketRuleIds"]
        # CAUSING ERRORS
        # result.secIdList = contractDetails["secIdList"]
        result.realExpirationDate = contractDetails["realExpirationDate"]
        result.lastTradeTime = contractDetails["lastTradeTime"]
        # BOND values
        result.cusip = contractDetails["cusip"]
        result.ratings = contractDetails["ratings"]
        result.descAppend = contractDetails["descAppend"]
        result.bondType = contractDetails["bondType"]
        result.couponType = contractDetails["couponType"]
        result.callable = bool(contractDetails["callable"])
        result.putable = bool(contractDetails["putable"])
        result.coupon = contractDetails[
            "coupon"
        ]  # for a reason i am not doing retyping, waiting if it is necessary
        result.convertible = contractDetails[
            "convertible"
        ]  # for a reason i am not doing retyping, waiting if it is necessary
        result.maturity = contractDetails["maturity"]
        result.issueDate = contractDetails["issueDate"]
        result.nextOptionDate = contractDetails["nextOptionDate"]
        result.nextOptionType = contractDetails["nextOptionType"]
        result.nextOptionPartial = contractDetails[
            "nextOptionPartial"
        ]  # for a reason i am not doing retyping, waiting if it is necessary
        result.notes = contractDetails["notes"]

        return result

    def createDict(self, contractDetails: ContractDetails) -> Dict[str, str]:
        resultContractDetail: Dict[str, Any] = {}

        # CONTRACT
        contract: Dict[str, Any] = self.contractFactory.createDict(
            contractDetails.contract
        )

        # Contract details
        resultContractDetail["contract"] = contract
        resultContractDetail["marketName"] = contractDetails.marketName
        resultContractDetail["minTick"] = contractDetails.minTick
        resultContractDetail["orderTypes"] = contractDetails.orderTypes
        resultContractDetail["validExchanges"] = contractDetails.validExchanges
        resultContractDetail["priceMagnifier"] = contractDetails.priceMagnifier
        resultContractDetail["underConId"] = contractDetails.underConId
        resultContractDetail["longName"] = contractDetails.longName
        resultContractDetail["contractMonth"] = contractDetails.contractMonth
        resultContractDetail["industry"] = contractDetails.industry
        resultContractDetail["category"] = contractDetails.category
        resultContractDetail["subcategory"] = contractDetails.subcategory
        resultContractDetail["timeZoneId"] = contractDetails.timeZoneId
        resultContractDetail["tradingHours"] = contractDetails.tradingHours
        resultContractDetail["liquidHours"] = contractDetails.liquidHours
        resultContractDetail["evRule"] = contractDetails.evRule
        resultContractDetail["evMultiplier"] = contractDetails.evMultiplier
        resultContractDetail[
            "mdSizeMultiplier"
        ] = contractDetails.mdSizeMultiplier
        resultContractDetail["aggGroup"] = contractDetails.aggGroup
        resultContractDetail["underSymbol"] = contractDetails.underSymbol
        resultContractDetail["underSecType"] = contractDetails.underSecType
        resultContractDetail["marketRuleIds"] = contractDetails.marketRuleIds
        # CAUSING ERRORS
        # resultContractDetail["secIdList"] = contractDetails.secIdList
        resultContractDetail[
            "realExpirationDate"
        ] = contractDetails.realExpirationDate
        resultContractDetail["lastTradeTime"] = contractDetails.lastTradeTime
        # BOND values
        resultContractDetail["cusip"] = contractDetails.cusip
        resultContractDetail["ratings"] = contractDetails.ratings
        resultContractDetail["descAppend"] = contractDetails.descAppend
        resultContractDetail["bondType"] = contractDetails.bondType
        resultContractDetail["couponType"] = contractDetails.couponType
        resultContractDetail["callable"] = contractDetails.callable
        resultContractDetail["putable"] = contractDetails.putable
        resultContractDetail["coupon"] = contractDetails.coupon
        resultContractDetail["convertible"] = contractDetails.convertible
        resultContractDetail["maturity"] = contractDetails.maturity
        resultContractDetail["issueDate"] = contractDetails.issueDate
        resultContractDetail["nextOptionDate"] = contractDetails.nextOptionDate
        resultContractDetail["nextOptionType"] = contractDetails.nextOptionType
        resultContractDetail[
            "nextOptionPartial"
        ] = contractDetails.nextOptionPartial
        resultContractDetail["notes"] = contractDetails.notes

        return resultContractDetail
