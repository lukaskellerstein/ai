from enum import Enum


class TimeFrame(Enum):
    day1 = "1 day"
    minute1 = "1 min"


class Duration(Enum):
    all = "All"
    years20 = "20 years"
    years10 = "10 years"
    year5 = "5 years"
    year1 = "1 year"
    quarter1 = "1 quarter"
    month1 = "1 month"
    week1 = "1 week"
