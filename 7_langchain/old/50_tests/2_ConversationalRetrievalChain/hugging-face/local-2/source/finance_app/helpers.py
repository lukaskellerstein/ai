import logging
import threading
from typing import List
from datetime import datetime

# create logger
log = logging.getLogger("CellarLogger")


# ------------------------------------
# HELEPRS
# ------------------------------------
def constructKey(symbol, localSymbol) -> str:
    return f"{symbol}|{localSymbol}"


def try_parsing_date(text, formats: List[str]) -> datetime:
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("no valid date format found")


def logThreads():
    log.info(f"Threads count: {threading.active_count()}")
    log.info("Thread names:")
    for thread in threading.enumerate():
        log.info(f"- {thread.getName()}")


def getColorByYieldValue(value: int):
    if value <= -25:
        return "#b71c1c"
    elif value > -25 and value <= -10:
        return "#d32f2f"
    elif value > -10 and value <= -6:
        return "#f44336"
    elif value > -6 and value <= -3:
        return "#e57373"
    elif value > -3 and value < 0:
        return "#ffcdd2"
    elif value == 0:
        return "white"
    elif value > 0 and value < 3:
        return "#c8e6c9"
    elif value >= 3 and value < 6:
        return "#81c784"
    elif value >= 6 and value < 10:
        return "#4caf50"
    elif value >= 10 and value < 25:
        return "#388e3c"
    elif value >= 25:
        return "#1b5e20"

