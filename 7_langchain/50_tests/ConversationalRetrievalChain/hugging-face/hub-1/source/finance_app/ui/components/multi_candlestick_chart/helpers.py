import pandas as pd


def printOHLCInfo(df: pd.DataFrame) -> str:
    result = ""

    for i in range(df.shape[0]):

        row = df.iloc[i]

        if i == 0:
            result += f"<div style='color:#1a237e'>{row['LocalSymbol']} - O={row['Open']}, H={row['High']}, L={row['Low']}, C={row['Close']}, V={row['Volume']:.0f}</div>"
        elif i == 1:
            result += f"<div style='color:#5c6bc0'>{row['LocalSymbol']} - O={row['Open']}, H={row['High']}, L={row['Low']}, C={row['Close']}, V={row['Volume']:.0f}</div>"
        elif i == 2:
            result += f"<div style='color:#7986cb'>{row['LocalSymbol']} - O={row['Open']}, H={row['High']}, L={row['Low']}, C={row['Close']}, V={row['Volume']:.0f}</div>"
        else:
            result += f"<div style='color:#9fa8da'>{row['LocalSymbol']} - O={row['Open']}, H={row['High']}, L={row['Low']}, C={row['Close']}, V={row['Volume']:.0f}</div>"

    return result
