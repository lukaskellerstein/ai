import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from typing import Dict, List, Any


def plot_time_series(
    data: Dict[str, pd.DataFrame], title: str, xlabel: str, ylabel: str
) -> Figure:
    # Plotting data
    figure = plt.figure(figsize=(10, 5))

    for key, value in data.items():
        df_temp = value.copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"])
        df_temp.set_index("date", inplace=True)

        plt.plot(df_temp, label=key)

    plt.gcf().autofmt_xdate()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    return figure


def plot(data: Dict[str, Any], title: str, xlabel: str, ylabel: str) -> Figure:
    # Plotting data
    figure = plt.figure(figsize=(10, 5))

    for key, value in data.items():
        plt.plot(value, label=key)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    return figure


def plot_data2(train, valid, test) -> Figure:
    df1 = pd.DataFrame(train, columns=["date", "close"])
    df1["date"] = pd.to_datetime(df1["date"])
    df1.set_index("date", inplace=True)

    df2 = pd.DataFrame(valid, columns=["date", "close"])
    df2["date"] = pd.to_datetime(df2["date"])
    df2.set_index("date", inplace=True)

    df3 = pd.DataFrame(test, columns=["date", "close"])
    df3["date"] = pd.to_datetime(df3["date"])
    df3.set_index("date", inplace=True)

    # Plotting data
    figure = plt.figure(figsize=(10, 5))
    plt.plot(df1)
    plt.plot(df2)
    plt.plot(df3)
    plt.gcf().autofmt_xdate()
    return figure


def plot_data(train) -> Figure:
    df1 = pd.DataFrame(train, columns=["date", "close"])
    df1["date"] = pd.to_datetime(df1["date"])
    df1.set_index("date", inplace=True)

    # Plotting data
    figure = plt.figure(figsize=(10, 5))
    plt.plot(df1)
    plt.gcf().autofmt_xdate()
    return figure


def plot_Predicted_vs_Actual(predictions, actuals, group, title, writer):
    figure = plt.figure(figsize=(10, 5))
    plt.plot(predictions, label="Predicted values")
    plt.plot(actuals, label="Actual values")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    writer.add_figure(group + "/" + title, figure)


def plot_Difference_Predicted_vs_Actual(predictions, actuals, group, title, writer):
    figure = plt.figure(figsize=(10, 5))
    plt.plot(predictions - actuals, label="Difference between Predicted and Actual")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Difference")
    plt.legend()
    plt.grid(True)
    writer.add_figure(group + "/" + title, figure)
