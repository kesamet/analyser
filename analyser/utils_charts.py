"""
Script containing commonly used functions.
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

XLSX_FILE = "summary/prices.xlsx"


def download_data(symbol, start_date, end_date, time_interval="daily", dirname="data"):
    """Download data given ticker symbols."""
    try:
        from yahoofinancials import YahooFinancials as yf

        tick = yf(symbol)
        data = tick.get_historical_price_data(
            start_date=start_date, end_date=end_date, time_interval=time_interval)
        df = pd.DataFrame.from_dict(data[symbol]["prices"])
        df["date"] = pd.to_datetime(df["formatted_date"])
        df = df[["date", "adjclose", "close", "high", "low", "open", "volume"]]
        if dirname is not None:
            df.to_csv(os.path.join(dirname, f"{symbol}.csv"), index=False)
        else:
            return df
    except KeyError:
        print(f"... Data not found for {symbol}")


def get_data(symbols, dates, base_symbol="D05.SI", col="adjclose", dirname="data"):
    """Load stock data for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    df.index.name = "date"
    if base_symbol not in symbols:
        symbols = [base_symbol] + symbols
    if "ES3.SI" not in symbols:
        symbols.append("ES3.SI")

    for symbol in symbols:
        if symbol == "SB.SI":
            df[symbol] = 1000
        elif symbol == "UT.SI":
            df[symbol] = 1
        else:
            df_temp = pd.read_csv(
                os.path.join(dirname, f"{symbol}.csv"), index_col="date",
                parse_dates=True, na_values=["nan"], usecols=["date", col])
            df_temp = df_temp.rename(columns={col: symbol})
            df = df.join(df_temp)

        if symbol == base_symbol:  # drop dates index did not trade
            df = df.dropna(subset=[base_symbol])

    df = df.replace([0], [np.nan])
    fill_missing_values(df)
    return df


def get_data_ohlcv(symbol, dates, base_symbol="ES3.SI", dirname="data"):
    """Load stock ohlcv data for given symbol from CSV files."""
    df_base = pd.read_csv(
        os.path.join(dirname, f"{base_symbol}.csv"), index_col="date",
        parse_dates=True, na_values=["nan"], usecols=["date", "close"])
    df_base = df_base.rename(columns={"close": base_symbol})

    df = pd.DataFrame(index=dates)
    df.index.name = "date"
    df = df.join(df_base)
    df = df.dropna(subset=[base_symbol])

    df_temp = pd.read_csv(
        os.path.join(dirname, f"{symbol}.csv"), index_col="date",
        parse_dates=True, na_values=["nan"],
        usecols=["date", "open", "high", "low", "close", "volume"])

    df = df.join(df_temp)
    df = df.replace([0], [np.nan])
    df = df.drop_duplicates()
    fill_missing_values(df)
    return df


def get_data_xlsx(symbols, dates, base_symbol="USDSGD", col="Close", dirname="data"):
    """Load stock data for given symbols from xlsx file."""
    df = pd.DataFrame(index=dates)
    df.index.name = "Date"
    if base_symbol not in symbols:
        symbols = [base_symbol] + symbols

    for symbol in symbols:
        df_temp = pd.read_excel(
            os.path.join(dirname, XLSX_FILE), index_col="Date",
            parse_dates=True, sheet_name=symbol, usecols=["Date", col])
        df_temp.index = df_temp.index.date
        df_temp = df_temp.rename(columns={col: symbol})
        df = df.join(df_temp)
        if symbol == base_symbol:  # drop dates index did not trade
            df = df.dropna(subset=[base_symbol])
    df = df.replace([0], [np.nan])
    fill_missing_values(df)
    return df


def get_data_xlsx_ohlcv(symbol, dates, base_symbol="USDSGD", dirname="data"):
    """Load stock ohlcv data for given symbol from xlsx files."""
    df_base = pd.read_excel(
        os.path.join(dirname, XLSX_FILE), index_col="Date",
        parse_dates=True, sheet_name=symbol, usecols=["Date", "Close"])
    df_base.columns = [base_symbol]
    df_base.index = df_base.index.date
    df_base.index.name = "date"

    df = pd.DataFrame(index=dates)
    df.index.name = "date"
    df = df.join(df_base)
    df = df.dropna(subset=[base_symbol])

    df_temp = pd.read_excel(
        os.path.join(dirname, XLSX_FILE),
        parse_dates=True, sheet_name=symbol, index_col="Date",
        usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    df_temp.columns = ["open", "high", "low", "close", "volume"]
    df_temp.index = df_temp.index.date
    df_temp.index.name = "date"

    df = df.join(df_temp)
    df = df.replace([0], [np.nan])
    df = df.drop_duplicates()
    fill_missing_values(df)
    return df


def fill_missing_values(df):
    """Fill missing values in dataframe."""
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)


def get_ie_data(start_date="1871-01-01", dirname="data"):
    """Load Shiller data."""
    df = pd.read_excel(os.path.join(dirname, "summary/ie_data.xls"), sheet_name="Data", skiprows=7)
    df.drop(["Fraction", "Unnamed: 13", "Unnamed: 15"], axis=1, inplace=True)
    df.columns = ["Date", "S&P500", "Dividend", "Earnings", "CPI", "Long_IR",
                  "Real_Price", "Real_Dividend", "Real_TR_Price",
                  "Real_Earnings", "Real_TR_Scaled_Earnings", "CAPE", "TRCAPE",
                  "Excess_CAPE_Yield", "Mth_Bond_TR", "Bond_RTR",
                  "10Y_Stock_RR", "10Y_Bond_RR", "10Y_Excess_RR"]
    df["Date"] = pd.to_datetime(df["Date"].astype(str))
    df.set_index("Date", inplace=True)
    df = df.iloc[:-1]
    df = df[df.index >= start_date]
    return df


# Plot functions
def plot_data(df, title="", xlabel="Date", ylabel="Price", ax=None):
    """Plot stock prices."""
    ax = df.plot(title=title, fontsize=12, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_normalized_data(df, title="", xlabel="Date", ylabel="Normalized", ax=None):
    """Plot normalized stock prices."""
    normdf = rebase(df)
    ax = plot_data(normdf, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    ax.axhline(y=1, linestyle="--", color="k")
    return ax


def plot_bollinger(df, title=None, ax=None):
    """Plot bollinger bands and SMA."""
    df2 = df[["close"]].copy()
    _, df2["upper"], df2["lower"] = compute_bbands(df["close"])
    df2["sma200"] = compute_sma(df["close"], 200)
    df2["sma50"] = compute_sma(df["close"], 50)

    ax = plot_data(df2, title=title, ax=ax)
    return df2, ax


def plot_with_two_scales(df1, df2, xlabel="Date", ylabel1="Normalized", ylabel2=None):
    """Plot two graphs together."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 6.5))

    color = "tab:blue"
    df1.plot(ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    df2.plot(ax=ax2, color=color, legend=None)
    ax2.set_ylabel(ylabel2, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# Common functions
def rebase(df, date=None):
    """Rebase."""
    if date is not None:
        return df.divide(df[df.index == date].values[0])
    return df.divide(df.iloc[0])


def pct_change(df, periods=1, freq=1):
    """Compute percentage change."""
    return (df / df.shift(periods)) ** freq - 1


def log_change(df, periods=1, freq=1):
    """Compute log change."""
    return np.log(df / df.shift(periods)) * freq


def diff_change(df, periods=1, freq=1):
    """Compute difference change."""
    return (df - df.shift(periods)) * freq


def last_bdate(df, date):
    """Get value for the most recent business date."""
    date = datetime.strptime(date, "%Y-%m-%d")
    v = df.loc[df.index == date]
    while date.weekday() > 4 or np.isnan(v.item()):
        date = date - timedelta(days=1)
        v = df.loc[df.index == date]
    return date.strftime("%Y-%m-%d"), v.item()


def annualise(p, years):
    """Compute annualized rate of p."""
    return (1 + p) ** (1 / years) - 1


# Technical indicators
def compute_sma(ts, window):
    """Compute simple moving average."""
    return ts.rolling(window=window, center=False).mean()


def compute_ema(ts, window=20):
    """Compute exponential moving average."""
    return ts.ewm(com=(window-1)/2).mean()


def compute_cci(ts, window=20):
    """Compute CCI."""
    typical = (ts["high"] + ts["low"] + ts["close"]) / 3
    return (typical - compute_sma(typical, window)) / (0.015 * typical.mad())


def compute_macd(ts, nfast=12, nslow=26):
    """Compute moving average convergence/divergence."""
    ema_fast = compute_ema(ts, window=nfast)
    ema_slow = compute_ema(ts, window=nslow)
    return ema_fast-ema_slow


def compute_bbands(ts, window=20, nbdevup=2, nbdevdn=2):
    """Compute upper and lower Bollinger Bands."""
    sma = compute_sma(ts, window=window)
    rstd = ts.rolling(window=window, center=False).std()
    upper_band = sma + nbdevup * rstd
    lower_band = sma - nbdevdn * rstd
    return sma, upper_band, lower_band
