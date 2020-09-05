"""
Script containing commonly used functions for analysis.
"""
import datetime
import os

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt


def get_data(symbols, dates, base_symbol='D05.SI', col='adjclose', dirname='data'):
    """Load stock data for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    df.index.name = 'date'
    if base_symbol not in symbols:
        symbols = [base_symbol] + symbols
    if "ES3.SI" not in symbols:
        symbols.append("ES3.SI")

    for symbol in symbols:
        if symbol == "SB.SI":
            df[symbol] = 1000
        else:
            df_temp = pd.read_csv(
                os.path.join(dirname, f'{symbol}.csv'), index_col='date',
                parse_dates=True, na_values=['nan'], usecols=['date', col])
            df_temp = df_temp.rename(columns={col: symbol})
            df = df.join(df_temp)

        if symbol == base_symbol:  # drop dates index did not trade
            df = df.dropna(subset=[base_symbol])

    df = df.replace([0], [np.nan])
    fill_missing_values(df)
    return df


def get_data_xlsx(symbols, dates, base_symbol='USDSGD', col='Close', dirname='data'):
    """Load stock data for given symbols from xlsx file."""
    df = pd.DataFrame(index=dates)
    df.index.name = 'Date'
    if base_symbol not in symbols:
        symbols = [base_symbol] + symbols

    for symbol in symbols:
        df_temp = pd.read_excel(
            os.path.join(dirname, 'prices.xlsx'), index_col='Date',
            parse_dates=True, sheet_name=symbol, usecols=['Date', col])
        df_temp.index = df_temp.index.date
        df_temp = df_temp.rename(columns={col: symbol})
        df = df.join(df_temp)
        if symbol == base_symbol:  # drop dates index did not trade
            df = df.dropna(subset=[base_symbol])
    df = df.replace([0], [np.nan])
    fill_missing_values(df)
    return df


def get_data_ohlcv(symbol, dates, base_symbol='ES3.SI', dirname='data'):
    """Load stock ohlcv data for given symbol from CSV files."""
    df = pd.DataFrame(index=dates)
    df.index.name = 'date'
    df_base = pd.read_csv(
        os.path.join(dirname, f'{base_symbol}.csv'), index_col='date',
        parse_dates=True, na_values=['nan'], usecols=['date', 'close'])
    df_base = df_base.rename(columns={'close': base_symbol})
    df = df.join(df_base)
    df = df.dropna(subset=[base_symbol])

    df_temp = pd.read_csv(
        os.path.join(dirname, f'{symbol}.csv'), index_col='date',
        parse_dates=True, na_values=['nan'],
        usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = df.join(df_temp)
    df = df.replace([0], [np.nan])
    df = df.drop_duplicates()
    fill_missing_values(df)
    return df


def fill_missing_values(df):
    """Fill missing values in dataframe."""
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)


def get_ie_data(start_date="1871-01-01"):
    """Load Shiller data."""
    df = pd.read_excel('data/ie_data.xls', sheet_name='Data', skiprows=7)
    df.drop(['Fraction', 'Unnamed: 13', 'Unnamed: 15'], axis=1, inplace=True)
    df.columns = ['Date', 'S&P500', 'Dividend', 'Earnings', 'CPI', 'Long_IR',
                  'Real_Price', 'Real_Dividend', 'Real_TR_Price',
                  'Real_Earnings', 'Real_TR_Scaled_Earnings', 'CAPE', 'TRCAPE']
    df['Date'] = pd.to_datetime(df['Date'].astype(str))
    df.set_index('Date', inplace=True)
    df = df.iloc[:-1]
    df = df[df.index >= start_date]
    return df


# Plot functions
def plot_data(df, title='', xlabel='Date', ylabel='Price', ax=None):
    """Plot stock prices."""
    ax = df.plot(title=title, fontsize=12, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_normalized_data(df, title='', xlabel='Date', ylabel='Normalized', ax=None):
    """Plot normalized stock prices."""
    normdf = rebase(df)
    ax = plot_data(normdf, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    ax.axhline(y=1, linestyle='--', color='k')
    return ax


def plot_bollinger(df, title=None, ax=None):
    """Plot bollinger bands and SMA."""
    df2 = df[['close']].copy()
    _, df2['upper'], df2['lower'] = compute_bbands(df['close'])
    df2['sma200'] = compute_sma(df['close'], 200)
    df2['sma50'] = compute_sma(df['close'], 50)

    ax = plot_data(df2, title=title, ax=ax)
    return df2, ax


def plot_with_two_scales(df1, df2, xlabel='Date', ylabel1='Normalized', ylabel2=None):
    """Plot two graphs together."""
    fig, ax1 = plt.subplots(figsize=(9, 6.5))

    color = 'tab:blue'
    df1.plot(ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    df2.plot(ax=ax2, color=color, legend=None)
    ax2.set_ylabel(ylabel2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

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
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    v = df.loc[df.index == date]
    while date.weekday() > 4 or np.isnan(v.item()):
        date = date - datetime.timedelta(days=1)
        v = df.loc[df.index == date]
    return date.strftime('%Y-%m-%d'), v.item()


# Technical indicators
def compute_sma(ts, window):
    """Compute simple moving average."""
    return ts.rolling(window=window, center=False).mean()


def compute_ema(ts, window=20):
    """Compute exponential moving average."""
    return ts.ewm(com=(window-1)/2).mean()


def compute_cci(ts, window=20):
    """Compute CCI."""
    typical = (ts['high'] + ts['low'] + ts['close']) / 3
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


def linearfit(ts):
    """Computes linear fit of values."""
    y = ts.values
    p = np.polyfit(range(len(y)), y, deg=1)
    yfit = np.polyval(p, range(len(y)))
    last = yfit[-1]
    residual = np.sqrt(np.mean((yfit - y) ** 2))
    level = 50 + 100 * (y[-1] - last) / (4 * residual)
    grad = p[0] / y[0]
    pred = np.polyval(p, [len(y)])[0]
    return yfit, level, residual, last, grad, pred


def compute_trend(dates, symbol):
    """Compute linear trend."""
    df = get_data([symbol], dates)[[symbol]]

    yfit, level, residual, last, grad, _ = linearfit(df[symbol])
    df["p0"] = yfit - residual * 2
    df["p25"] = yfit - residual
    df["p50"] = yfit
    df["p75"] = yfit + residual
    df["p100"] = yfit + residual * 2
    return df, level, residual, last, grad


def plot_trend(symbol, start_date, end_date, name='', ax=None):
    """Plot time series with trends."""
    dates = pd.date_range(start_date, end_date)
    df, level, res, last, grad = compute_trend(dates, symbol)

    title = (
        f'{name} ({symbol}): {df[symbol].iloc[-1]:.3f} ({level:.1f}%)\n'
        f'[{last-2*res:.3f}, {last-res:.3f}, {last:.3f}, {last+res:.3f}, {last+2*res:.3f}], '
        f'{grad * 1e3:.3f}'
    )

    df[symbol].plot(color='blue', ax=ax)
    df["p0"].plot(color='green', ax=ax)
    df["p25"].plot(color='green', ax=ax)
    df["p50"].plot(color='green', ax=ax)
    df["p75"].plot(color='red', ax=ax)
    df["p100"].plot(color='red', ax=ax)
    ax.set_title(title)
    return ax


def get_trending(dates, symbols, misc_symbols=None):
    """Compute list of top trending."""
    results = list()
    df1 = get_data(symbols, dates)
    for symbol in symbols:
        _, level, res, _, grad, pred = linearfit(df1[symbol])
        results.append([symbol, df1[symbol].iloc[-1], level, grad * 1e3,
                        pred - 2 * res, pred - res, pred, pred + res, pred + 2 * res])

    if misc_symbols is not None:
        df2 = get_data_xlsx(misc_symbols, dates)
        for symbol in misc_symbols:
            _, level, res, _, grad, pred = linearfit(df2[symbol])
            results.append([symbol, df2[symbol].iloc[-1], level, grad * 1e3,
                            pred - 2 * res, pred - res, pred, pred + res, pred + 2 * res])

    results = pd.DataFrame(results, columns=[
        "symbol", "close", "level", "grad", "p0", "p25", "p50", "p75", "p100"])
    return results


# Portfolio
def compute_xnpv(cashflows, rate):
    """Compute the net present value of a series of cashflows
    at irregular intervals.

    Args:
        cashflows: pandas.Series of values with dates as index

    Returns:
        NPV of the given cash flows
    """
    arr = cashflows.reset_index().values
    t0 = arr[0, 0]
    return np.sum([r[1] / (1 + rate) ** ((r[0] - t0).days / 365.25) for r in arr])


def compute_xirr(cashflows, initial=0.1):
    """Compute internal rate of return of a series of cashflows
    at irregular intervals.

    Args:
        cashflows: numpy.array of datetimes and values
        initial: initial guess

    Returns:
        XIRR
    """
    return sco.newton(lambda r: compute_xnpv(cashflows, r), initial)
