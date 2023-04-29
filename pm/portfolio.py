"""
Script containing commonly used functions for portfolio.
"""
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from analyser.data import get_data
from analyser.plots import plot_normalized_data
from pm import FLOWDATA, SUMMARY_DIR


BASE_SYMBOLS = {
    "SGD": "ES3.SI",
    "USD": "IWDA.L",
    "SRS": "USDSGD=X",
    "Fund": "USDSGD=X",
    "Bond": "ES3.SI",
    "IDR": "SGDIDR=X",
}
MAIN_SYMBOLS = {
    "USD": ["IWDA.L", "EIMI.L"],
    "IDR": ["GOTO.JK"],
}


def _get_data(
    symbols: List[str],
    dates: pd.DatetimeIndex,
    base_symbol: str = "ES3.SI",
    col: str = "adjclose",
    dirname: str = "data",
):
    """Wrapper for get_data."""
    df = get_data(symbols, dates, base_symbol=base_symbol, col=col, dirname=dirname)
    if "SB" in df.columns:
        df["SB"] = 100  # default value for bonds
    return df


def get_portfolio_value(
    prices: pd.DataFrame,
    allocs: Optional[np.ndarray] = None,
    start_val: float = 1.,
    units: Optional[int] = None,
):
    """Compute daily portfolio value given stock prices, allocations and
    starting value, or units.

    Args:
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)
        units: initial number of units

    Returns:
        port_val: daily portfolio value
    """
    if allocs is None and units is None:
        raise Exception("Input 'allocs' or 'units'")

    if units is None:
        units = np.divide(allocs, prices.iloc[0]) * start_val
    port_val = pd.Series(np.dot(prices, units), index=prices.index, name="Portfolio")
    return port_val


def portfolio_const(
    start_date: str,
    end_date: str,
    symbols: List[str],
    allocs: Optional[np.ndarray] = None,
    start_val: float = 1.,
    units: Optional[int] = None,
):
    """Constant portfolio."""
    if allocs is None and units is None:
        raise Exception("Input 'allocs' or 'units'")

    dates = pd.date_range(start_date, end_date)
    prices_all = _get_data(symbols, dates)  # automatically adds index
    prices = prices_all[symbols]  # only portfolio symbols
    return get_portfolio_value(prices, allocs, start_val, units)


def get_portfolio_stats(
    port_val: pd.DataFrame, rfr: float = 0., freq: int = 252
) -> Tuple[float]:
    """Calculate statistics on given portfolio values.

    Args:
        port_val: daily portfolio value
        rfr: risk-free rate of return (default: 0%)
        freq: annual sampling frequency (default: 252 trading days)

    Returns:
        cum_ret: cumulative return
        annual_ret: annualized return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    cumul_ret = port_val.iloc[-1] / port_val.iloc[0] - 1
    annual_ret = (1 + cumul_ret) ** (freq / len(port_val)) - 1
    daily_ret = port_val / port_val.shift(1) - 1
    avg_ret = daily_ret.mean() * freq
    std_ret = daily_ret.std() * np.sqrt(freq)
    sharpe = (avg_ret - rfr) / (std_ret + 1e-8)

    return cumul_ret, annual_ret, avg_ret, std_ret, sharpe


def assess_portfolio(
    port_val: pd.DataFrame, rfr: float = 0., bm: str = "ES3.SI"
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Simulates and assesses the performance of a stock portfolio.

    Args:
        portval: portfolio value time series
        rfr: risk-free rate of return (default: 0%)
        bm: benchmark
    """
    start_date = port_val.index[0].date()
    end_date = port_val.index[-1].date()

    # Benchmark
    dates = pd.date_range(start_date, end_date)
    price_bm = _get_data([bm], dates)[[bm]]
    port_val_bm = get_portfolio_value(price_bm, units=[1.0])
    (
        cumul_ret_bm,
        annual_ret_bm,
        avg_ret_bm,
        std_ret_bm,
        sharpe_bm,
    ) = get_portfolio_stats(port_val_bm, rfr=rfr)
    m2_bm = sharpe_bm * std_ret_bm + rfr

    # Get portfolio statistics
    cumul_ret, annual_ret, avg_ret, std_ret, sharpe = get_portfolio_stats(
        port_val, rfr=rfr
    )
    # Compute Modigliani risk-adjusted performance
    m2 = sharpe * std_ret_bm + rfr

    results_df = pd.DataFrame(
        [
            [m2, sharpe, cumul_ret, annual_ret, avg_ret, std_ret],
            [m2_bm, sharpe_bm, cumul_ret_bm, annual_ret_bm, avg_ret_bm, std_ret_bm],
        ],
        columns=[
            "M2",
            "Sharpe Ratio",
            "Cumulative Return",
            "Annualized Return",
            "Average Daily Return",
            "Volatility",
        ],
        index=["Fund", "Benchmark"],
    )

    ts_df = price_bm.join(port_val)
    return (
        {"start_date": start_date, "end_date": end_date, "port_val": port_val[-1]},
        results_df,
        ts_df,
    )


def print_results(dct: dict, results_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    """Print statistics."""
    print("Date Range: {} to {}\n".format(dct["start_date"], dct["end_date"]))
    print("Final Portfolio Value: {:.2f}".format(dct["port_val"]))

    display(results_df)

    # Compare daily portfolio value with index using a normalized plot
    _, ax = plt.subplots()
    plot_normalized_data(ts_df, title="Daily portfolio value and benchmark", ax=ax)
    plt.show()


def _map_stock_to_symbol(sheet: str, xlsx_file: str) -> pd.DataFrame:
    """Map stock to symbol."""
    all_symbols = pd.read_excel(
        xlsx_file,
        sheet_name=f"{sheet} Summary",
        usecols=["Name", "Yahoo Quote"],
    )
    all_symbols.columns = ["Stock", "Symbol"]
    all_symbols = all_symbols.dropna()
    return all_symbols


def compute_portvals(
    end_date: str, start_date: str = "2015-01-01", sheet: str = "SGD", xlsx_file: str = FLOWDATA
) -> pd.DataFrame:
    """Compute daily portfolio value."""
    # Load data
    df = pd.read_excel(
        xlsx_file,
        sheet_name=f"{sheet} Txn",
        parse_dates=["Date"],
        usecols=["Date", "Type", "Stock", "Transacted Units"],
    )
    df.columns = ["Date", "Type", "Stock", "Units"]
    df = df[df["Type"].isin(["Buy", "Sell"])]
    df = df.groupby(["Date", "Type", "Stock"], as_index=False).sum()

    # Convert Stock to Symbol
    all_symbols = _map_stock_to_symbol(sheet=sheet, xlsx_file=xlsx_file)
    df_orders = pd.merge(df, all_symbols, on=["Stock"])

    # Get daily prices
    if sheet in ["SRS", "Fund", "SGD", "Bond"]:
        symbols = df_orders["Symbol"].drop_duplicates().values.tolist()
    else:
        symbols = MAIN_SYMBOLS[sheet]
    dates = pd.date_range(start_date, end_date)
    prices = _get_data(symbols, dates, base_symbol=BASE_SYMBOLS[sheet], col="close")[symbols]
    if sheet == "IDR":  # HACK
        prices /= 10000

    # Get daily units
    def _get_units(ttype):
        df_tmp = (
            df_orders
            .query("Type == @ttype")
            .groupby(["Date", "Symbol"], as_index=False)[["Units"]]
            .sum()
            .pivot(index="Date", columns="Symbol", values="Units")
        )
        df_tmp = pd.DataFrame(index=prices.index).join(df_tmp)
        df_tmp = df_tmp.fillna(0)

        for s in symbols:
            if s not in df_tmp.columns:
                df_tmp[s] = 0.0
        df_tmp = df_tmp[symbols]
        return df_tmp

    df_units = _get_units("Buy") - _get_units("Sell")
    df_units = df_units.cumsum(axis=0)

    # Compute portfolio values
    portvals = (prices * df_units).sum(axis=1)

    # HACK: for SRS and Fund, concat with csv
    if sheet == "SRS":
        ut = pd.read_csv(f"{SUMMARY_DIR}/SRS.csv", index_col="date", parse_dates=True)
        portvals = pd.concat([portvals[portvals.index < ut.index[0]], ut["close"]])
    elif sheet == "Fund":
        core = pd.read_csv(f"{SUMMARY_DIR}/Core.csv", index_col="date", parse_dates=True)
        esg = pd.read_csv(f"{SUMMARY_DIR}/ESG.csv", index_col="date", parse_dates=True)
        portvals = pd.concat([portvals, core["close"] + esg["close"]]).groupby(level=0).sum()
    return portvals


def agg_daily_cost(sheet: str, xlsx_file: str) -> pd.DataFrame:
    """Aggregate cost daily."""
    # Load data
    df = pd.read_excel(
        xlsx_file,
        sheet_name=f"{sheet} Txn",
        parse_dates=["Date"],
        usecols=["Date", "Type", "Stock", "Transacted Value", "Gains from Sale"],
    )
    if sheet == "USD":
        df = df[df["Stock"].isin(["iShares MSCI EM ETF", "iShares MSCI World ETF"])]
    df.columns = ["Date", "Type", "Stock", "Value", "Realised_Gain"]
    df["Value"] -= df["Realised_Gain"]  # to obtain original cost

    # Compute cost
    cost_df = df[df["Type"].isin(["Buy", "Sell"])].copy()

    cost_df["Value"] *= (cost_df["Type"] == "Buy") * 2 - 1
    cost_df = cost_df.groupby(["Date"])[["Value"]].sum()
    return cost_df


def compute_cost(
    end_date: str, start_date: str = "2015-01-01", sheet: str = "SGD", xlsx_file: str = FLOWDATA
) -> pd.DataFrame:
    """Compute cumulative cost and daily benchmark portfolio value."""
    cost_df = agg_daily_cost(sheet, xlsx_file)

    dates = pd.date_range(start_date, end_date)
    prices_bm = _get_data(["ES3.SI"], dates, col="close")
    cost_df = prices_bm.join(cost_df)
    cost_df.fillna(0, inplace=True)
    cost_df["Units_bm"] = cost_df["Value"] / cost_df["ES3.SI"]

    for c in ["Value", "Units_bm"]:
        cost_df[c] = cost_df[c].cumsum()

    # Compute benchmark portfolio
    cost_df["Value_bm"] = cost_df["ES3.SI"] * cost_df["Units_bm"]

    cost_df = cost_df[["Value", "Value_bm"]]
    cost_df.columns = ["Cost", "Benchmark"]
    return cost_df


def agg_daily_gain(gain_type: str, sheet: str, xlsx_file: str) -> pd.DataFrame:
    """Aggregate gain daily."""
    # Load data
    df = pd.read_excel(
        xlsx_file,
        sheet_name=f"{sheet} Txn",
        parse_dates=["Date"],
        usecols=["Date", "Type", "Stock", "Gains from Sale"],
    )
    if sheet == "USD":
        df = df[df["Stock"].isin(["iShares MSCI EM ETF", "iShares MSCI World ETF"])]
    elif sheet == "IDR":
        df = df[df["Stock"].isin(["GoTo"])]
    df.columns = ["Date", "Type", "Stock", "Value"]

    # Compute gains
    gains_df = df[df["Type"] == gain_type].groupby(["Date"])[["Value"]].sum()
    return gains_df


def compute_gains(
    gain_type: str,
    end_date: str,
    start_date: str = "2015-01-01",
    sheet: str = "SGD",
    xlsx_file: str = FLOWDATA,
) -> pd.DataFrame:
    """Compute cumulative gains."""
    gains_df = agg_daily_gain(gain_type, sheet, xlsx_file)

    gains_df = gains_df.cumsum()

    dates = pd.date_range(start_date, end_date)
    prices_bm = _get_data([BASE_SYMBOLS[sheet]], dates, col="close")
    gains_df = pd.DataFrame(index=prices_bm.index).join(gains_df)
    gains_df.iloc[0] = 0
    gains_df.fillna(method="ffill", inplace=True)
    return gains_df["Value"]


def compute_etf(end_date: str, start_date: str = "2019-07-01", xlsx_file: str = FLOWDATA) -> pd.DataFrame:
    """Compute cumulative cost and cash."""
    # Load data
    df = pd.read_excel(
        xlsx_file,
        sheet_name="USD Txn",
        parse_dates=["Date"],
        usecols=[
            "Date",
            "Type",
            "Stock",
            "Transacted Units",
            "Transacted Value",
            "Gains from Sale",
        ],
    )
    df.columns = ["Date", "Type", "Stock", "Units", "Value", "Realised_Gain"]
    df["Value"] -= df["Realised_Gain"]  # to obtain original cost

    # Compute cost
    df = (
        df[df["Type"].isin(["Buy", "Sell"])]
        .groupby(["Date", "Type", "Stock"], as_index=False)
        .sum()
    )
    for c in ["Units", "Value"]:
        df[c] *= (df["Type"] == "Buy") * 2 - 1

    cost_df = df.pivot(index="Date", columns="Stock", values="Value")
    cost_df.rename(
        columns={
            "SGD Deposit": "SGD_Deposit",
            "USD Deposit": "USD_Deposit",
            "iShares MSCI EM ETF": "EIMI.L",
            "iShares MSCI World ETF": "IWDA.L",
        },
        inplace=True,
    )

    dates = pd.date_range(start_date, end_date)
    # prices = get_xlsx(["USDSGD"], dates, col="Close", xlsx=XLSX_FILE)
    prices = _get_data([], dates, col="close", base_symbol="USDSGD=X")
    prices.rename(columns={"USDSGD=X": "USDSGD"}, inplace=True)

    cost_df = prices[["USDSGD"]].join(cost_df)
    cost_df.fillna(0, inplace=True)
    for c in ["SGD_Deposit", "USD_Deposit", "USD-SGD", "EIMI.L", "IWDA.L"]:
        cost_df[c] = cost_df[c].cumsum()

    # Compute Cash
    units_df = df.query("Stock == 'USD-SGD'").pivot(
        index="Date", columns="Stock", values="Units"
    )
    units_df = prices[["USDSGD"]].join(units_df)[["USD-SGD"]]
    units_df.fillna(0, inplace=True)
    units_df = units_df.cumsum()

    cost_df["Cash_SGD"] = cost_df["SGD_Deposit"] - cost_df["USD-SGD"]
    cost_df["Cash_USD"] = (
        cost_df["USD_Deposit"]
        + units_df["USD-SGD"]
        - cost_df["IWDA.L"]
        - cost_df["EIMI.L"]
    )
    cost_df["Cash"] = cost_df["Cash_USD"] + cost_df["Cash_SGD"] / cost_df["USDSGD"]
    # TODO: Cost fluctuating due to FX
    cost_df["Cost"] = (
        cost_df["USD_Deposit"]
        + units_df["USD-SGD"]
        + cost_df["Cash_SGD"] / cost_df["USDSGD"]
    )
    return cost_df[["USDSGD", "Cost", "Cash"]]


def compute_idr(end_date: str, start_date: str,  xlsx_file: str):
    df = pd.read_excel(
        xlsx_file,
        sheet_name="IDR Txn",
        parse_dates=["Date"],
        usecols=[
            "Date",
            "Type",
            "Stock",
            "Transacted Units",
            "Transacted Value",
            "Gains from Sale",
        ],
    )
    df.columns = ["Date", "Type", "Stock", "Units", "Value", "Realised_Gain"]
    df["Value"] -= df["Realised_Gain"]  # to obtain original cost

    # Compute cost
    df = (
        df[df["Type"].isin(["Buy", "Sell"])]
        .groupby(["Date", "Type", "Stock"], as_index=False)
        .sum()
    )
    for c in ["Units", "Value"]:
        df[c] *= (df["Type"] == "Buy") * 2 - 1

    cost_df = df.pivot(index="Date", columns="Stock", values="Value")
    cost_df.rename(
        columns={
            "SGD Deposit": "SGD_Deposit",
            "IDR Deposit": "IDR_Deposit",
            "GoTo": "GOTO.JK",
        },
        inplace=True,
    )

    dates = pd.date_range(start_date, end_date)
    prices = _get_data([], dates, col="close", base_symbol="SGDIDR=X")
    # prices.rename(columns={"SGDIDR=X": "SGDIDR"}, inplace=True)
    prices["IDRSGD"] = 10000 / prices["SGDIDR=X"]  # HACK

    cost_df = prices[["IDRSGD"]].join(cost_df)
    cost_df.fillna(0, inplace=True)
    for c in ["SGD_Deposit", "IDR_Deposit", "IDR-SGD", "GOTO.JK"]:
        cost_df[c] = cost_df[c].cumsum()

    # Compute Cash
    units_df = df.query("Stock == 'IDR-SGD'").pivot(
        index="Date", columns="Stock", values="Units"
    )
    units_df = prices[["IDRSGD"]].join(units_df)[["IDR-SGD"]]
    units_df.fillna(0, inplace=True)
    units_df = units_df.cumsum()

    cost_df["Cash_SGD"] = cost_df["SGD_Deposit"] - cost_df["IDR-SGD"]
    cost_df["Cash_IDR"] = (
        cost_df["IDR_Deposit"]
        + units_df["IDR-SGD"]
        - cost_df["GOTO.JK"]
    )
    cost_df["Cash"] = cost_df["Cash_IDR"] + cost_df["Cash_SGD"] / cost_df["IDRSGD"]
    # TODO: Cost fluctuating due to FX
    cost_df["Cost"] = (
        cost_df["IDR_Deposit"]
        + units_df["IDR-SGD"]
        + cost_df["Cash_SGD"] / cost_df["IDRSGD"]
    )
    return cost_df[["IDRSGD", "Cost", "Cash"]]


# TODO: from analyser/data.py

# def get_xlsx(
#     symbols: List[str],
#     dates: pd.DatetimeIndex,
#     base_symbol: str = "USDSGD",
#     col: str = "Close",
#     xlsx: str = "data.xlsx",
# ) -> pd.DataFrame:
#     """Load stock data for given symbols from xlsx file."""
#     df = pd.DataFrame(index=dates)
#     df.index.name = "Date"
#     if base_symbol not in symbols:
#         symbols = [base_symbol] + symbols

#     for symbol in symbols:
#         df_temp = pd.read_excel(
#             xlsx,
#             index_col="Date",
#             parse_dates=True,
#             sheet_name=symbol,
#             usecols=["Date", col],
#         )
#         df_temp.index = df_temp.index.date
#         df_temp.rename(columns={col: symbol}, inplace=True)
#         df = df.join(df_temp)
#         if symbol == base_symbol:  # drop dates that base_symbol did not trade
#             df = df.dropna(subset=[base_symbol])
#     df = df.replace([0], [np.nan])
#     fill_missing_values(df)
#     return df


# def get_xlsx_ohlcv(
#     symbol: str,
#     dates: pd.DatetimeIndex,
#     base_symbol: str = "USDSGD",
#     xlsx: str = "data.xlsx",
# ) -> pd.DataFrame:
#     """Load stock ohlcv data for given symbol from xlsx files."""
#     df_base = pd.read_excel(
#         xlsx,
#         index_col="Date",
#         parse_dates=True,
#         sheet_name=symbol,
#         usecols=["Date", "Close"],
#     )
#     df_base.columns = [base_symbol]
#     df_base.index = df_base.index.date
#     df_base.index.name = "date"

#     df = pd.DataFrame(index=dates)
#     df.index.name = "date"
#     df = df.join(df_base)
#     df = df.dropna(subset=[base_symbol])

#     df_temp = pd.read_excel(
#         xlsx,
#         parse_dates=True,
#         sheet_name=symbol,
#         index_col="Date",
#         usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
#     )
#     df_temp.columns = ["open", "high", "low", "close", "volume"]
#     df_temp.index = df_temp.index.date
#     df_temp.index.name = "date"

#     df = df.join(df_temp)
#     df = df.replace([0], [np.nan])
#     df = df.drop_duplicates()
#     fill_missing_values(df)
#     return df


def get_portfolio(end_date, start_date, sheet, xlsx_file):
    if sheet in ["SGD", "Fund", "SRS", "Bond"]:
        df = compute_cost(end_date, start_date, sheet, xlsx_file)
        df["Portfolio"] = compute_portvals(end_date, start_date, sheet, xlsx_file)
    elif sheet == "USD":
        df = compute_etf(end_date, start_date,  xlsx_file)
        df["Equity"] = compute_portvals(end_date, start_date, sheet, xlsx_file)
        df["Portfolio"] = df["Equity"] + df["Cash"]
    elif sheet == "IDR":
        df = compute_idr(end_date, start_date,  xlsx_file)
        df["Equity"] = compute_portvals(end_date, start_date, sheet, xlsx_file)
        df["Portfolio"] = df["Equity"] + df["Cash"]
    else:
        raise NotImplementedError

    df["Div"] = compute_gains("Div", end_date, start_date, sheet, xlsx_file)
    df["Realised_Gain"] = compute_gains("Sell", end_date, start_date, sheet, xlsx_file)
    df["Paper_Gain"] = df["Portfolio"] - df["Cost"]
    df["Net_Gain"] = df["Paper_Gain"] + df["Realised_Gain"] + df["Div"]
    df = df.dropna()
    return df


if __name__ == "__main__":
    end_date = datetime.today().isoformat()

    print("Generating portfolio_sgd...")
    sgd_df = get_portfolio(end_date, "2015-01-01", "SGD", "data/summary/aSummary.xlsx")
    sgd_df.to_csv("data/summary/portfolio_sgd.csv")

    print("Generating portfolio_usd...")
    usd_df = get_portfolio(end_date, "2019-07-01", "USD", "data/summary/aSummary.xlsx")
    usd_df.to_csv("data/summary/portfolio_usd.csv")

    print("Generating portfolio_fund...")
    fund_df = get_portfolio(end_date, "2021-04-01", "Fund", "data/summary/aSummary.xlsx")
    fund_df.to_csv("data/summary/portfolio_fund.csv")

    print("Generating portfolio_srs...")
    srs_df = get_portfolio(end_date, "2019-02-01", "SRS", "data/summary/aSummary.xlsx")
    srs_df.to_csv("data/summary/portfolio_srs.csv")

    print("Generating portfolio_bond...")
    bond_df = get_portfolio(end_date, "2015-11-01", "Bond", "data/summary/aSummary.xlsx")
    bond_df.to_csv("data/summary/portfolio_bond.csv")

    print("Generating portfolio_idr...")
    idr_df = get_portfolio(end_date, "2023-02-01", "IDR", "data/summary/aSummary2.xlsx")
    idr_df.to_csv("data/summary/portfolio_idr.csv")
