from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyser.data import get_data
from analyser.plots import plot_normalized_data
from pm import CFG, logger


def _get_data(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    sheet: str,
    col: str = "adjclose",
):
    """Wrapper for get_data."""
    df = get_data(
        symbols,
        dates,
        base_symbol=CFG.PFL[sheet].BASE_SYMBOL,
        col=col,
        dirname=CFG.DATA_DIR,
    )
    if "SB" in df.columns:
        df["SB"] = 100  # default value for bonds
    return df


def get_portfolio_value(
    prices: pd.DataFrame,
    allocs: Optional[np.ndarray] = None,
    start_val: float = 1.0,
    units: Optional[int] = None,
) -> pd.Series:
    """Compute daily portfolio value given stock prices, allocations and
    starting value, or units.

    Args:
        prices (pd.DataFrame): Prices of the assets in the portfolio.
        allocs (Optional[np.ndarray]): Allocations of the assets in the portfolio.
        start_val (float): Initial value of the portfolio.
        units (Optional[int]): Number of units of each asset in the portfolio.

    Returns:
        pd.Series: The value of the portfolio over time.
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
    symbols: list[str],
    allocs: Optional[np.ndarray] = None,
    start_val: float = 1.0,
    units: Optional[int] = None,
) -> pd.Series:
    """Constructs a constant portfolio.

    Args:
        start_date (str): Start date of the portfolio.
        end_date (str): End date of the portfolio.
        symbols (list[str]): list of symbols to include in the portfolio.
        allocs (Optional[np.ndarray]): Weights of each symbol in the portfolio.
        start_val (float): Initial value of the portfolio.
        units (Optional[int]): Number of units of each symbol to purchase.

    Returns:
        pd.Series: The value of the portfolio over time.
    """
    if allocs is None and units is None:
        raise Exception("Input 'allocs' or 'units'")

    dates = pd.date_range(start_date, end_date)
    prices_all = _get_data(symbols, dates, "SGD")  # automatically adds index
    prices = prices_all[symbols]  # only portfolio symbols
    return get_portfolio_value(prices, allocs, start_val, units)


def get_portfolio_stats(port_val: pd.DataFrame, rfr: float = 0.0, freq: int = 252) -> tuple[float]:
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
    port_val: pd.DataFrame, rfr: float = 0.0, bm: str = "ES3.SI"
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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
    price_bm = _get_data([bm], dates, "SGD")[[bm]]
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
    cumul_ret, annual_ret, avg_ret, std_ret, sharpe = get_portfolio_stats(port_val, rfr=rfr)
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
        {"start_date": start_date, "end_date": end_date, "port_val": port_val.iloc[-1]},
        results_df,
        ts_df,
    )


def print_results(dct: dict, results_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    """Print statistics."""
    from IPython.display import display

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


def _load_fund_portvals():
    dfs = None
    for fn in CFG.FUNDNAMES:
        df = pd.read_csv(f"{CFG.SUMMARY_DIR}/{fn}.csv", index_col="date", parse_dates=True)
        df.columns = [fn]
        if dfs is None:
            dfs = df
        else:
            dfs = dfs.join(df, how="outer")

    dfs.ffill(inplace=True)
    dfs.fillna(0, inplace=True)
    portvals = dfs[CFG.FUNDNAMES].sum(axis=1)
    return portvals


def compute_portvals(
    end_date: str,
    start_date: str = "2015-01-01",
    sheet: str = "SGD",
    xlsx_file: str = CFG.FLOWDATA,
) -> pd.DataFrame:
    """Compute daily portfolio value."""
    if sheet == "Fund":
        # HACK: bypass
        return _load_fund_portvals()

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
        # symbols = MAIN_SYMBOLS[sheet]
        symbols = list(CFG.PFL[sheet].MAIN_SYMBOLS.values())
    dates = pd.date_range(start_date, end_date)
    prices = _get_data(symbols, dates, sheet=sheet, col="close")[symbols]

    # Get daily units
    def _get_units(ttype):
        df_tmp = (
            df_orders.query("Type == @ttype")
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

    # HACK: for SRS, concat with csv
    if sheet == "SRS":
        ut = pd.read_csv(f"{CFG.SUMMARY_DIR}/SRS.csv", index_col="date", parse_dates=True)
        portvals = pd.concat([portvals[portvals.index < ut.index[0]], ut["close"]])
    return portvals


def agg_daily_cost(sheet: str, xlsx_file: str) -> pd.DataFrame:
    """Aggregate daily cost from a sheet in an xlsx file.

    Args:
      sheet (str): The name of the sheet to aggregate.
      xlsx_file (str): The path to the xlsx file.

    Returns:
      pd.DataFrame: A DataFrame with the aggregated daily cost.
    """
    # Load data
    df = pd.read_excel(
        xlsx_file,
        sheet_name=f"{sheet} Txn",
        parse_dates=["Date"],
        usecols=["Date", "Type", "Stock", "Transacted Value", "Gains from Sale"],
    )
    df.columns = ["Date", "Type", "Stock", "Value", "Realised_Gain"]
    df["Value"] -= df["Realised_Gain"]  # to obtain original cost

    # Compute cost
    cost_df = df[df["Type"].isin(["Buy", "Sell"])].copy()

    cost_df["Value"] *= (cost_df["Type"] == "Buy") * 2 - 1
    cost_df = cost_df.groupby(["Date"])[["Value"]].sum()
    return cost_df


def compute_cost(
    trading_dates: pd.DatetimeIndex, sheet: str, xlsx_file: str = CFG.FLOWDATA
) -> pd.DataFrame:
    """Compute cumulative cost and daily benchmark portfolio value."""
    cost_df = agg_daily_cost(sheet, xlsx_file)
    cost_df = pd.DataFrame(index=trading_dates).join(cost_df)
    cost_df.fillna(0, inplace=True)

    if sheet == "SGD":
        prices_bm = _get_data([], trading_dates, sheet, col="close")
        cost_df = cost_df.join(prices_bm)
        cost_df.ffill(inplace=True)
        cost_df["Units_bm"] = cost_df["Value"] / cost_df["ES3.SI"]

        for c in ["Value", "Units_bm"]:
            cost_df[c] = cost_df[c].cumsum()

        # Compute benchmark portfolio
        cost_df["Value_bm"] = cost_df["ES3.SI"] * cost_df["Units_bm"]

        cost_df = cost_df[["Value", "Value_bm"]]
        cost_df.columns = ["Cost", "Benchmark"]
    else:
        cost_df["Value"] = cost_df["Value"].cumsum()
        cost_df = cost_df[["Value"]]
        cost_df.columns = ["Cost"]
    return cost_df


def compute_usd(trading_dates: pd.DatetimeIndex, xlsx_file: str = CFG.FLOWDATA) -> pd.DataFrame:
    """Compute cumulative cost and cash specific to USD."""
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
    _cols_to_rename = {
        "SGD Deposit": "SGD_Deposit",
        "USD Deposit": "USD_Deposit",
    }
    _cols_to_rename.update(CFG.PFL["USD"].MAIN_SYMBOLS)
    cost_df.rename(columns=_cols_to_rename, inplace=True)
    cost_df = pd.DataFrame(index=trading_dates).join(cost_df)

    prices = _get_data(["USDSGD=X"], trading_dates, "USD", col="close")
    prices.rename(columns={"USDSGD=X": "USDSGD"}, inplace=True)

    cost_df = cost_df.join(prices[["USDSGD"]])
    cost_df.fillna(0, inplace=True)

    symbols = list(CFG.PFL["USD"].MAIN_SYMBOLS.values())
    for c in ["SGD_Deposit", "USD_Deposit", "USD-SGD"] + symbols:
        cost_df[c] = cost_df[c].cumsum()

    # Compute Cash
    units_df = df.query("Stock == 'USD-SGD'").pivot(index="Date", columns="Stock", values="Units")
    units_df = cost_df[["USDSGD"]].join(units_df)[["USD-SGD"]]
    units_df.fillna(0, inplace=True)
    units_df = units_df.cumsum()

    cost_df["Cash_SGD"] = cost_df["SGD_Deposit"] - cost_df["USD-SGD"]
    cost_df["Cash_USD"] = (
        cost_df["USD_Deposit"] + units_df["USD-SGD"] - cost_df[symbols].sum(axis=1)
    )
    cost_df["Cash"] = cost_df["Cash_USD"] + cost_df["Cash_SGD"] / cost_df["USDSGD"]
    # TODO: Cost fluctuating due to FX
    cost_df["Cost"] = (
        cost_df["USD_Deposit"] + units_df["USD-SGD"] + cost_df["Cash_SGD"] / cost_df["USDSGD"]
    )
    return cost_df[["USDSGD", "Cost", "Cash"]]


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
        df = df[df["Stock"].isin(list(CFG.PFL[sheet].MAIN_SYMBOLS.keys()))]
    df.columns = ["Date", "Type", "Stock", "Value"]

    # Compute gains
    gains_df = df[df["Type"] == gain_type].groupby(["Date"])[["Value"]].sum()
    return gains_df


def compute_gains(
    gain_type: str,
    trading_dates: pd.DatetimeIndex,
    sheet: str = "SGD",
    xlsx_file: str = CFG.FLOWDATA,
) -> pd.DataFrame:
    """Compute cumulative gains."""
    gains_df = agg_daily_gain(gain_type, sheet, xlsx_file)
    gains_df = gains_df.cumsum()
    gains_df = pd.DataFrame(index=trading_dates).join(gains_df)
    gains_df.iloc[0] = 0
    gains_df.ffill(inplace=True)
    return gains_df["Value"]


def get_portfolio(end_date, start_date, sheet, xlsx_file):
    portvals = compute_portvals(end_date, start_date, sheet, xlsx_file)
    trading_dates = portvals.index

    if sheet in ["SGD", "Fund", "SRS", "Bond"]:
        df = compute_cost(trading_dates, sheet, xlsx_file)
        df["Portfolio"] = portvals
    elif sheet == "USD":
        df = compute_usd(trading_dates, xlsx_file)
        df["Equity"] = portvals
        df["Portfolio"] = df["Equity"] + df["Cash"]
    else:
        raise NotImplementedError

    df["Div"] = compute_gains("Div", trading_dates, sheet, xlsx_file)
    df["Realised_Gain"] = compute_gains("Sell", trading_dates, sheet, xlsx_file)
    df["Paper_Gain"] = df["Portfolio"] - df["Cost"]
    df["Net_Gain"] = df["Paper_Gain"] + df["Realised_Gain"] + df["Div"]
    df = df.dropna()
    return df


if __name__ == "__main__":
    i = input("  Enter sheet (SGD=1, USD=2, Fund=3, SRS=4, Bond=5) (default=All): ")
    i = 0 if i == "" else int(i)
    if i not in range(6):
        raise ValueError("Invalid sheet. Number must be between 1 and 5.")

    end_date = datetime.today().date().isoformat()

    if i in [0, 1]:
        logger.info("Generating portfolio_sgd...")
        sgd_df = get_portfolio(end_date, "2015-03-23", "SGD", f"{CFG.SUMMARY_DIR}/aSummary.xlsx")
        sgd_df.to_csv(f"{CFG.SUMMARY_DIR}/portfolio_sgd.csv")

    if i in [0, 2]:
        logger.info("Generating portfolio_usd...")
        usd_df = get_portfolio(end_date, "2019-07-01", "USD", f"{CFG.SUMMARY_DIR}/aSummary.xlsx")
        usd_df.to_csv(f"{CFG.SUMMARY_DIR}/portfolio_usd.csv")

    if i in [0, 3]:
        logger.info("Generating portfolio_fund...")
        fund_df = get_portfolio(end_date, "2021-04-06", "Fund", f"{CFG.SUMMARY_DIR}/aSummary.xlsx")
        fund_df.to_csv(f"{CFG.SUMMARY_DIR}/portfolio_fund.csv")

    if i in [0, 4]:
        logger.info("Generating portfolio_srs...")
        srs_df = get_portfolio(end_date, "2019-02-01", "SRS", f"{CFG.SUMMARY_DIR}/aSummary.xlsx")
        srs_df.to_csv(f"{CFG.SUMMARY_DIR}/portfolio_srs.csv")

    if i in [0, 5]:
        logger.info("Generating portfolio_bond...")
        bond_df = get_portfolio(end_date, "2015-11-01", "Bond", f"{CFG.SUMMARY_DIR}/aSummary.xlsx")
        bond_df.to_csv(f"{CFG.SUMMARY_DIR}/portfolio_bond.csv")
