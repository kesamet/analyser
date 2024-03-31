from datetime import date, timedelta

import pandas as pd
import streamlit as st

from analyser.app.constants import str2days
from analyser.data import get_data

import pm.ta as ta
from pm import CFG


@st.cache_data
def _table_trend_by_days(last_date: date, days: int) -> pd.DataFrame:
    symbols = list(CFG.SYMBOLS.values())
    dates = pd.date_range(last_date - timedelta(days=days), last_date)

    results = list()
    df1 = get_data(symbols, dates, col="adjclose", dirname=CFG.DATA_DIR)
    for symbol in symbols:
        _, level, res, _, grad, pred = ta.linearfit(df1[symbol])
        results.append(
            [
                symbol,
                df1[symbol].iloc[-1],
                level,
                grad * 1e3,
                pred - 2 * res,
                pred - res,
                pred,
                pred + res,
                pred + 2 * res,
            ]
        )

    results = pd.DataFrame(
        results,
        columns=["symbol", "close", "level", "grad", "p0", "p25", "p50", "p75", "p100"],
    )
    results.index = symbols
    results["symbol"] = list(CFG.SYMBOLS.keys())
    return results.sort_values("level")


@st.cache_data
def _table_trend_by_symbol(last_date: date, symbol: str) -> pd.DataFrame:
    dates = pd.date_range(last_date - timedelta(days=730), last_date)
    df = get_data([symbol], dates, col="adjclose", dirname=CFG.DATA_DIR)

    periods = ["3M", "6M", "9M", "1Y", "2Y"]
    results = list()
    for period in periods:
        date = last_date - timedelta(days=str2days[period])
        df1 = df[df.index.date >= date]
        _, level, res, _, grad, pred = ta.linearfit(df1[symbol])
        results.append(
            [
                level,
                grad * 1e3,
                pred - 2 * res,
                pred - res,
                pred,
                pred + res,
                pred + 2 * res,
            ]
        )

    results = pd.DataFrame(
        results, columns=["pct_level", "grad", "p0", "p25", "p50", "p75", "p100"]
    )
    results.index = periods
    return results


@st.cache_data
def _get_trend_df(last_date: date, days: int, symbol: str) -> pd.DataFrame:
    dates = pd.date_range(last_date - timedelta(days=days), last_date)
    return ta.compute_trend(dates, symbol)


def page_trend(last_date: date) -> None:
    """Trend page."""
    st.header("By days")
    c0, _, _ = st.columns(3)
    s1 = c0.selectbox("Select lookback period", ["3M", "6M", "9M", "1Y", "2Y"], 3)
    select_days1 = str2days[s1]
    df1 = _table_trend_by_days(last_date, select_days1)

    def highlight(x):
        if x < 25:
            return "background-color: #82E0AA"  # green
        elif x > 75:
            return "background-color: #F1948A"  # red
        return ""

    st.subheader("Main")
    st.dataframe(
        df1[df1.index.isin(CFG.TREND_SYMBOLS)]
        .style.format(precision=3)
        .map(lambda x: "color: red" if x < 0 else "", subset=["grad"])
        .map(highlight, subset=["level"]),
        height=400,
    )

    st.subheader("All")
    st.dataframe(
        df1.style.format(precision=3)
        .map(lambda x: "color: red" if x < 0 else "", subset=["grad"])
        .map(highlight, subset=["level"]),
    )

    st.header("By equity")
    cols = st.columns((2, 1))
    select_eq = cols[0].selectbox("Select equity", list(CFG.SYMBOLS.keys()))
    symbol = CFG.SYMBOLS[select_eq]

    s2 = cols[1].selectbox("Select period", ["3M", "6M", "9M", "1Y", "2Y"], 3)
    select_days2 = str2days[s2]

    df2 = _table_trend_by_symbol(last_date, symbol)
    st.table(df2)

    df, level, grad = _get_trend_df(last_date, select_days2, symbol)
    st.text(
        f"""
        Price = {df[symbol].iloc[-1]:.2f}
        Level = {level:.1f}%
        Gradient = {grad * 1e3:.3f}
        """
    )
    st.line_chart(df)
