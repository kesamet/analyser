"""
Charting
"""
from datetime import date, datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import ta
import altair as alt
import streamlit as st
from streamlit.elements import legacy_altair as altair

from analyser.constants import str2days
from analyser.utils_charts import (
    get_data,
    get_data_ohlcv,
    get_ie_data,
    pct_change,
    rebase,
)

try:
    from pm.config import DIRNAME, EQ_DICT
except ModuleNotFoundError:
    DIRNAME = "samples"
    EQ_DICT = {
        "MSCI ACWI": "ACWI",
        "MSCI World": "URTH",
    }


def get_start_date(
    today_date: datetime,
    start_date="2015-01-01",
    options=("YTD", "1M", "6M", "1Y", "2Y", "3Y", "All time"),
) -> str:
    select_range = st.selectbox("Select time range", options)
    if select_range[-1] == "Y":
        _yr = today_date.year - int(select_range[:-1])
        return today_date.replace(year=_yr).strftime("%Y-%m-%d")
    elif select_range[-1] == "M":
        _mths = int(select_range[:-1])
        return (today_date - timedelta(days=30 * _mths)).strftime("%Y-%m-%d")
    elif select_range == "YTD":
        return today_date.replace(month=1, day=1).strftime("%Y-%m-%d")
    elif select_range == "MTD":
        return today_date.replace(day=1).strftime("%Y-%m-%d")
    return start_date


@st.cache
def load_ie_data(start_date="1990-01-01") -> pd.DataFrame:
    df = get_ie_data(start_date, dirname=DIRNAME)
    return df
    # df["10xReal_Earnings"] = 10 * df["Real_Earnings"]
    # df["10xLong_IR"] = 10 * df["Long_IR"]
    # return df[["Real_Price", "10xReal_Earnings", "CAPE", "10xLong_IR"]]


def _get_chart(start_date, dates, symbols, symbol_names=None, base_symbol="ES3.SI", title=""):
    df = get_data(symbols, dates, base_symbol=base_symbol, dirname=DIRNAME)[symbols]
    df1 = rebase(df[df.index >= start_date].copy())
    if symbol_names is not None:
        df1.columns = symbol_names
    chart = altair.generate_chart("line", df1).properties(
        title=title,
        height=200,
        width=260,
    )
    return chart


def page_charts(today_date: datetime = date.today() - timedelta(days=1)) -> None:
    start_date = get_start_date(today_date, options=("3Y", "2Y", "1Y"))
    dates = pd.date_range(today_date - timedelta(days=365 * 2), today_date)

    df0 = load_ie_data()
    chart0 = altair.generate_chart("line", df0[["CAPE"]]).properties(
        title="Shiller PE (CAPE) Plot",
        height=200,
        width=260,
    )
    st.altair_chart(chart0, use_container_width=True)
    # c1 = altair.generate_chart(
    #     "line", df0[["Real_Price", "10xReal_Earnings"]]
    # ).properties(
    #     title="Index Plot",
    #     height=200,
    #     width=260,
    # )
    # c2 = altair.generate_chart("line", df0[["CAPE", "10xLong_IR"]]).properties(
    #     title="PE (CAPE) Plot",
    #     height=200,
    #     width=260,
    # )
    # st.altair_chart(alt.concat(c1, c2, columns=2), use_container_width=True)

    df1 = get_data(["^VIX"], dates, base_symbol="^VIX", dirname=DIRNAME)["^VIX"]
    df1.columns = ["VIX"]
    chart1 = altair.generate_chart("line", df1).properties(
        title="VIX",
        height=200,
        width=260,
    )
    st.altair_chart(chart1, use_container_width=True)

    st.subheader("Stock charts")
    # MSCI
    chart2 = _get_chart(
        start_date,
        dates,
        ["URTH", "EEM", "SPY", "ES3.SI"],
        symbol_names=["MSCI World", "MSCI EM", "S&P500", "ES3.SI"],
        base_symbol="SPY",
        title="MSCI",
    )
    st.altair_chart(chart2, use_container_width=True)

    # ETFs
    chart3 = _get_chart(
        start_date,
        dates,
        ["IWDA.L", "EIMI.L"],
        base_symbol="IWDA.L",
        title="ETF",
    )
    st.altair_chart(chart3, use_container_width=True)

    # banks
    chart4 = _get_chart(
        start_date,
        dates,
        ["ES3.SI", "D05.SI", "O39.SI", "U11.SI"],
        symbol_names=["ES3", "DBS", "OCBC", "UOB"],
        title="Banks",
    )
    st.altair_chart(chart4, use_container_width=True)

    # industrial
    chart5 = _get_chart(
        start_date,
        dates,
        ["ES3.SI", "O5RU.SI", "A17U.SI", "BUOU.SI", "ME8U.SI", "M44U.SI"],
        symbol_names=["ES3", "AA", "Ascendas", "FLCT", "MIT", "MLT"],
        title="Industrial",
    )
    st.altair_chart(chart5, use_container_width=True)

    # retail
    chart6 = _get_chart(
        start_date,
        dates,
        ["ES3.SI", "C38U.SI", "J69U.SI", "N2IU.SI"],
        symbol_names=["ES3", "CICT", "FCT", "MPACT"],
        title="Retail/Commercial",
    )
    st.altair_chart(chart6, use_container_width=True)


@st.cache(allow_output_mutation=True)
def load_ohlcv_data(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    # Load ohlc data
    if symbol in ["IWDA.L", "EIMI.L"]:
        base_symbol = "IWDA.L"
    else:
        base_symbol = "ES3.SI"
    df = get_data_ohlcv(symbol, dates, base_symbol=base_symbol, dirname=DIRNAME)

    # Apply technical analysis
    df = ta.add_volatility_ta(df, "high", "low", "close", fillna=False, colprefix="ta_")
    df = ta.add_momentum_ta(
        df, "high", "low", "close", "volume", fillna=False, colprefix="ta_"
    )
    df = add_custom_trend(df, "close", fillna=False, colprefix="ta_")
    return df


def add_custom_trend(
    df: pd.DataFrame, close: str, fillna: bool, colprefix: str
) -> pd.DataFrame:
    # MACD
    indicator_macd = ta.trend.MACD(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}trend_macd"] = indicator_macd.macd()
    df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
    df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    # SMAs
    df[f"{colprefix}trend_sma_fast"] = ta.trend.SMAIndicator(
        close=df[close], window=50, fillna=fillna
    ).sma_indicator()
    df[f"{colprefix}trend_sma_slow"] = ta.trend.SMAIndicator(
        close=df[close], window=200, fillna=fillna
    ).sma_indicator()
    df[f"{colprefix}trend_sma_10"] = ta.trend.SMAIndicator(
        close=df[close], window=10, fillna=fillna
    ).sma_indicator()
    df[f"{colprefix}trend_sma_25"] = ta.trend.SMAIndicator(
        close=df[close], window=25, fillna=fillna
    ).sma_indicator()
    return df


def chart_candlestick(source: pd.DataFrame, cols: List = []) -> None:
    """Candlestick chart."""
    base = alt.Chart(source).encode(
        alt.X("date:T"),
        color=alt.condition(
            "datum.open <= datum.close",
            alt.value("#06982d"),
            alt.value("#ae1325"),
        ),
    )
    rule = base.mark_rule().encode(
        alt.Y("low:Q", title="Price", scale=alt.Scale(zero=False)), alt.Y2("high:Q")
    )
    bar = base.mark_bar().encode(alt.Y("open:Q"), alt.Y2("close:Q"))
    chart = rule + bar
    for col in cols:
        line = (
            alt.Chart(source)
            .mark_line(color="gray")
            .encode(
                alt.X("date:T"),
                alt.Y(col),
                tooltip=["date", alt.Tooltip(col, format=".4f")],
            )
        )
        chart += line
    return chart


def page_ta(today_date: datetime = date.today() - timedelta(days=1)) -> None:
    """Technical analysis page."""
    ta_type = {
        "Bollinger": {
            "price": ["ta_volatility_bbm", "ta_volatility_bbh", "ta_volatility_bbl"],
            "ind": ["ta_volatility_bbhi", "ta_volatility_bbli"],
        },
        "SMA": {
            "price": ["ta_trend_sma_fast", "ta_trend_sma_slow"],
        },
        "RSI": {
            "price": ["ta_trend_sma_10", "ta_trend_sma_25"],
            "ind": ["ta_momentum_rsi"],
        },
        "MACD": {
            "price": [],
            "ind": ["ta_trend_macd", "ta_trend_macd_signal", "ta_trend_macd_diff"],
        },
        "Momentum": {
            "price": [],
            "ind": ["ta_momentum_tsi", "ta_momentum_rsi", "ta_momentum_stoch"],
        },
    }

    select_eq = st.selectbox("Select equity", list(EQ_DICT.keys()))
    dates = pd.date_range(today_date - timedelta(days=800), today_date)
    df = load_ohlcv_data(EQ_DICT[select_eq], dates)

    col0, col1 = st.columns(2)
    select_days = col0.selectbox("Lookback period", ["1M", "2M", "3M", "6M"], 2)
    select_days = str2days[select_days]
    select_ta = col1.selectbox("Add TA", ["Bollinger", "SMA", "RSI"])

    source = df.iloc[-select_days:].reset_index()
    st.altair_chart(
        chart_candlestick(source, cols=ta_type[select_ta]["price"]),
        use_container_width=True,
    )

    col2, col3 = st.columns(2)
    select_days2 = col2.selectbox("Select period", ["6M", "9M", "1Y", "2Y"], 2)
    select_days2 = str2days[select_days2]
    select_ta2 = col3.selectbox(
        "Select TA", ["Bollinger", "SMA", "RSI", "MACD", "Momentum"]
    )
    st.line_chart(df[["close"] + ta_type[select_ta2]["price"]].iloc[-select_days2:])
    if ta_type[select_ta2].get("ind") is not None:
        st.line_chart(df[ta_type[select_ta2]["ind"]].iloc[-select_days2:])

    # Prepare target: X Periods Return
    select_periods = st.slider("Select periods", 7, 28, 14)
    df["y"] = pct_change(df["close"], select_periods) * 100

    st.subheader(f"{select_periods}-day Returns")
    df1 = df[["y"]].copy().dropna()
    st.area_chart(df1["y"])

    st.subheader("Target Histogram")
    hist_values, hist_indexes = np.histogram(df1["y"], bins=np.arange(-10, 10, 0.5))
    st.bar_chart(pd.DataFrame(data=hist_values, index=hist_indexes[0:-1]))
    st.write(
        "Target value min: `{0:.2f}%`; max: `{1:.2f}%`; mean: `{2:.2f}%`; std: `{3:.2f}`".format(
            np.min(df1["y"]), np.max(df1["y"]), np.mean(df1["y"]), np.std(df1["y"])
        )
    )

    # Univariate Analysis
    st.subheader("Correlation coefficient ta features and target column")
    x_cols = [col for col in df.columns if col != "y" and col.startswith("ta_")]
    df2 = df[x_cols + ["y"]].copy().dropna()
    values = [np.corrcoef(df2[col], df2["y"])[0, 1] for col in x_cols]
    st.bar_chart(data=pd.DataFrame(data=values, index=x_cols))
