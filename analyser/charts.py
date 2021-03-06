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
from streamlit.elements import altair

from analyser.constants import str2days
from analyser.utils_charts import (
    get_ie_data,
    get_data,
    get_xlsx,
    get_ohlcv,
    rebase,
    pct_change,
)
try:
    from pm.config import DIRNAME, EQ_DICT, XLSX_FILE
    _dct = {
        "IWDA": "IWDA",
        "EIMI": "EIMI", 
    }
    _dct.update(EQ_DICT)
    EQ_DICT = _dct
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
def load_ie_data() -> pd.DataFrame:
    df = get_ie_data(start_date="1990-01-01", dirname=DIRNAME)
    df["10xReal_Earnings"] = 10 * df["Real_Earnings"]
    df["10xLong_IR"] = 10 * df["Long_IR"]
    return df[["Real_Price", "10xReal_Earnings", "CAPE", "10xLong_IR"]]


@st.cache
def load_data(dates: pd.DatetimeIndex, symbols: List[str], base_symbol="ES3.SI") -> pd.DataFrame:
    if "IWDA" not in symbols and "EIMI" not in symbols:
        return get_data(symbols, dates, base_symbol=base_symbol, dirname=DIRNAME)
    return get_xlsx(symbols, dates, base_symbol="IWDA", xlsx=XLSX_FILE)


def page_charts(today_date: datetime = date.today() - timedelta(days=1)) -> None:
    st.subheader("Shiller charts")
    df0 = load_ie_data()
    c1 = altair.generate_chart("line", df0[["Real_Price", "10xReal_Earnings"]]).properties(
        title="Index Plot",
        height=200,
        width=260,
    )
    c2 = altair.generate_chart("line", df0[["CAPE", "10xLong_IR"]]).properties(
        title="PE (CAPE) Plot",
        height=200,
        width=260,
    )
    st.altair_chart(alt.concat(c1, c2, columns=2), use_container_width=True)

    st.subheader("Stock charts")
    start_date = get_start_date(today_date, options=("3Y", "2Y", "1Y"))
    dates = pd.date_range(today_date - timedelta(days=365 * 2), today_date)

    # MSCI
    symbols = ["URTH", "EEM", "SPY", "ES3.SI"]
    colnames = ["MSCI World", "MSCI EM", "S&P500", "ES3"]
    df1 = load_data(dates, symbols, "SPY")
    df1.columns = colnames
    rebased_df1 = rebase(df1[df1.index >= start_date])
    chart1 = altair.generate_chart("line", rebased_df1).properties(
        title="MSCI",
        height=200,
        width=260,
    )

    # VIX
    symbols = ["^VIX"]
    colnames = ["VIX"]
    df2 = load_data(dates, symbols)[symbols]
    df2.columns = colnames
    chart2 = altair.generate_chart("line", df2[df2.index >= start_date]).properties(
        title="VIX",
        height=200,
        width=260,
    )

    st.altair_chart(alt.concat(chart1, chart2, columns=2), use_container_width=True)

    # etfs
    symbols = ["IWDA", "EIMI"]
    colnames = ["World", "EM"]
    df3a = load_data(dates, symbols)
    df3a.columns = colnames
    rebased_df3a = rebase(df3a[df3a.index >= start_date])
    chart3a = altair.generate_chart("line", rebased_df3a).properties(
        title="ETF",
        height=200,
        width=260,
    )
    symbols = ["O87.SI", "ES3.SI", "CLR.SI"]
    colnames = ["GLD", "ES3", "Lion-Phillip"]
    df3b = load_data(dates, symbols)
    df3b.columns = colnames
    rebased_df3b = rebase(df3b[df3b.index >= start_date])
    chart3b = altair.generate_chart("line", rebased_df3b).properties(
        title="ETF SGX",
        height=200,
        width=260,
    )
    st.altair_chart(alt.concat(chart3a, chart3b, columns=2), use_container_width=True)

    # industrial
    symbols = ["ES3.SI", "O5RU.SI", "A17U.SI", "J91U.SI", "BUOU.SI", "ME8U.SI", "M44U.SI"]
    colnames = ["ES3", "AA", "Ascendas", "ESR", "FLCT", "MIT", "MLT"]
    df4 = load_data(dates, symbols)
    df4.columns = colnames
    rebased_df4 = rebase(df4[df4.index >= start_date])
    chart4a = altair.generate_chart(
        "line",
        rebased_df4[["ES3", "Ascendas", "FLCT", "MIT", "MLT"]],
    ).properties(
        title="Industrial 1",
        height=200,
        width=260,
    )
    chart4b = altair.generate_chart(
        "line",
        rebased_df4[["ES3", "AA", "ESR"]],
    ).properties(
        title="Industrial 2",
        height=200,
        width=260,
    )
    st.altair_chart(alt.concat(chart4a, chart4b, columns=2), use_container_width=True)

    # retail
    symbols = ["ES3.SI", "C38U.SI", "J69U.SI", "N2IU.SI"]
    colnames = ["ES3", "CICT", "FCT", "MCT"]
    df5 = load_data(dates, symbols)
    df5.columns = colnames
    rebased_df5 = rebase(df5[df5.index >= start_date])
    chart5 = altair.generate_chart("line", rebased_df5).properties(
        title="Retail & Commercial",
        height=200,
        width=250,
    )

    # banks
    symbols = ["ES3.SI", "D05.SI", "O39.SI", "U11.SI"]
    colnames = ["ES3", "DBS", "OCBC", "UOB"]
    df6 = load_data(dates, symbols)
    df6.columns = colnames
    rebased_df6 = rebase(df6[df6.index >= start_date])
    chart6 = altair.generate_chart("line", rebased_df6).properties(
        title="Banks",
        height=200,
        width=250,
    )
    st.altair_chart(alt.concat(chart5, chart6, columns=2), use_container_width=True)


@st.cache(allow_output_mutation=True)
def load_ohlcv_data(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    # Load ohlc data
    df = get_ohlcv(symbol, dates, dirname=DIRNAME, xlsx=XLSX_FILE)

    # Apply technical analysis
    df = ta.add_volatility_ta(df, "high", "low", "close", fillna=False, colprefix="ta_")
    df = ta.add_momentum_ta(df, "high", "low", "close", "volume", fillna=False, colprefix="ta_")
    df = add_custom_trend(df, "close", fillna=False, colprefix="ta_")
    return df


def add_custom_trend(df: pd.DataFrame, close: str, fillna: bool, colprefix: str) -> pd.DataFrame:
    # MACD
    indicator_macd = ta.trend.MACD(close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna)
    df[f"{colprefix}trend_macd"] = indicator_macd.macd()
    df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
    df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    # SMAs
    df[f"{colprefix}trend_sma_fast"] = ta.trend.SMAIndicator(
        close=df[close], window=50, fillna=fillna).sma_indicator()
    df[f"{colprefix}trend_sma_slow"] = ta.trend.SMAIndicator(
        close=df[close], window=200, fillna=fillna).sma_indicator()
    df[f"{colprefix}trend_sma_10"] = ta.trend.SMAIndicator(
        close=df[close], window=10, fillna=fillna).sma_indicator()
    df[f"{colprefix}trend_sma_25"] = ta.trend.SMAIndicator(
        close=df[close], window=25, fillna=fillna).sma_indicator()
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
        alt.Y("low:Q", title="Price", scale=alt.Scale(zero=False)),
        alt.Y2("high:Q")
    )
    bar = base.mark_bar().encode(
        alt.Y("open:Q"),
        alt.Y2("close:Q")
    )
    chart = rule + bar
    for col in cols:
        line = alt.Chart(source).mark_line(color="gray").encode(
            alt.X("date:T"),
            alt.Y(col),
            tooltip=["date", col],
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

    col0, col1 = st.beta_columns(2)
    select_days = col0.selectbox("Lookback period", ["1M", "2M", "3M", "6M"], 2)
    select_days = str2days[select_days]
    select_ta = col1.selectbox("Add TA", ["Bollinger", "SMA", "RSI"])

    source = df.iloc[-select_days:].reset_index()
    st.altair_chart(chart_candlestick(source, cols=ta_type[select_ta]["price"]), use_container_width=True)

    col2, col3 = st.beta_columns(2)
    select_days2 = col2.selectbox("Select period", ["6M", "9M", "1Y", "2Y"], 2)
    select_days2 = str2days[select_days2]
    select_ta2 = col3.selectbox("Select TA", ["Bollinger", "SMA", "RSI", "MACD", "Momentum"])
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
    st.write("Target value min: `{0:.2f}%`; max: `{1:.2f}%`; mean: `{2:.2f}%`; std: `{3:.2f}`".format(
        np.min(df1["y"]), np.max(df1["y"]), np.mean(df1["y"]), np.std(df1["y"])))

    # Univariate Analysis
    st.subheader("Correlation coefficient ta features and target column")
    x_cols = [col for col in df.columns if col != "y" and col.startswith("ta_")]
    df2 = df[x_cols + ["y"]].copy().dropna()
    values = [np.corrcoef(df2[col], df2["y"])[0, 1] for col in x_cols]
    st.bar_chart(data=pd.DataFrame(data=values, index=x_cols))
