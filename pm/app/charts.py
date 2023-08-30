from typing import List, Optional
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.elements import legacy_altair as altair

from analyser.data import get_data, rebase
from pm import CFG
from pm.app.utils import get_start_date


@st.cache_data
def _load_pe_data(start_date: str = "1990-01-01") -> pd.DataFrame:
    """Shiller monthly PE data downloaded from nasdaq-data-link."""
    df = pd.read_csv(f"{CFG.SUMMARY_DIR}/pe_data.csv")
    df["Date"] = pd.to_datetime(df["Date"].astype(str))
    df.set_index("Date", inplace=True)
    if start_date is not None:
        df = df[df.index >= start_date]
    return df


# @st.cache_data
# def _load_ie_data(start_date: str = "1990-01-01") -> pd.DataFrame:
#     """Data downloaded from http://www.econ.yale.edu/~shiller/data.htm."""
#     df = pd.read_excel(f"{CFG.SUMMARY_DIR}/ie_data.xls", sheet_name="Data", skiprows=7)
#     df.drop(["Fraction", "Unnamed: 13", "Unnamed: 15"], axis=1, inplace=True)
#     df.columns = [
#         "Date",
#         "S&P500",
#         "Dividend",
#         "Earnings",
#         "CPI",
#         "Long_IR",
#         "Real_Price",
#         "Real_Dividend",
#         "Real_TR_Price",
#         "Real_Earnings",
#         "Real_TR_Scaled_Earnings",
#         "CAPE",
#         "TRCAPE",
#         "Excess_CAPE_Yield",
#         "Mth_Bond_TR",
#         "Bond_RTR",
#         "10Y_Stock_RR",
#         "10Y_Bond_RR",
#         "10Y_Excess_RR",
#     ]
#     df["Date"] = pd.to_datetime(df["Date"].astype(str))
#     df.set_index("Date", inplace=True)
#     df = df.iloc[:-1]
#     if start_date is not None:
#         df = df[df.index >= start_date]

#     df["10xReal_Earnings"] = 10 * df["Real_Earnings"]
#     df["10xLong_IR"] = 10 * df["Long_IR"]
#     return df[["Real_Price", "10xReal_Earnings", "CAPE", "10xLong_IR"]]


def _get_chart(
    dates: pd.DatetimeIndex,
    symbols: List[str],
    symbol_names: Optional[List[str]] = None,
    base_symbol: str = "ES3.SI",
    title: str = "",
):
    df = get_data(symbols, dates, base_symbol=base_symbol, dirname=CFG.DATA_DIR)[
        symbols
    ]
    df = rebase(df)
    if symbol_names is not None:
        df.columns = symbol_names
    chart = altair.generate_chart("line", df).properties(
        title=title,
        height=200,
        width=260,
    )
    return chart


def page_charts(last_date: date) -> None:
    # df0 = _load_ie_data()
    # c1 = altair.generate_chart(
    #     "line", df0[["Real_Price", "10xReal_Earnings"]]
    # ).properties(
    #     title="Index",
    #     height=200,
    #     width=260,
    # )
    # c2 = altair.generate_chart("line", df0[["CAPE", "10xLong_IR"]]).properties(
    #     title="CAPE",
    #     height=200,
    #     width=260,
    # )
    # st.altair_chart(alt.concat(c1, c2, columns=2), use_container_width=True)

    start_date = get_start_date(last_date, options=("1Y", "2Y", "3Y"))
    dates = pd.date_range(start_date, last_date)

    df0 = _load_pe_data()
    chart0 = altair.generate_chart("line", df0[["CAPE"]]).properties(
        title="Shiller PE",
        height=200,
        width=260,
    )

    df1 = get_data(["^VIX"], dates, base_symbol="^VIX", dirname=CFG.DATA_DIR)["^VIX"]
    df1.columns = ["VIX"]
    chart1 = altair.generate_chart("line", df1).properties(
        title="VIX",
        height=200,
        width=260,
    )
    st.altair_chart(alt.concat(chart0, chart1, columns=2), use_container_width=True)

    st.subheader("Stock charts")
    # MSCI
    chart2 = _get_chart(
        dates,
        ["URTH", "EEM", "IEUR", "SPY", "ES3.SI"],
        symbol_names=["MSCI World", "MSCI EM", "MSCI EUR", "S&P500", "ES3.SI"],
        base_symbol="SPY",
        title="MSCI",
    )
    st.altair_chart(chart2, use_container_width=True)

    # ETFs
    chart3 = _get_chart(
        dates,
        ["IWDA.L", "EIMI.L"],
        base_symbol="IWDA.L",
        title="ETF",
    )
    st.altair_chart(chart3, use_container_width=True)

    # banks
    chart4 = _get_chart(
        dates,
        ["ES3.SI", "D05.SI", "O39.SI", "U11.SI"],
        symbol_names=["ES3", "DBS", "OCBC", "UOB"],
        title="Banks",
    )
    st.altair_chart(chart4, use_container_width=True)

    # industrial
    chart5 = _get_chart(
        dates,
        ["ES3.SI", "O5RU.SI", "A17U.SI", "BUOU.SI", "ME8U.SI", "M44U.SI"],
        symbol_names=["ES3", "AA", "Ascendas", "FLCT", "MIT", "MLT"],
        title="Industrial",
    )
    st.altair_chart(chart5, use_container_width=True)

    # Commercial
    chart6 = _get_chart(
        dates,
        ["ES3.SI", "C38U.SI", "J69U.SI", "N2IU.SI"],
        symbol_names=["ES3", "CICT", "FCT", "MPACT"],
        title="Commercial",
    )
    st.altair_chart(chart6, use_container_width=True)

    # Others
    chart7 = _get_chart(
        dates,
        ["GOTO.JK", "GRAB"],
        symbol_names=["GoTo", "Grab"],
        title="Tech",
    )
    st.altair_chart(chart7, use_container_width=True)
