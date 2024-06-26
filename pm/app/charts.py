from typing import List, Optional
from datetime import date

import pandas as pd
import streamlit as st

from analyser.data import get_data, rebase
from pm import CFG
from pm.app.utils import get_start_date


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


@st.cache_data
def _load_pe_data(start_date: str = "1990-01-01") -> pd.DataFrame:
    """Shiller monthly PE data downloaded from nasdaq-data-link."""
    df = pd.read_csv(f"{CFG.SUMMARY_DIR}/pe_data.csv")
    df["Date"] = pd.to_datetime(df["Date"].astype(str))
    df.set_index("Date", inplace=True)
    if start_date is not None:
        df = df[df.index >= start_date]
    return df


def _linechart(df: pd.DataFrame):
    """A simple streamlit.elements.vega_charts.generate_chart wrapper."""
    from streamlit.elements.vega_charts import ChartType, generate_chart

    chart, _ = generate_chart(chart_type=ChartType.LINE, data=df)
    return chart


def _get_chart(
    dates: pd.DatetimeIndex,
    symbols: List[str],
    symbol_names: Optional[List[str]] = None,
    base_symbol: str = "ES3.SI",
    to_rebase: bool = True,
) -> None:
    df = get_data(symbols, dates, base_symbol=base_symbol, dirname=CFG.DATA_DIR)
    df = df[symbols]
    if to_rebase:
        df = rebase(df)
    if symbol_names is not None:
        df.columns = symbol_names
    st.line_chart(df)


def page_charts(last_date: date) -> None:
    df0 = _load_pe_data()
    chart0 = _linechart(df0[["CAPE"]]).properties(title="Shiller PE")
    st.altair_chart(chart0, use_container_width=True)

    start_date = get_start_date(last_date, options=("1Y", "2Y", "3Y"))
    dates = pd.date_range(start_date, last_date)

    with st.expander("VIX"):
        _get_chart(
            dates,
            ["^VIX"],
            symbol_names=["VIX"],
            base_symbol="^VIX",
            to_rebase=False,
        )

    # with st.expander("MSCI"):
    #     _get_chart(
    #         dates,
    #         ["URTH", "EEM", "IEUR", "SPY", "ES3.SI"],
    #         symbol_names=["MSCI World", "MSCI EM", "MSCI EUR", "S&P500", "ES3.SI"],
    #         base_symbol="SPY",
    #     )

    with st.expander("ETFs"):
        _get_chart(
            dates,
            ["VWRA.L", "IWDA.L", "EIMI.L", "SPY", "ES3.SI"],
            base_symbol="IWDA.L",
        )

    with st.expander("Banks"):
        _get_chart(
            dates,
            ["ES3.SI", "D05.SI", "O39.SI", "U11.SI"],
            symbol_names=["ES3", "DBS", "OCBC", "UOB"],
        )

    with st.expander("Industrial"):
        _get_chart(
            dates,
            ["ES3.SI", "O5RU.SI", "A17U.SI", "BUOU.SI", "ME8U.SI", "M44U.SI"],
            symbol_names=["ES3", "AA", "Ascendas", "FLCT", "MIT", "MLT"],
        )

    with st.expander("Commercial"):
        _get_chart(
            dates,
            ["ES3.SI", "C38U.SI", "J69U.SI", "N2IU.SI"],
            symbol_names=["ES3", "CICT", "FCT", "MPACT"],
        )

    with st.expander("Tech"):
        _get_chart(
            dates,
            ["GOTO.JK", "GRAB"],
            symbol_names=["GoTo", "Grab"],
        )
