from datetime import date

import pandas as pd
import streamlit as st

from analyser.data import get_data, rebase
from pm import CFG
from pm.app.utils import get_start_date
from pm.ta import compute_trend


@st.cache_data
def _load_ie_data():
    """Data downloaded from https://shillerdata.com/."""
    from pm.data import load_ie_data

    df = load_ie_data(start_date="1946-01-01")
    df, _, _ = compute_trend(df["CAPE"])
    return df


@st.cache_data
def _load_pe_data():
    """Shiller monthly PE data downloaded from nasdaq-data-link."""
    from pm.data import load_pe_data

    return load_pe_data(start_date="1990-01-01")


def _linechart(df: pd.DataFrame):
    """A simple streamlit.elements.vega_charts.generate_chart wrapper."""
    from streamlit.elements.vega_charts import ChartType, generate_chart

    chart, _ = generate_chart(chart_type=ChartType.LINE, data=df)
    return chart


def _get_chart(
    dates: pd.DatetimeIndex,
    symbols: list[str],
    symbol_names: list[str] | None = None,
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
    df0 = _load_ie_data()
    chart0 = _linechart(df0).properties(title="Shiller PE")
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
