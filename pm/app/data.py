from datetime import date
from dateutil import parser

import pandas as pd
import streamlit as st

from analyser.data import get_data, rebase, annualise
from analyser.plots import barchart

import pm.portfolio as F
from pm import CFG
from pm.app.utils import get_start_date


def _load_portfolio(filename: str):
    df = pd.read_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.dropna()
    return df


@st.cache_data
def get_portfolio(sheet: str) -> pd.DataFrame:
    if sheet == "USD":
        return _load_portfolio("data/summary/portfolio_usd.csv")
    elif sheet == "SRS":
        return _load_portfolio("data/summary/portfolio_srs.csv")
    elif sheet == "Fund":
        return _load_portfolio("data/summary/portfolio_fund.csv")
    elif sheet == "Bond":
        return _load_portfolio("data/summary/portfolio_bond.csv")
    elif sheet == "SGD":
        df = _load_portfolio("data/summary/portfolio_sgd.csv")
        _df0 = get_data(["ES3.SI"], df.index, col="adjclose", dirname=CFG.DATA_DIR)
        df["bm"] = _df0["ES3.SI"]
        return df
    else:
        raise NotImplementedError


def subset_portfolio(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    try:
        _start_date = df.index[df.index < start_date][-1]
    except IndexError:
        _start_date = start_date

    # subset by start_date and remove cost = 0
    subset_df = df[df.index >= _start_date].query("Cost > 0").copy()

    if len(subset_df) == 0:
        return None
    return subset_df


@st.cache_data
def rebase_table(subset_df: pd.DataFrame, sheet: str, currency: str | None = None) -> pd.DataFrame:
    df = subset_df.copy()

    if sheet != currency and currency == "SGD":
        if f"{sheet}SGD" in df.columns:
            fx = df[f"{sheet}SGD"]
        else:
            fx = 1 / df[f"SGD{sheet}"]

        for c in [
            "Cost",
            "Portfolio",
            "Div",
            "Realised_Gain",
            "Paper_Gain",
            "Net_Gain",
        ]:
            df[c] *= fx
        if "Cash" in df.columns:
            df["Cash"] *= fx
        if "Equity" in df.columns:
            df["Equity"] *= fx

    df["Div"] -= df["Div"].iloc[0]
    df["Realised_Gain"] -= df["Realised_Gain"].iloc[0]
    df["Paper_Gain"] -= df["Paper_Gain"].iloc[0]
    df["Net_Gain"] -= df["Net_Gain"].iloc[0]

    df["Net_Yield"] = df["Net_Gain"] / df["Cost"]
    df["Div_Yield"] = df["Div"] / df["Cost"]

    if "bm" in df.columns:
        df["Net_Yield_bm"] = rebase(df["bm"]) - 1
    return df


@st.cache_data
def get_whatif_portfolio(start_date: date, end_date: date) -> pd.DataFrame:
    df = pd.read_excel(
        CFG.FLOWDATA,
        sheet_name="SGD Summary",
        usecols=["Yahoo Quote", "Units"],
    ).query("Units > 0")

    symbols = df["Yahoo Quote"].tolist()
    units = df["Units"].tolist()
    port_val = F.portfolio_const(start_date, end_date, symbols, units=units)
    return F.assess_portfolio(port_val)


@st.cache_data
def get_overall_portfolio(inclu_bond: bool = True) -> pd.DataFrame:
    sgd_df = get_portfolio("SGD")
    usd_df = get_portfolio("USD")
    fund_df = get_portfolio("Fund")
    srs_df = get_portfolio("SRS")
    last = {
        "SGD": sgd_df["Portfolio"].iloc[-1],
        "USD": usd_df["Portfolio"].iloc[-1] * usd_df["USDSGD"].iloc[-1],
        "Fund": fund_df["Portfolio"].iloc[-1],
        "SRS": srs_df["Portfolio"].iloc[-1],
    }
    if inclu_bond:
        bond_df = get_portfolio("Bond")
        last["Bond"] = bond_df["Portfolio"].iloc[-1]

    dfs = [sgd_df, fund_df, srs_df]

    # Use trading days of "USDSGD=X"
    dates = pd.date_range(srs_df.index[0], srs_df.index[-1])
    df = pd.DataFrame(index=dates)
    for c in ["Div", "Realised_Gain", "Paper_Gain", "Cost"]:
        tmp = pd.DataFrame(index=dates)

        for i, df0 in enumerate(dfs):
            tmp1 = df0[[c]].copy()
            tmp1.columns = [f"x{i}"]
            tmp = tmp.join(tmp1)

        tmp2 = usd_df[[c]].copy()
        tmp2.columns = ["y2"]
        tmp2["y2"] = tmp2["y2"] * usd_df["USDSGD"]
        tmp = tmp.join(tmp2)

        if inclu_bond:
            tmp3 = bond_df[[c]].copy()
            tmp3.columns = ["y3"]
            tmp = tmp.join(tmp3)

        tmp.ffill(inplace=True)
        tmp.fillna(0, inplace=True)
        tmp[c] = tmp.sum(axis=1)
        df = df.join(tmp[[c]])

    # fill_missing_values(df)
    df["Net_Gain"] = df["Div"] + df["Realised_Gain"] + df["Paper_Gain"]
    df["Portfolio"] = df["Paper_Gain"] + df["Cost"]
    return df, last


@st.cache_data
def sum_by_time(sheet: str, last_date: date, timeunits: str) -> pd.DataFrame:
    def filter_before_today(df):
        return df[df.index <= last_date.isoformat()]

    df = filter_before_today(F.agg_daily_cost(sheet, CFG.FLOWDATA))
    df = df.resample(timeunits).sum()
    df.columns = ["cost"]

    div_df = filter_before_today(F.agg_daily_gain("Div", sheet, CFG.FLOWDATA))
    if not div_df.empty:
        div_df = div_df.resample(timeunits).sum()
        div_df.columns = ["div"]
        df = df.join(div_df, how="outer")

    gain_df = filter_before_today(F.agg_daily_gain("Sell", sheet, CFG.FLOWDATA))
    if not gain_df.empty:
        gain_df = gain_df.resample(timeunits).sum()
        gain_df.columns = ["gain"]
        df = df.join(gain_df, how="outer")

    df.fillna(0, inplace=True)
    return df


def page_data(last_date: date) -> None:
    """Portfolio page."""
    tentative_start_date = get_start_date(last_date)

    sheets = ["Overall", "Overall Equity", "SGD", "USD", "Fund", "SRS", "Bond"]
    tabs = st.tabs(sheets)

    for tab, sheet in zip(tabs, sheets):
        with tab:
            tab_portfolio(last_date, sheet, tentative_start_date)


def tab_portfolio(last_date: date, sheet: str, tentative_start_date: date) -> None:
    """Portfolio page."""
    if sheet == "Overall":
        df, _ = get_overall_portfolio()
    elif sheet == "Overall Equity":
        df, _ = get_overall_portfolio(inclu_bond=False)
    else:
        df = get_portfolio(sheet)

    _df = subset_portfolio(df, tentative_start_date)
    if _df is None:
        st.warning("No data found")
        return

    if sheet in ["USD"]:
        currency = st.radio("Currency", [sheet, "SGD"])
    else:
        currency = None

    subset_df = rebase_table(_df, sheet, currency)

    st.write(subset_df.index[-1].strftime("Last updated on `%Y-%m-%d`"))

    start_date = subset_df.index[0].date().isoformat()
    curr_val = subset_df["Portfolio"].iloc[-1]
    cost = subset_df["Cost"].iloc[-1] - subset_df["Cost"].iloc[0]

    net_gain = subset_df["Net_Gain"].iloc[-1]
    div = subset_df["Div"].iloc[-1]
    realised_gain = subset_df["Realised_Gain"].iloc[-1]
    paper_gain = subset_df["Paper_Gain"].iloc[-1]

    years = (last_date - parser.parse(start_date).date()).days / 365.25
    net_yield = subset_df["Net_Yield"].iloc[-1]  # compute_dietz_ret(subset_df)
    div_yield = subset_df["Div_Yield"].iloc[-1]
    ann_div_yield = annualise(div_yield, years)

    c0, c1, c2 = st.columns(3)
    c0.metric("Portfolio Value", f"{curr_val:,.2f}", f"{net_yield:.2%}")
    c1.metric("Investment", f"{cost:,.2f}")
    c2.metric("Net Gain", f"{net_gain:,.2f}")
    d0, d1, d2 = st.columns(3)
    d0.metric("Dividends", f"{div:,.2f}", f"{ann_div_yield:.2%}")
    d1.metric("Realised Gain", f"{realised_gain:,.2f}")
    d2.metric("Paper Gain", f"{paper_gain:,.2f}")

    if sheet in ["Overall", "Overall Equity"]:
        st.line_chart(subset_df[["Portfolio", "Cost"]])
        st.line_chart(subset_df[["Net_Gain", "Paper_Gain", "Div"]])
    elif sheet == "USD":
        st.line_chart(subset_df[["Portfolio", "Cost", "Equity", "Cash"]])
        st.line_chart(subset_df[["Paper_Gain"]])
    elif sheet == "SGD":
        st.line_chart(subset_df[["Benchmark", "Portfolio", "Cost"]])
        st.line_chart(subset_df[["Net_Gain", "Paper_Gain", "Div"]])
    else:
        st.line_chart(subset_df[["Portfolio", "Cost"]])
        st.line_chart(subset_df[["Net_Gain", "Paper_Gain", "Div"]])

    if "Net_Yield_bm" in subset_df.columns:
        tmp_df = subset_df[["Net_Yield", "Net_Yield_bm"]].copy()
    else:
        tmp_df = subset_df[["Net_Yield"]].copy()
    tmp_df["zero"] = 0
    tmp_df["max"] = tmp_df["Net_Yield"].max()
    tmp_df["min"] = tmp_df["Net_Yield"].min()
    st.line_chart(tmp_df)

    if sheet == "USD":
        cols = [
            "Portfolio",
            "Cost",
            "Net_Gain",
            "Net_Yield",
            "Realised_Gain",
            "Paper_Gain",
            "Equity",
            "Cash",
            "USDSGD",
        ]
    else:
        cols = [
            "Portfolio",
            "Cost",
            "Net_Gain",
            "Net_Yield",
            "Div",
            "Realised_Gain",
            "Paper_Gain",
        ]
    st.write(subset_df[cols].tail(30)[::-1])

    if sheet in ["Overall", "Overall Equity"]:
        return

    tu_map = {
        "yearmonth": "1ME",
        "yearquarter": "1QE",
        "year": "1YE",
    }
    timeunits = st.selectbox("Select time units.", list(tu_map.keys()), 1, key=sheet)
    df = sum_by_time(sheet, last_date, tu_map[timeunits])
    st.altair_chart(barchart(df[["cost"]], "Cost", timeunits), use_container_width=True)
    if "div" in df.columns:
        st.altair_chart(barchart(df[["div"]], "Dividends", timeunits), use_container_width=True)
    if "gain" in df.columns:
        st.altair_chart(
            barchart(df[["gain"]], "Realised Gain", timeunits),
            use_container_width=True,
        )

    if sheet == "SGD":
        st.header("Assess portfolio")
        for i in [2, 1]:
            st.subheader(f"{i}Y")
            results = get_whatif_portfolio(last_date.replace(year=last_date.year - i), last_date)
            _print_results(*results)


def _print_results(dct, results_df, ts_df):
    st.write("Date Range: `{}` to `{}` (Values in %)".format(dct["start_date"], dct["end_date"]))
    st.table(results_df * 100)
    # Compare daily portfolio value with index using a normalized plot
    st.line_chart(rebase(ts_df))
