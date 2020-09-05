from datetime import date, timedelta

import numpy as np
import pandas as pd
import ta
import altair as alt
import streamlit as st
from streamlit.elements import altair

import analyser.utils_charts as F
from symbols_dicts import reits_dict


@st.cache
def reverse_dict(dct):
    return {v: k for k, v in dct.items()}


EQ_DICT = {
    'ES3.SI': 'STI ETF',
    'O87.SI': 'GLD US$',
    'D05.SI': 'DBS Group Holdings Ltd',
}
EQ_DICT.update(reits_dict)
EQ_DICT = reverse_dict(EQ_DICT)


def get_start_date(today_date, start_date="2015-01-01",
                   options=("1Y", "2Y", "YTD", "MTD", "All")):
    select_range = st.selectbox("Select time range", options)
    if select_range == "5Y":
        _yr = today_date.year - 5
        return today_date.replace(year=_yr).strftime("%Y-%m-%d")
    if select_range == "2Y":
        _yr = today_date.year - 2
        return today_date.replace(year=_yr).strftime("%Y-%m-%d")
    elif select_range == "1Y":
        _yr = today_date.year - 1
        return today_date.replace(year=_yr).strftime("%Y-%m-%d")
    elif select_range == "9M":
        return (today_date - timedelta(days=30 * 9)).strftime("%Y-%m-%d")
    elif select_range == "6M":
        return (today_date - timedelta(days=30 * 6)).strftime("%Y-%m-%d")
    elif select_range == "3M":
        return (today_date - timedelta(days=30 * 3)).strftime("%Y-%m-%d")
    elif select_range == "1M":
        return (today_date - timedelta(days=30)).strftime("%Y-%m-%d")
    elif select_range == "YTD":
        return today_date.replace(month=1, day=1).strftime("%Y-%m-%d")
    elif select_range == "MTD":
        return today_date.replace(day=1).strftime("%Y-%m-%d")
    return start_date


@st.cache
def load_ie_data():
    df = F.get_ie_data(start_date="1990-01-01")
    df["10xReal_Earnings"] = 10 * df["Real_Earnings"]
    df["10xLong_IR"] = 10 * df["Long_IR"]
    return df[["Real_Price", "10xReal_Earnings", "CAPE", "10xLong_IR"]]


@st.cache
def load_data(dates, symbols, base_symbol="ES3.SI"):
    if "IWDA" not in symbols and "EIMI" not in symbols:
        df = F.get_data(symbols, dates, base_symbol=base_symbol)
    else:
        df = F.get_data_xlsx(symbols, dates, base_symbol="IWDA")
    return df


def page_charts(today_date=date.today() - timedelta(days=1)):
    df0 = load_ie_data()
    c1 = altair.generate_chart("line", df0[["Real_Price", "10xReal_Earnings"]]).properties(
        title="Index Plot"
    )
    c2 = altair.generate_chart("line", df0[["CAPE", "10xLong_IR"]]).properties(
        title="PE (CAPE) Plot"
    )
    st.altair_chart(alt.concat(c1, c2, columns=2), use_container_width=True)

    # 5Y
    dates = pd.date_range(today_date - timedelta(days=365 * 5), today_date)

    # MSCI
    symbols = ['URTH', 'EEM', 'SPY', 'ES3.SI']
    colnames = ['MSCI World', 'MSCI EM', 'S&P500', 'ES3']
    df1 = load_data(dates, symbols, "SPY")
    df1.columns = colnames
    chart1 = altair.generate_chart("line", F.rebase(df1)).properties(
        title="MSCI"
    )

    # VIX
    symbols = ['^VIX']
    colnames = ['VIX']
    df2 = load_data(dates, symbols)[symbols]
    df2.columns = colnames
    chart2 = altair.generate_chart("line", df2).properties(
        title="VIX"
    )

    st.altair_chart(alt.concat(chart1, chart2, columns=2), use_container_width=True)

    start_date = get_start_date(today_date, options=("2Y", "1Y", "9M", "6M", "3M", "YTD"))

    # etfs
    symbols = ['IWDA', 'EIMI']
    colnames = ['World', 'EM']
    df3a = load_data(dates, symbols)
    df3a.columns = colnames
    rebased_df3a = F.rebase(df3a[df3a.index >= start_date])
    chart3a = altair.generate_chart("line", rebased_df3a).properties(
        title="ETF"
    )
    symbols = ['O87.SI', 'ES3.SI', 'CLR.SI']
    colnames = ['GLD', 'ES3', 'Lion-Phillip']
    df3b = load_data(dates, symbols)
    df3b.columns = colnames
    rebased_df3b = F.rebase(df3b[df3b.index >= start_date])
    chart3b = altair.generate_chart("line", rebased_df3b).properties(
        title="ETF SGX"
    )
    st.altair_chart(alt.concat(chart3a, chart3b, columns=2), use_container_width=True)

    # industrial
    symbols = ['ES3.SI', 'O5RU.SI', 'A17U.SI', 'J91U.SI', 'BUOU.SI', 'ME8U.SI', 'M44U.SI']
    colnames = ['ES3', 'AA', 'Ascendas', 'ESR', 'FLCT', 'MIT', 'MLT']
    df4 = load_data(dates, symbols)
    df4.columns = colnames
    rebased_df4 = F.rebase(df4[df4.index >= start_date])
    chart4a = altair.generate_chart("line", rebased_df4[['ES3', 'Ascendas', 'FLCT', 'MIT', 'MLT']]).properties(
        title="Industrial 1"
    )
    chart4b = altair.generate_chart("line", rebased_df4[['ES3', 'AA', 'ESR']]).properties(
        title="Industrial 2"
    )
    st.altair_chart(alt.concat(chart4a, chart4b, columns=2), use_container_width=True)

    # retail
    symbols = ['ES3.SI', 'C38U.SI', 'J69U.SI', 'N2IU.SI']
    colnames = ['ES3', 'CMT', 'FCT', 'MCT']
    df5 = load_data(dates, symbols)
    df5.columns = colnames
    rebased_df5 = F.rebase(df5[df5.index >= start_date])
    chart5 = altair.generate_chart("line", rebased_df5).properties(
        title="Retail & Commercial"
    )

    # banks
    symbols = ['ES3.SI', 'D05.SI', 'O39.SI', 'U11.SI']
    colnames = ['ES3', 'DBS', 'OCBC', 'UOB']
    df6 = load_data(dates, symbols)
    df6.columns = colnames
    rebased_df6 = F.rebase(df6[df6.index >= start_date])
    chart6 = altair.generate_chart("line", rebased_df6).properties(
        title="Banks"
    )
    st.altair_chart(alt.concat(chart5, chart6, columns=2), use_container_width=True)


@st.cache
def table_trend(dates, symbols, colnames):
    results = F.get_trending(dates, symbols, ["IWDA", "EIMI"])
    results.index = symbols + ["IWDA", "EIMI"]
    results["symbol"] = colnames + ["IWDA", "EIMI"]
    return results.sort_values("level")


@st.cache
def compute_trend(dates, symbol):
    return F.compute_trend(dates, symbol)


def get_trend_chart(symbol, today_date, days):
    dates = pd.date_range(today_date - timedelta(days=days), today_date)
    df, level, res, last, grad = compute_trend(dates, symbol)

    st.text(f"Last close = {df[symbol].iloc[-1]:.3f} ({level:.1f}%)\n"
            f"Levels = {last - 2 * res:.3f}, {last - res:.3f}, {last:.3f}, "
            f"{last + res:.3f}, {last + 2 * res:.3f}\n"
            f"Gradient = {grad * 1e3:.3f}")

    return altair.generate_chart("line", df)


def page_analysis(today_date=date.today() - timedelta(days=1)):
    """Analysis page."""
    st.subheader("Top trending")
    select_days1 = st.selectbox("Select lookback days", [30, 60, 91, 121, 182, 365, 730], 3)
    dates = pd.date_range(today_date - timedelta(days=select_days1), today_date)
    colnames = list(EQ_DICT.keys())
    symbols = list(EQ_DICT.values())
    df = table_trend(dates, symbols, colnames)

    def highlight(x):
        if x < 25:
            return "background-color: #82E0AA"  # green
        elif x > 75:
            return "background-color: #F1948A"  # red
        return ""

    st.write("**Main**")
    symbols = ['O5RU.SI', 'A17U.SI', 'J91U.SI', 'BUOU.SI', 'ME8U.SI', 'M44U.SI',
               'D05.SI', 'C38U.SI', 'N2IU.SI', 'CJLU.SI', 'ES3.SI', 'IWDA', 'EIMI']
    # colnames = ['AA', 'Ascendas', 'ESR', 'FLCT', 'MIT', 'MLT',
    #             'DBS', 'CMT', 'MCT', 'Netlink', 'ES3', 'IWDA', 'EIMI']
    st.dataframe(
        df[df.index.isin(symbols)].style
        .set_precision(3)
        .applymap(lambda x: "color: red" if x < 0 else "", subset=["grad"])
        .applymap(highlight, subset=["level"]),
        height=400,
    )

    st.write("**All**")
    st.dataframe(
        df.style
        .set_precision(3)
        .applymap(lambda x: "color: red" if x < 0 else "", subset=["grad"])
        .applymap(highlight, subset=["level"]),
    )

    st.subheader("Trend")
    select_eq = st.selectbox("Select equity", list(EQ_DICT.keys()))
    symbol = EQ_DICT[select_eq]

    select_days = st.selectbox("Select lookback days", [30, 60, 91, 121, 182, 730, 1095], 4)

    st.write("Lookback days = `365`")
    chart7 = get_trend_chart(symbol, today_date, 365)
    st.write(f"Lookback days = `{select_days}`")
    chart8 = get_trend_chart(symbol, today_date, select_days)
    st.altair_chart(alt.concat(chart7, chart8, columns=2), use_container_width=True)


@st.cache(allow_output_mutation=True)
def load_ohlcv_data(symbol, dates, base_symbol='ES3.SI'):
    # Load ohlc data
    df = F.get_data_ohlcv(symbol, dates, base_symbol=base_symbol)

    # Apply feature engineering (technical analysis)
    df = ta.add_volatility_ta(df, "high", "low", "close", fillna=False, colprefix="ta_")
    df = ta.add_momentum_ta(df, "high", "low", "close", "volume", fillna=False, colprefix="ta_")
    df = add_custom_trend(df, "close", fillna=False, colprefix="ta_")
    return df


def add_custom_trend(df, close, fillna, colprefix):
    # MACD
    indicator_macd = ta.trend.MACD(close=df[close], n_slow=26, n_fast=12, n_sign=9, fillna=fillna)
    df[f'{colprefix}trend_macd'] = indicator_macd.macd()
    df[f'{colprefix}trend_macd_signal'] = indicator_macd.macd_signal()
    df[f'{colprefix}trend_macd_diff'] = indicator_macd.macd_diff()

    # SMAs
    df[f'{colprefix}trend_sma_fast'] = ta.trend.SMAIndicator(
        close=df[close], n=50, fillna=fillna).sma_indicator()
    df[f'{colprefix}trend_sma_slow'] = ta.trend.SMAIndicator(
        close=df[close], n=200, fillna=fillna).sma_indicator()
    df[f'{colprefix}trend_sma_10'] = ta.trend.SMAIndicator(
        close=df[close], n=10, fillna=fillna).sma_indicator()
    df[f'{colprefix}trend_sma_25'] = ta.trend.SMAIndicator(
        close=df[close], n=25, fillna=fillna).sma_indicator()
    return df


def chart_candlestick(df, last):
    source = df.iloc[-last:]
    base = alt.Chart(source).encode(
        alt.X('date:T'),
        color=alt.condition(
            "datum.open <= datum.close",
            alt.value("#06982d"),
            alt.value("#ae1325"),
        ),
    )
    rule = base.mark_rule().encode(
        alt.Y('low:Q', title='Price', scale=alt.Scale(zero=False)),
        alt.Y2('high:Q')
    )
    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )
    return rule + bar


def page_ta(today_date=date.today() - timedelta(days=1)):
    """Technical analysis page."""
    select_eq = st.selectbox("Select equity", list(EQ_DICT.keys()))

    dates = pd.date_range(today_date - timedelta(days=365 * 5), today_date)
    df = load_ohlcv_data(EQ_DICT[select_eq], dates)

    st.subheader('Price')
    st.altair_chart(chart_candlestick(df.reset_index(), 126), use_container_width=True)

    select_ta = st.selectbox("Select TA type", ["Bollinger", "SMA", "MACD", "RSI", "Momentum"])
    ta_type = {
        "Bollinger": {
            "price": ["ta_volatility_bbm", "ta_volatility_bbh", "ta_volatility_bbl"],
            "ind": ["ta_volatility_bbhi", "ta_volatility_bbli"],
        },
        "SMA": {
            "price": ["ta_trend_sma_fast", "ta_trend_sma_slow"],
        },
        "MACD": {
            "price": ["ta_trend_macd_signal"],
            "ind": ["ta_trend_macd", "ta_trend_macd_signal", "ta_trend_macd_diff"],
        },
        "RSI": {
            "price": ["ta_trend_sma_10", "ta_trend_sma_25"],
            "ind": ["ta_momentum_rsi"],
        },
        "Momentum": {
            "price": [],
            "ind": ["ta_momentum_tsi", "ta_momentum_rsi", "ta_momentum_stoch"],
        },
    }
    last = 126
    st.line_chart(df[["close"] + ta_type[select_ta]["price"]].iloc[-last:])
    if ta_type[select_ta].get("ind") is not None:
        st.line_chart(df[ta_type[select_ta]["ind"]].iloc[-last:])

    # Prepare target: X Periods Return
    select_periods = st.slider("Select periods", 1, 14, 1)
    df['y'] = F.pct_change(df["close"], select_periods) * 100

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
