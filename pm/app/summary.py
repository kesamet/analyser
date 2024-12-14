from datetime import date

import pandas as pd
import streamlit as st

from analyser.plots import plotly_ringchart
from pm import CFG
from pm.app.data import get_overall_portfolio


def page_summary(last_date: date) -> None:
    """Summary page."""
    _, last = get_overall_portfolio(last_date)
    _last = last.copy()

    c0, c1 = st.columns(2)

    _cash_filename = f"{CFG.SUMMARY_DIR}/cash.txt"
    with open(_cash_filename, "r") as f:
        cash = int(f.read())
    with c1.form("my_form"):
        new_cash = st.number_input("Cash (in '000)", value=cash)
        submitted = st.form_submit_button("Submit")
        if submitted:
            _last["Cash"] = new_cash * 1000
            with open(_cash_filename, "w") as f:
                f.write(str(new_cash))
        else:
            _last["Cash"] = cash * 1000

    with c0:
        df = pd.DataFrame.from_dict(_last, orient="index", columns=["Value"])
        df = df.loc[["Cash", "SRS", "Fund", "USD", "SGD", "Bond"]]
        df.loc["Total"] = df["Value"].sum()

        def _highlight_last(x):
            return ["font-weight: bold" if v == x.iloc[-1] else "" for v in x]

        st.table(df.style.format("{:,.2f}").apply(_highlight_last))

    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "black"]

    names = list(_last.keys())
    data = [_last[k] for k in names]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    st.plotly_chart(plotly_ringchart(data, labels))

    names = ["World", "SGD", "CCE"]
    data = [
        _last["USD"] + _last["Fund"] + _last["SRS"],
        _last["SGD"],
        _last["Cash"] + _last["Bond"],
    ]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    # st.pyplot(py_ringchart(data, labels, colors))
    st.plotly_chart(plotly_ringchart(data, labels))

    names = ["USD", "Fund", "SRS"]
    data = [_last[k] for k in names]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    st.plotly_chart(plotly_ringchart(data, labels, title="World"))
