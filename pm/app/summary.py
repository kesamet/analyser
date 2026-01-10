from datetime import date

import pandas as pd
import streamlit as st

from analyser.plots import plotly_ringchart
from pm.app.data import get_overall_portfolio


def page_summary(last_date: date) -> None:
    """Summary page."""
    _, last = get_overall_portfolio(last_date)
    _last = last.copy()

    df = pd.DataFrame.from_dict(_last, orient="index", columns=["Value"])
    df = df.loc[["SRS", "Fund", "USD", "SGD", "Bond"]]
    df.loc["Total"] = df["Value"].sum()
    st.table(df.style.format("{:,.2f}"))

    names = list(_last.keys())
    data = [_last[k] for k in names]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    st.plotly_chart(plotly_ringchart(data, labels))

    names = ["World", "SGD", "CCE"]
    data = [
        _last["USD"] + _last["Fund"] + _last["SRS"],
        _last["SGD"],
        _last["Bond"],
    ]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    st.plotly_chart(plotly_ringchart(data, labels))
