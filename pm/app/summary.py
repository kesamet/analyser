from datetime import date

import pandas as pd
import streamlit as st

from analyser.plots import plotly_ringchart
from pm.app.data import get_overall_portfolio


def page_summary(last_date: date) -> None:
    """Summary page."""
    _, last = get_overall_portfolio()

    df = pd.DataFrame.from_dict(last, orient="index", columns=["Value", "start"])
    df = df.loc[["SRS", "Fund", "USD", "SGD", "Bond"]]
    df["% Change YTD"] = (df["Value"] / df["start"] - 1) * 100
    total = df["Value"].sum()
    st.metric("Portfolio Value (SGD)", f"{total:,.2f}")
    st.dataframe(
        df[["Value", "% Change YTD"]].style.format({"Value": "{:,.2f}", "% Change YTD": "{:.2f}"})
    )

    names = list(last.keys())
    data = [last[k][0] for k in names]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    st.plotly_chart(plotly_ringchart(data, labels))

    names = ["World", "SGD", "CCE"]
    data = [
        last["USD"][0] + last["Fund"][0] + last["SRS"][0],
        last["SGD"][0],
        last["Bond"][0],
    ]
    total = sum(data)
    labels = [f"{k}: {v / total:.1%}" for k, v in zip(names, data)]
    st.plotly_chart(plotly_ringchart(data, labels))
