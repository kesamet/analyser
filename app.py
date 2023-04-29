"""
Streamlit app
"""
from datetime import date, timedelta

import streamlit as st

from analyser.app.charts import page_ta
from pm.symbols import EQ_DICT
from pm.app.charts import page_charts
from pm.app.data import page_portfolio
from pm.app.summary import page_summary
from pm.app.trend import page_trend


def main():
    st.sidebar.title("PM")
    today_date = date.today()

    dict_pages = {
        "Summary": page_summary,
        "Overall": lambda x: page_portfolio(x, "Overall"),
        "Overall Equity": lambda x: page_portfolio(x, "Overall Equity"),
        "SGD": lambda x: page_portfolio(x, "SGD"),
        "USD": lambda x: page_portfolio(x, "USD"),
        "Fund": lambda x: page_portfolio(x, "Fund"),
        "SRS": lambda x: page_portfolio(x, "SRS"),
        "IDR": lambda x: page_portfolio(x, "IDR"),
        "Bond": lambda x: page_portfolio(x, "Bond"),
        "Trend": page_trend,
        "Charts": page_charts,
        "Technical Analysis": lambda x: page_ta(x, EQ_DICT),
    }

    select = st.sidebar.radio(
        "pages", list(dict_pages.keys()), label_visibility="collapsed"
    )
    st.title(select)
    dict_pages[select](today_date - timedelta(days=1))

    left = (date(2023, 8, 28) - today_date).days
    milestone = today_date + timedelta(days=(left % 50))
    st.sidebar.info(f"{left} days left\n\n{milestone.isoformat()}")
    if left % 50 == 0:
        st.balloons()


if __name__ == "__main__":
    main()
