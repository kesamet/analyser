"""
Streamlit app
"""
from datetime import date, timedelta

import streamlit as st
from streamlit_option_menu import option_menu

from analyser.app.charts import page_ta
from pm.symbols import EQ_DICT
from pm.app.charts import page_charts
from pm.app.data import page_portfolio
from pm.app.summary import page_summary
from pm.app.trend import page_trend


def main():
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

    with st.sidebar:
        selected = option_menu("PM", list(dict_pages.keys()), menu_icon="cast")

    st.title(selected)
    dict_pages[selected](today_date - timedelta(days=1))


if __name__ == "__main__":
    main()
