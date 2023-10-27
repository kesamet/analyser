"""
Streamlit app
"""
from datetime import date, timedelta

import streamlit as st
from streamlit_option_menu import option_menu

from analyser.app.charts import page_ta
from pm.symbols import EQ_DICT
from pm.app.charts import page_charts
from pm.app.data import page_data
from pm.app.summary import page_summary
from pm.app.trend import page_trend


def main():
    today_date = date.today()

    dict_pages = {
        "Summary": page_summary,
        "Portfolio": page_data,
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
