"""
Streamlit app
"""

from datetime import date, timedelta

import streamlit as st
from streamlit_option_menu import option_menu

from analyser.app.charts import page_ta
from analyser.app.parser import search_highlight, table_ocr, search_extract
from symbols import SYMBOLS

st.set_page_config(page_title="Analyser")


def main():
    today_date = date.today()

    dict_pages = {
        "Technical Analysis": page_ta,
        "Search and Highlight": search_highlight,
        "Search and Extract": search_extract,
        "Table OCR": table_ocr,
    }

    with st.sidebar:
        selected = option_menu("Analyser", list(dict_pages.keys()), menu_icon="cast")

    st.title(selected)
    dict_pages[selected](last_date=today_date - timedelta(days=1), eq_dict=SYMBOLS)


if __name__ == "__main__":
    main()
