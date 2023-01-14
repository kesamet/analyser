"""
Streamlit app
"""
from datetime import date, timedelta

import streamlit as st

from analyser.app.charts import page_ta
from analyser.app.parser import search_highlight, table_ocr, search_extract
from symbols import EQ_DICT


def main():
    st.sidebar.title("Analyser")

    dict_pages = {
        "Technical Analysis": page_ta,
        "Search and Highlight": search_highlight,
        "Search and Extract": search_extract,
        "Table OCR": table_ocr,
    }

    select_page = st.sidebar.radio("pages", list(dict_pages.keys()), label_visibility="collapsed")
    st.title(select_page)

    today_date = date.today()
    dict_pages[select_page](
        last_date=today_date - timedelta(days=1),
        eq_dict=EQ_DICT,
    )


if __name__ == "__main__":
    main()
