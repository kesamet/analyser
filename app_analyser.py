"""
Streamlit app
"""
import streamlit as st

from analyser.charts import page_ta
from analyser.parser import search_highlight, table_ocr, search_extract


def main():
    st.sidebar.title("Analyser")

    dict_pages = {
        "Technical Analysis": page_ta,
        "Search and Highlight": search_highlight,
        "Search and Extract": search_extract,
        "Table OCR": table_ocr,
    }

    select_page = st.sidebar.radio("Pages", list(dict_pages.keys()))
    st.title(select_page)
    dict_pages[select_page]()


if __name__ == "__main__":
    main()
