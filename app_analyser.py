import streamlit as st

from charts import page_charts, page_ta, page_analysis
from parser import search_highlight, table_ocr, search_extract, search_extract_v2


def main():
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: 1000px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    dict_pages = {
        "Charts": page_charts,
        "Technical Analysis": page_ta,
        "Analysis": page_analysis,
        "Search and Highlight": search_highlight,
        "Table OCR": table_ocr,
        "Search and Extract": search_extract,
        "Search and Extract (custom)": search_extract_v2,
    }

    select_page = st.sidebar.selectbox("Pages", list(dict_pages.keys()))
    st.title(select_page)
    dict_pages[select_page]()


if __name__ == "__main__":
    main()
