"""
Parser
"""
import os
import tempfile

import streamlit as st

from analyser.constants import (
    PHRASES_SEARCH,
    KEYWORDS_EXTRACT_SLIDES,
    KEYWORDS_EXTRACT_REPORT,
)
from analyser.utils_parser import (
    perform,
    extract_pages_keyword,
    extract_all_lines_slides,
    extract_all_lines_report,
    extract_most_plausible,
    page_parse_table,
)
from analyser.utils_app import get_pdf_display, download_button


def search_highlight():
    st.write(
        "**Extract pages containing the searched terms, and highlight the terms in the pages.**"
    )
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF.")

    option = st.sidebar.radio("options", ["Predefined", "Enter your own search"], label_visibility="collapsed")
    if option == "Predefined":
        input_txt = st.sidebar.selectbox(
            "Predefined options", list(PHRASES_SEARCH.keys())
        )
        mode = PHRASES_SEARCH[input_txt]
    else:
        input_txt = st.sidebar.text_input(
            "Enter search terms (For multiple terms, use comma to separate)"
        )
        mode = "or"

    if uploaded_file is not None and input_txt != "":
        extracted_doc, page_nums = perform(
            lambda x: extract_pages_keyword(
                x, [x.strip() for x in input_txt.split(",")], mode=mode
            ),
            uploaded_file.read(),
        )

        st.header("Output")
        if extracted_doc is not None:
            fh, temp_filename = tempfile.mkstemp()
            try:
                extracted_doc.save(temp_filename, garbage=4, deflate=True, clean=True)
                with open(temp_filename, "rb") as f:
                    pdfbytes = f.read()
                pdf_display = get_pdf_display(pdfbytes)
            finally:
                os.close(fh)
                os.remove(temp_filename)

            page_nums = ", ".join(str(n + 1) for n in page_nums)
            st.write(f"`{input_txt}` found on pages: `{page_nums}`")
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.write("`None found.`")


@st.cache
def extract_lines(uploaded_file, mode):
    if mode == "slides":
        return perform(
            extract_all_lines_slides,
            uploaded_file.read(),
            dict_keywords=KEYWORDS_EXTRACT_SLIDES,
        )
    return perform(
        extract_all_lines_report,
        uploaded_file.read(),
        dict_keywords=KEYWORDS_EXTRACT_REPORT,
    )


def search_extract():
    st.write("**Extract plausible lines containing predefined terms.**")
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF.")
    select_doctype = st.sidebar.radio("options", ["slides", "financials"], label_visibility="collapsed")

    if uploaded_file is not None:
        all_results = extract_lines(uploaded_file, select_doctype)

        st.header("Output")
        c0, _ = st.columns(2)
        c0.table(extract_most_plausible(all_results))

        select_keyphase = st.selectbox("Select a keyphase.", list(all_results.keys()))
        results = all_results[select_keyphase]

        st.subheader("Possible values")
        if not results:
            st.write("`None found.`")
        else:
            for dct in results:
                st.text(dct["value"])
                st.write("Found on page `{}` in line".format(dct["page_num"]))
                st.text(dct["line"])
                st.markdown("---")


@st.cache
def read_table_custom(uploaded_file, page_num, heading, ending):
    return perform(
        lambda x: page_parse_table(x, int(page_num) - 1, heading, ending),
        uploaded_file.read(),
    )


def table_ocr():
    st.write("**To extract table from a PDF page into a csv.**")
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF.")
    page_num = int(st.sidebar.number_input("Page to extract from.", min_value=1))
    heading = st.sidebar.text_input("Enter table heading.", "Group")
    ending = st.sidebar.text_input("Enter table ending.", "Page")
    run_ocr = st.sidebar.button("Extract")

    if run_ocr:
        df = read_table_custom(uploaded_file, page_num, heading, ending)

        st.header("Output")
        button = download_button(df, "download.csv", "Export as CSV")
        st.markdown(button, unsafe_allow_html=True)
        st.dataframe(df, height=800)
