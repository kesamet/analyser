"""
Parser
"""
import os
import tempfile

import streamlit as st

from analyser.utils_parser import (
    perform,
    extract_pages_keyword,
    extract_all_lines_slides,
    extract_all_lines_report,
    extract_most_plausible,
    page_parse_table,
    extract_all_tables_report,
)
from analyser.utils_app import get_pdf_display, download_button


def search_highlight():
    st.write("**To search for terms in the uploaded PDF, extract the pages containing these terms, and highlight the terms in the pages.**")
    search_words = {
        "Aggregate leverage, Gearing": "or",
        "Cost of debt": "or",
        "Interest cover": "or",
        "Average term to maturity": "or",
        "WALE,Weighted average lease expiry": "or",
        "Unit price performance, Closing, Highest, Lowest": "and",
        "Net property income": "or",
        "Distribution per unit,DPU": "or",
        "Financial position, Total assets, Total liabilities, Investment properties": "and",
        "Total debts": "or",
        "Units in issue": "or",
        "Net asset value, NAV": "or",
    }

    uploaded_file = st.file_uploader("Upload a PDF.")
    option = st.radio("Mode", ["Predefined", "Enter your own search"])
    if option == "Predefined":
        input_txt = st.selectbox("Predefined options", list(search_words.keys()))
        mode = search_words[input_txt]
    else:
        input_txt = st.text_input("Enter search terms (For multiple terms, use comma to separate)")
        mode = "or"
        
    if uploaded_file is not None and input_txt != "":
        input_txt = [x.strip() for x in input_txt.split(",")]
        extracted_doc, page_nums = perform(
            uploaded_file.read(), lambda x: extract_pages_keyword(x, input_txt, mode=mode))

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
            st.write(f"Pages: `{page_nums}`")
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.write("`None found.`")


# @st.cache
# def read_table_tabula(uploaded_file, page_num):
#     return perform(uploaded_file.read(),
#                    lambda x: tabula.read_pdf(x, pages=[int(page_num)]))


@st.cache
def read_table_custom(uploaded_file, page_num, heading, ending):
    return perform(uploaded_file.read(),
                   lambda x: page_parse_table(x, int(page_num) - 1, heading, ending))


def table_ocr():
    st.write("**To extract table from a PDF page into a csv.**")
    uploaded_file = st.file_uploader("Upload a PDF.")       
    page_num = st.text_input("Page to extract from.") 
    heading = st.text_input("Enter table heading.", "Group")
    ending = st.text_input("Enter table ending.", "Page")
    run_ocr = st.button("Run")

    if run_ocr:
        assert page_num.isdigit()
        df = read_table_custom(uploaded_file, page_num, heading, ending)

        st.header("Output")
        button = download_button(df, "download.csv", "Export as CSV")
        st.markdown(button, unsafe_allow_html=True)
        st.dataframe(df, height=800)
        
#         st.header("Uploaded PDF")
#         pdf_display = get_pdf_display(uploaded_file.read())
#         st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache
def extract_lines(uploaded_file, mode):
    if mode == "slides":
        return perform(uploaded_file.read(), extract_all_lines_slides)
    elif mode == "financials":
        return perform(uploaded_file.read(), extract_all_lines_report)


def search_extract():
    st.write("**To search for predefined terms in the uploaded PDF, and extract plausible lines containing the information.**")
    mode = st.radio("Select PDF type.", ["slides", "financials"])
    uploaded_file = st.file_uploader("Upload a PDF.")

    if uploaded_file is not None:
        results = extract_lines(uploaded_file, mode)

        st.header("Output")
        st.dataframe(extract_most_plausible(results), height=400)

        select_keyword = st.selectbox("Select a keyword.", list(results.keys()))
        tmp = results[select_keyword]

        st.subheader("Possible values")
        if not tmp:
            st.write("`None found.`")
        else:
            for k, v in tmp.items():
                st.text(k)
                st.write("Found on page `{}` in line".format(v["page_num"]))
                st.text(v["line"])
                st.write("with title")
                st.text(v["first_line"])
                # st.text(json.dumps(v, indent=2))
                st.write("=" * 73)


@st.cache
def extract_tables(uploaded_file, key):
    return perform(uploaded_file.read(), lambda x: extract_all_tables_report(x, key))


def search_extract_v2():
    from analyser.constants import dct

    key = st.selectbox("Select report.", list(dct.keys()))
    uploaded_file = st.file_uploader("Upload a PDF.")
    run_extract = st.button("Run")

    if run_extract and uploaded_file is not None:
        results = extract_tables(uploaded_file, key)

        st.header("Output")
        for k, v in results.items():
            st.write(f"**{k}**")
            if v:
                st.dataframe(v[0])
            else:
                st.write("`None found.`")
