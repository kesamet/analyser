import streamlit as st
import os
from google import genai
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
if not load_dotenv():
    logger.warning("Unable to load environment variables from .env file")

# Configure Gemini Client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    client = None
else:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None

# Metrics Template
DEFAULT_TEMPLATE = """Retrieve the following metrics from the document. 
For each metric, provide the value found, the section of the source of the value, \
and the page number where it was retrieved.
The section of the source of the value should be a short description of the section of the document \
where the value was found.

Metrics to extract:
- net property income
- distribution per unit (DPU)
- investment properties
- total assets
- total liabilities
- total debts
- total number of units
- net asset value (NAV)
- aggregate leverage
- cost of debt
- interest cover
- average term to maturity / weighted average tenor of debt
- weighted average lease expiry (WALE)

Output the response in a JSON format where each entry contains a "value", a "section", and a "page_number":

{
    "net_property_income": {"value": "", "section": "", "page_number": ""},
    "distribution_per_unit": {"value": "", "section": "", "page_number": ""},
    "investment_properties": {"value": "", "section": "", "page_number": ""},
    ...
}
"""


def analyze_pdf(pdf_bytes, prompt, model_name):
    """Passes PDF bytes and prompt to Gemini using the new SDK."""
    if client is None:
        return "", "Gemini client not initialized. Check your GOOGLE_API_KEY."

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
        )
        return response.text, None
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return "", f"Error: {e}"


def main():
    st.set_page_config(page_title="PDF Metrics Extractor", layout="wide")
    st.title("📄 PDF Metrics Extractor")

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Select Model",
            ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash"],
            index=0,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload & Prompt")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        prompt = st.text_area("Analysis Prompt", value=DEFAULT_TEMPLATE, height=400)

        analyze_button = st.button("Extract Metrics", type="primary", disabled=not uploaded_file)

    with col2:
        st.subheader("Results")
        if analyze_button and uploaded_file:
            with st.spinner(f"Processing PDF with {model_name}..."):
                # Reset file pointer to beginning just in case
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                response, error = analyze_pdf(file_bytes, prompt, model_name)
                if error:
                    st.error(error)
                else:
                    st.markdown(response)
        elif not uploaded_file:
            st.info("Please upload a PDF file to begin.")


if __name__ == "__main__":
    main()
