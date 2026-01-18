import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from loguru import logger

# Configure loguru to use stderr to avoid corrupting MCP stdout transport
logger.remove()
logger.add(sys.stderr, level="INFO")

# Load environment variables
if not load_dotenv():
    logger.warning("Unable to load environment variables from .env file")

# Configure Gemini Client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    client = None
else:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None

# Default Template from app_metrics.py
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

# Create FastMCP server
mcp = FastMCP("PDF Analyzer")


@mcp.tool(
    name="analyze_pdf",
    description="Analyzes a PDF file and extracts metrics.",
)
async def analyze_pdf(
    pdf_path: str, prompt: str = None, model_name: str = "gemini-2.0-flash"
) -> str:
    """
    Analyzes a PDF file using Gemini and extracts metrics.

    Args:
        pdf_path: The absolute path to the PDF file.
        prompt: The extraction prompt. Defaults to the standard metrics template if not provided.
        model_name: The Gemini model to use (e.g., gemini-2.0-flash, gemini-1.5-flash).
    """
    if client is None:
        return "Gemini client not initialized. Check your GOOGLE_API_KEY."

    if not os.path.exists(pdf_path):
        return f"File not found at {pdf_path}"

    if not prompt:
        prompt = DEFAULT_TEMPLATE

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        response = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
        )

        # Accumulate all chunks into a single string
        result = ""
        async for chunk in response:
            if chunk.text:
                result += chunk.text

        return result
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
