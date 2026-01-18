import asyncio
import os
import json
from typing import Optional
import reflex as rx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Default template from mcp_server.py
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


class State(rx.State):
    """The app state."""

    # File upload
    uploaded_file: Optional[str] = None
    file_name: str = ""

    # Configuration
    custom_prompt: str = ""
    selected_model: str = "gemini-2.0-flash"

    # Analysis state
    is_analyzing: bool = False
    analysis_result: str = ""
    error_message: str = ""
    success_message: str = ""

    def set_custom_prompt(self, value: str):
        self.custom_prompt = value

    def set_selected_model(self, value: str):
        self.selected_model = value

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle file upload."""
        if not files:
            self.error_message = "No file selected"
            return

        file = files[0]

        # Check if it's a PDF
        if not file.filename.endswith(".pdf"):
            self.error_message = "Please upload a PDF file"
            return

        # Save the uploaded file
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        # Read and save file
        upload_data = await file.read()
        with open(file_path, "wb") as f:
            f.write(upload_data)

        self.uploaded_file = file_path
        self.file_name = file.filename
        self.error_message = ""
        self.success_message = f"File '{file.filename}' uploaded successfully!"

    async def analyze_pdf(self):
        """Analyze the uploaded PDF using MCP client."""
        if not self.uploaded_file:
            self.error_message = "Please upload a PDF file first"
            return

        self.is_analyzing = True
        self.error_message = ""
        self.success_message = ""
        self.analysis_result = ""

        try:
            # Get the MCP server script path
            server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

            server_params = StdioServerParameters(
                command="python",
                args=[server_script],
            )

            # Prepare arguments
            arguments = {"pdf_path": self.uploaded_file}

            # Add custom prompt if provided
            if self.custom_prompt.strip():
                arguments["prompt"] = self.custom_prompt

            # Add model selection
            arguments["model_name"] = self.selected_model

            # Connect to MCP server and analyze
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    result = await session.call_tool("analyze_pdf", arguments=arguments)

                    # Extract result text
                    result_text = ""
                    if hasattr(result, "content"):
                        for content in result.content:
                            if content.type == "text":
                                result_text += content.text

                    # Try to format as JSON if possible
                    try:
                        parsed_json = json.loads(result_text)
                        self.analysis_result = json.dumps(parsed_json, indent=2)
                    except json.JSONDecodeError:
                        self.analysis_result = result_text

                    self.success_message = "Analysis completed successfully!"

        except Exception as e:
            self.error_message = f"Error during analysis: {str(e)}"

        finally:
            self.is_analyzing = False

    def clear_messages(self):
        """Clear success and error messages."""
        self.error_message = ""
        self.success_message = ""

    def reset_form(self):
        """Reset the form."""
        self.uploaded_file = None
        self.file_name = ""
        self.custom_prompt = ""
        self.analysis_result = ""
        self.error_message = ""
        self.success_message = ""


def index() -> rx.Component:
    """The main page."""
    return rx.container(
        rx.vstack(
            # Header
            rx.heading(
                "📄 PDF Analyzer",
                size="9",
                weight="bold",
                background_image="linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
                background_clip="text",
                margin_bottom="0.5rem",
            ),
            rx.text(
                "Extract financial metrics from PDF documents using AI",
                size="4",
                color="gray",
                margin_bottom="2rem",
            ),
            # Upload Section
            rx.card(
                rx.vstack(
                    rx.heading("Upload PDF", size="6", margin_bottom="1rem"),
                    rx.upload(
                        rx.vstack(
                            rx.button(
                                "Select File",
                                color_scheme="purple",
                                size="3",
                            ),
                            rx.text(
                                "Drag and drop or click to select",
                                size="2",
                                color="gray",
                            ),
                        ),
                        id="upload1",
                        border="2px dashed #667eea",
                        padding="2rem",
                        border_radius="8px",
                    ),
                    rx.hstack(
                        rx.foreach(
                            rx.selected_files("upload1"),
                            lambda file: rx.text(file),
                        ),
                    ),
                    rx.button(
                        "Upload",
                        on_click=State.handle_upload(rx.upload_files(upload_id="upload1")),
                        color_scheme="purple",
                        size="3",
                    ),
                    rx.cond(
                        State.file_name != "",
                        rx.text(
                            f"✓ {State.file_name}",
                            color="green",
                            weight="bold",
                        ),
                    ),
                    spacing="4",
                    width="100%",
                ),
                width="100%",
            ),
            # Configuration Section
            rx.card(
                rx.vstack(
                    rx.heading("Configuration", size="6", margin_bottom="1rem"),
                    # Model Selection
                    rx.vstack(
                        rx.text("Model:", weight="bold", size="3"),
                        rx.select(
                            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
                            value=State.selected_model,
                            on_change=State.set_selected_model,
                            size="3",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    # Custom Prompt
                    rx.vstack(
                        rx.text("Custom Prompt (optional):", weight="bold", size="3"),
                        rx.text_area(
                            placeholder="Enter custom extraction prompt or leave empty for default metrics...",
                            value=State.custom_prompt,
                            on_change=State.set_custom_prompt,
                            rows="6",
                            width="100%",
                        ),
                        rx.text(
                            "Leave empty to use default financial metrics template",
                            size="1",
                            color="gray",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    spacing="4",
                    width="100%",
                ),
                width="100%",
            ),
            # Action Buttons
            rx.hstack(
                rx.button(
                    rx.cond(
                        State.is_analyzing,
                        rx.hstack(
                            rx.spinner(size="3"),
                            rx.text("Analyzing..."),
                            spacing="2",
                        ),
                        rx.text("🔍 Analyze PDF"),
                    ),
                    on_click=State.analyze_pdf,
                    disabled=State.is_analyzing,
                    color_scheme="purple",
                    size="4",
                    width="200px",
                ),
                rx.button(
                    "Reset",
                    on_click=State.reset_form,
                    color_scheme="gray",
                    variant="soft",
                    size="4",
                ),
                spacing="4",
                justify="center",
                width="100%",
            ),
            # Messages
            rx.cond(
                State.success_message != "",
                rx.callout(
                    State.success_message,
                    icon="check",
                    color_scheme="green",
                    role="alert",
                ),
            ),
            rx.cond(
                State.error_message != "",
                rx.callout(
                    State.error_message,
                    icon="triangle_alert",
                    color_scheme="red",
                    role="alert",
                ),
            ),
            # Results Section
            rx.cond(
                State.analysis_result != "",
                rx.card(
                    rx.vstack(
                        rx.heading("Analysis Results", size="6", margin_bottom="1rem"),
                        rx.code_block(
                            State.analysis_result,
                            language="json",
                            width="100%",
                        ),
                        spacing="4",
                        width="100%",
                    ),
                    width="100%",
                ),
            ),
            spacing="6",
            width="100%",
            max_width="900px",
            padding="2rem",
        ),
        center_content=True,
    )


# Create the app
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="purple",
    )
)
app.add_page(index, title="PDF Analyzer")
