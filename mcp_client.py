import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_pdf>")
        return

    pdf_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    # Assuming mcp_server.py is in the same directory
    server_script = os.path.join(os.path.dirname(__file__), "reflex_app/mcp_server.py")

    server_params = StdioServerParameters(
        command="python",
        args=[server_script],
    )

    print(f"Connecting to MCP server at {server_script}...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print("Available tools:")
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    print(f" - {tool.name}: {tool.description}")

                print(f"\nCalling 'analyze_pdf' for: {pdf_path}...")
                result = await session.call_tool("analyze_pdf", arguments={"pdf_path": pdf_path})

                print("\nResponse from Gemini:")
                if hasattr(result, "content"):
                    for content in result.content:
                        if content.type == "text":
                            print(content.text, end="", flush=True)
                        else:
                            print(f"[Received non-text content: {content.type}]")
                else:
                    # If result is an iterator (for streaming)
                    async for chunk in result:
                        print(chunk, end="", flush=True)
                print()

    except Exception as e:
        print(f"Error during MCP communication: {e}")


if __name__ == "__main__":
    asyncio.run(main())
