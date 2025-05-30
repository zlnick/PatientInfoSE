# server.py
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

# Create an MCP server
mcp = FastMCP("IRIS MCP wrapper server")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!" 

if __name__ == "__main__":
    #mcp.run()
    mcp.run(transport="sse")