from fastmcp import FastMCP

mcp = FastMCP("A calculator tool")

@mcp.tool
def return_name(name: str) -> str:
    """Gives back the firstname for a given lastname"""
    return f"Bob kevin {name}"


