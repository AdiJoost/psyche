import asyncio
from fastmcp import Client
from openai import OpenAI
import json

client_url = "http://localhost:8000/mcp"

async def run_agent():
    client = OpenAI(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
        )
    response = client.chat.completions.create(
        model = "granite4:latest",
        messages=[
            {"role":"system", "content": "You have some tools to call."},
            {"role": "user", "content": "What tools do you have available? List me them."}
        ],
        tools=[
            {
                "type":"mcp",
                "server_label":"A calculator tool",
                "server_url": client_url,
                "require_approval": "never"
            }
        ],
        )
    msg = response.choices[0].message
    # check for tool/function call first
    tool_call = getattr(msg, "tool_calls", None) or getattr(msg, "function_call", None)
    if tool_call:
        print("Tool call returned by model:", tool_call[0])
        # If the tool call contains a payload you need to execute, print/inspect it here
        args = json.loads(tool_call[0].function.arguments)
        
        # Call the MCP server
        async with Client(client_url) as mcp_client:
            #result = await mcp_client.call_tool(tool_call[0].function.name, args)
            result = await mcp_client.call_tool("return_name", {"name": "Ortega"})
        print("Tool result:", result)
    else:
        # fallback to assistant text
        print("Assistant text:", msg.content)

    print(response)