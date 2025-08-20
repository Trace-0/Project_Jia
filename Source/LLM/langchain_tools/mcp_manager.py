from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "blender": {
            "command": "uvx",
            "args": ["blender-mcp"],
            "transport" : "stdio"
        },
        "ddg-search": {
            "command": "uvx",
            "args": ["duckduckgo-mcp-server"],
            "transport": "stdio"
        }
    }
)

