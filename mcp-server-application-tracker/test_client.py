"""
Quick manual test script for the Job Application Tracker MCP server.

Calls tools directly against the in-memory `mcp` server object -
no subprocess, no transport, no separate client needed.

Usage:
    python test_client.py
"""

import asyncio
from mcp import Client
from server import mcp


async def main():
    async with Client(mcp) as client:
        # List current applications
        result = await client.call_tool("list_applications", {})
        print("Current applications:")
        print(result.content[0].text)
        print()

        # Add a new application
        result = await client.call_tool(
            "add_application",
            {
                "company": "Example Corp",
                "role": "Software Engineer",
                "date_applied": "2026-08-01",
                "notes": "Test entry from test_client.py",
            },
        )
        print("Add result:")
        print(result.content[0].text)
        print()

        # List again to confirm
        result = await client.call_tool("list_applications", {})
        print("Updated applications:")
        print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
