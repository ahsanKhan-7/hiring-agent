"""How the workflow reaches its tools: directly in-process, or over MCP.

The graph only depends on the `ToolBackend` protocol, so the same workflow runs against the
local functions (fast, used in most tests) or against any MCP server (used in production and in
an in-process MCP integration test).
"""
from __future__ import annotations

import json
from typing import Protocol

from hiring_agent import scoring


class ToolBackend(Protocol):
    async def extract_skills(self, text: str) -> list: ...
    async def score_resumes(self, jd_text: str, resumes: list) -> list: ...


class LocalTools:
    async def extract_skills(self, text: str) -> list:
        return scoring.extract_skills(text)

    async def score_resumes(self, jd_text: str, resumes: list) -> list:
        return scoring.score_resumes(jd_text, resumes)


class McpTools:
    """Calls the tools through an `mcp.Client`. `target` is anything Client accepts:
    an MCPServer instance (in-process), a URL, or StdioServerParameters."""

    def __init__(self, target):
        self.target = target

    @staticmethod
    def _payload(result):
        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {result.content}")
        if result.structured_content is not None:
            data = result.structured_content
            return data.get("result", data) if isinstance(data, dict) else data
        return json.loads(result.content[0].text)

    async def _call(self, name: str, args: dict):
        from mcp import Client
        async with Client(self.target) as client:
            return self._payload(await client.call_tool(name, args))

    async def extract_skills(self, text: str) -> list:
        return await self._call("extract_skills", {"text": text})

    async def score_resumes(self, jd_text: str, resumes: list) -> list:
        return await self._call("score_resumes", {"jd_text": jd_text, "resumes": resumes})
