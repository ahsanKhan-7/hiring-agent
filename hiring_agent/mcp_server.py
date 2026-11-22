"""MCP server exposing the screening tools, so any MCP client (Claude Desktop, an agent, the
LangGraph workflow in this repo) can call them.

Run over stdio:  python -m hiring_agent.mcp_server
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from hiring_agent import scoring

MAX_RESUMES = 50
MAX_CHARS = 20_000

server = MCPServer("hiring-tools", instructions=(
    "Tools for screening resumes against a job description. Scores are a transparent weighted "
    "scorecard for prioritising human review, not hiring decisions."))


def _check(resumes: list) -> list:
    if len(resumes) > MAX_RESUMES:
        raise ValueError(f"At most {MAX_RESUMES} resumes per call.")
    clean = []
    for r in resumes:
        if not isinstance(r, dict) or "name" not in r or "text" not in r:
            raise ValueError("Each resume must be an object with 'name' and 'text'.")
        clean.append({"name": str(r["name"])[:100], "text": str(r["text"])[:MAX_CHARS]})
    return clean


@server.tool()
def extract_skills(text: str) -> list[str]:
    """Return the known skills mentioned in a job description or resume."""
    return scoring.extract_skills(text[:MAX_CHARS])


@server.tool()
def score_resumes(jd_text: str, resumes: list[dict]) -> list[dict]:
    """Rank resumes against a job description. Each resume is {"name": str, "text": str}.

    Returns best first, with an exact per-signal score breakdown, matched and missing skills,
    skills that were only listed (no evidence in experience), and estimated experience years.
    """
    return scoring.score_resumes(jd_text[:MAX_CHARS], _check(resumes))


if __name__ == "__main__":
    server.run()
