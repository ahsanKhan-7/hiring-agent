import pytest
from conftest import GOOD, JD, STUFFER
from langgraph.checkpoint.memory import MemorySaver
from mcp import Client

from hiring_agent import scoring
from hiring_agent.graph import build_graph, initial_state
from hiring_agent.mcp_server import server
from hiring_agent.tools_client import LocalTools, McpTools


async def test_server_lists_both_tools_with_schemas():
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"extract_skills", "score_resumes"}
    assert "jd_text" in tools["score_resumes"].input_schema["properties"]


async def test_mcp_results_match_the_direct_functions():
    via_mcp = await McpTools(server).score_resumes(JD, [GOOD, STUFFER])
    direct = scoring.score_resumes(JD, [GOOD, STUFFER])
    assert [(c["name"], c["score"]) for c in via_mcp] == [(c["name"], c["score"]) for c in direct]
    assert await McpTools(server).extract_skills(JD) == await LocalTools().extract_skills(JD)


async def test_workflow_gives_identical_results_over_mcp_and_in_process():
    results = []
    for i, tools in enumerate([LocalTools(), McpTools(server)]):
        g = build_graph(tools=tools, checkpointer=MemorySaver())
        out = await g.ainvoke(initial_state(JD, [GOOD, STUFFER], top_k=1), {"configurable": {"thread_id": f"m{i}"}})
        results.append([(c["name"], c["score"]) for c in out["ranking"]])
    assert results[0] == results[1]


async def test_server_rejects_malformed_and_oversized_input():
    async with Client(server) as client:
        bad = await client.call_tool("score_resumes", {"jd_text": JD, "resumes": [{"nope": 1}]})
        assert bad.is_error
        too_many = [{"name": str(i), "text": "x"} for i in range(51)]
        assert (await client.call_tool("score_resumes", {"jd_text": JD, "resumes": too_many})).is_error
