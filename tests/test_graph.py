import pytest
from conftest import CLEAN_SET, GOOD, JD, OFF, STUFFER
from langgraph.types import Command

from hiring_agent.graph import find_flags, initial_state
from hiring_agent.llm import TemplateLLM

CFG = {"configurable": {"thread_id": "t"}}


def cfg(i):
    return {"configurable": {"thread_id": f"t{i}"}}


# ---- the critic, as a pure function -------------------------------------
def rank(name, score, listed_only=(), rank_no=1):
    return {"name": name, "score": score, "listed_only_skills": list(listed_only), "rank": rank_no}


def test_critic_flags_keyword_stuffing():
    flags = find_flags([rank("a", 80), rank("b", 70, ["x", "y", "z"])], ["x", "y", "z", "w"], top_k=2)
    assert [f["kind"] for f in flags] == ["keyword_stuffing"]


def test_critic_flags_close_call_at_the_cutoff_and_thin_jd():
    flags = find_flags([rank("a", 80), rank("b", 60), rank("c", 59)], ["x"], top_k=2)
    assert {f["kind"] for f in flags} == {"close_call", "thin_jd"}


def test_critic_is_quiet_on_a_clear_ranking():
    assert find_flags([rank("a", 80), rank("b", 60), rank("c", 20)], ["x", "y", "z"], top_k=2) == []


# ---- the workflow ----------------------------------------------------------
async def test_clean_run_completes_without_human_review(graph):
    out = await graph.ainvoke(initial_state(JD, CLEAN_SET, top_k=2), cfg(1))
    assert out["status"] == "completed" and "__interrupt__" not in out
    assert [c["name"] for c in out["shortlist"]] == ["good", "ok"]
    assert all(len(q) == 5 for q in out["interviews"].values())
    assert out["trace"][0].startswith("analyze_jd") and out["trace"][-1] == "finalize: completed"


async def test_flags_pause_the_graph_for_human_review(graph):
    out = await graph.ainvoke(initial_state(JD, [GOOD, STUFFER, OFF], top_k=2), cfg(2))
    payload = out["__interrupt__"][0].value
    assert any(f["kind"] == "keyword_stuffing" and f["candidate"] == "stuffer" for f in payload["flags"])
    state = await graph.aget_state(cfg(2))
    assert state.next == ("human_review",) and "shortlist" not in state.values


async def test_approve_with_exclusion_removes_the_candidate(graph):
    await graph.ainvoke(initial_state(JD, [GOOD, STUFFER, OFF], top_k=2), cfg(3))
    out = await graph.ainvoke(Command(resume={"approve": True, "exclude": ["stuffer"], "note": "stuffed"}), cfg(3))
    assert out["status"] == "completed"
    assert [c["name"] for c in out["shortlist"]] == ["good", "off"]
    assert out["review"]["note"] == "stuffed"


async def test_reject_ends_without_a_shortlist_or_llm_calls(graph):
    await graph.ainvoke(initial_state(JD, [GOOD, STUFFER, OFF], top_k=2), cfg(4))
    out = await graph.ainvoke(Command(resume={"approve": False}), cfg(4))
    assert out["status"] == "rejected" and "shortlist" not in out and "interviews" not in out


async def test_unknown_exclusions_are_ignored(graph):
    await graph.ainvoke(initial_state(JD, [GOOD, STUFFER, OFF], top_k=2), cfg(5))
    out = await graph.ainvoke(Command(resume={"approve": True, "exclude": ["nobody"]}), cfg(5))
    assert out["review"]["exclude"] == []


async def test_garbage_resume_value_is_treated_as_reject(graph):
    await graph.ainvoke(initial_state(JD, [GOOD, STUFFER, OFF], top_k=2), cfg(6))
    out = await graph.ainvoke(Command(resume="yes please"), cfg(6))
    assert out["status"] == "rejected"


async def test_no_resumes_finishes_cleanly(graph):
    out = await graph.ainvoke(initial_state(JD, [], top_k=2), cfg(7))
    assert out["status"] == "no_candidates"


async def test_runs_are_isolated_by_thread_id(graph):
    await graph.ainvoke(initial_state(JD, [GOOD, STUFFER, OFF], top_k=2), cfg(8))
    out = await graph.ainvoke(initial_state(JD, CLEAN_SET, top_k=2), cfg(9))
    assert out["status"] == "completed"
    assert (await graph.aget_state(cfg(8))).next == ("human_review",)


# ---- LLM failure handling ----------------------------------------------------
class BrokenLLM:
    async def generate(self, prompt):
        raise RuntimeError("provider down")


class JunkLLM:
    async def generate(self, prompt):
        return "Sure! Here are some thoughts."


@pytest.mark.parametrize("llm", [BrokenLLM(), JunkLLM()])
async def test_bad_llm_never_produces_a_short_plan(llm):
    from langgraph.checkpoint.memory import MemorySaver
    from hiring_agent.graph import build_graph
    g = build_graph(llm=llm, checkpointer=MemorySaver())
    out = await g.ainvoke(initial_state(JD, CLEAN_SET, top_k=2), cfg(10))
    assert out["status"] == "completed" and all(len(q) == 5 for q in out["interviews"].values())


async def test_good_llm_output_is_used():
    from langgraph.checkpoint.memory import MemorySaver
    from hiring_agent.graph import build_graph

    class Fixed:
        async def generate(self, prompt):
            return "\n".join(f"{i}. custom question {i}?" for i in range(1, 6))
    g = build_graph(llm=Fixed(), checkpointer=MemorySaver())
    out = await g.ainvoke(initial_state(JD, CLEAN_SET, top_k=1), cfg(11))
    assert out["interviews"]["good"][0] == "custom question 1?"
