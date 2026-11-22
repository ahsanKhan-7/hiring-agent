"""The screening workflow as a LangGraph state machine with a human-in-the-loop checkpoint.

    analyze_jd -> rank -> critic --(flags)--> human_review --(approve)--> shortlist -> interview_plan -> finalize
                              \\--(no flags)------------------------------^                              ^
                                                           human_review --(reject)-----------------------+

Roles (be honest about what each is):
  * analyze_jd / rank   call tools (in-process or over MCP): deterministic.
  * critic              rule-based reviewer that audits the ranking and raises flags. Not an LLM.
  * human_review        pauses the graph with `interrupt`; a person approves, rejects or excludes candidates.
  * interview_plan      the only LLM step; falls back to templates if the LLM fails or returns junk.
State is checkpointed per thread id, so a paused run survives restarts when a persistent saver is used.
"""
from __future__ import annotations

import asyncio
import math
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from hiring_agent.llm import LLM, TemplateLLM, build_prompt, parse_numbered
from hiring_agent.tools_client import LocalTools, ToolBackend

CLOSE_CALL_GAP = 3.0  # points between the last shortlisted and first excluded candidate


class ScreeningState(TypedDict, total=False):
    jd_text: str
    resumes: list
    top_k: int
    jd_skills: list
    ranking: list
    flags: list
    review: dict
    excluded: list
    shortlist: list
    interviews: dict
    status: str
    trace: Annotated[list, operator.add]


def find_flags(ranking: list, jd_skills: list, top_k: int, gap: float = CLOSE_CALL_GAP) -> list:
    """Rule-based audit of a ranking. Pure function so it can be tested directly."""
    flags = []
    if len(jd_skills) < 3:
        flags.append({"kind": "thin_jd", "detail": f"Only {len(jd_skills)} known skills found in the job "
                                                    "description, so skill coverage is unreliable."})
    threshold = max(3, math.ceil(len(jd_skills) / 2))
    for c in ranking[: top_k + 2]:
        if len(c.get("listed_only_skills", [])) >= threshold:
            flags.append({"kind": "keyword_stuffing", "candidate": c["name"],
                          "detail": f"{len(c['listed_only_skills'])} JD skills appear only in a bare skills "
                                    "list with no supporting experience."})
    if len(ranking) > top_k:
        diff = ranking[top_k - 1]["score"] - ranking[top_k]["score"]
        if diff < gap:
            flags.append({"kind": "close_call", "detail": f"Rank {top_k} and rank {top_k + 1} are only "
                                                          f"{diff:.1f} points apart."})
    return flags


def build_graph(tools: Optional[ToolBackend] = None, llm: Optional[LLM] = None, checkpointer=None):
    tools = tools or LocalTools()
    llm = llm or TemplateLLM()
    fallback = TemplateLLM()

    async def analyze_jd(state: ScreeningState):
        skills = await tools.extract_skills(state["jd_text"])
        return {"jd_skills": skills, "trace": [f"analyze_jd: {len(skills)} skills found"]}

    async def rank(state: ScreeningState):
        ranking = await tools.score_resumes(state["jd_text"], state["resumes"])
        return {"ranking": ranking, "trace": [f"rank: scored {len(ranking)} resumes"]}

    async def critic(state: ScreeningState):
        flags = find_flags(state["ranking"], state["jd_skills"], state["top_k"])
        return {"flags": flags, "trace": [f"critic: {len(flags)} flag(s) raised"]}

    def route_after_critic(state: ScreeningState) -> str:
        if not state["ranking"]:
            return "finalize"
        return "human_review" if state["flags"] else "shortlist"

    async def human_review(state: ScreeningState):
        decision = interrupt({
            "reason": "The critic raised flags. Approve, reject, or exclude candidates.",
            "flags": state["flags"],
            "top_candidates": [{"name": c["name"], "score": c["score"], "rank": c["rank"]}
                               for c in state["ranking"][: state["top_k"] + 2]],
        })
        decision = decision if isinstance(decision, dict) else {}
        approve = bool(decision.get("approve", False))
        known = {c["name"] for c in state["ranking"]}
        excluded = [n for n in decision.get("exclude", []) if n in known]
        return {"review": {"approve": approve, "exclude": excluded, "note": str(decision.get("note", ""))[:500]},
                "excluded": excluded,
                "trace": [f"human_review: {'approved' if approve else 'rejected'}, excluded {excluded}"]}

    def route_after_review(state: ScreeningState) -> str:
        return "shortlist" if state["review"]["approve"] else "finalize"

    async def shortlist(state: ScreeningState):
        out = [c for c in state["ranking"] if c["name"] not in state.get("excluded", [])][: state["top_k"]]
        return {"shortlist": out, "trace": [f"shortlist: {[c['name'] for c in out]}"]}

    async def interview_plan(state: ScreeningState):
        async def one(c):
            prompt = build_prompt(state["jd_text"], c["name"], c["matched_skills"], c["missing_skills"])
            try:
                qs = parse_numbered(await llm.generate(prompt))
            except Exception:
                qs = []
            if len(qs) < 5:  # LLM failed or returned junk: never ship a short or empty plan
                qs = parse_numbered(await fallback.generate(prompt))
            return c["name"], qs
        pairs = await asyncio.gather(*(one(c) for c in state["shortlist"]))
        return {"interviews": dict(pairs), "trace": [f"interview_plan: {len(pairs)} plan(s)"]}

    async def finalize(state: ScreeningState):
        if not state["ranking"]:
            status = "no_candidates"
        elif state.get("review") and not state["review"]["approve"]:
            status = "rejected"
        else:
            status = "completed"
        return {"status": status, "trace": [f"finalize: {status}"]}

    g = StateGraph(ScreeningState)
    for name, fn in [("analyze_jd", analyze_jd), ("rank", rank), ("critic", critic),
                     ("human_review", human_review), ("shortlist", shortlist),
                     ("interview_plan", interview_plan), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze_jd")
    g.add_edge("analyze_jd", "rank")
    g.add_edge("rank", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"human_review": "human_review", "shortlist": "shortlist", "finalize": "finalize"})
    g.add_conditional_edges("human_review", route_after_review, {"shortlist": "shortlist", "finalize": "finalize"})
    g.add_edge("shortlist", "interview_plan")
    g.add_edge("interview_plan", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def initial_state(jd_text: str, resumes: list, top_k: int = 3) -> dict:
    return {"jd_text": jd_text, "resumes": resumes, "top_k": top_k, "trace": []}
