"""Run the workflow end to end without the API: python examples/demo.py"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from hiring_agent.graph import build_graph, initial_state  # noqa: E402

JD = "We need Python, pandas, scikit-learn, sql and machine learning skills, plus git."
RESUMES = [
    {"name": "Ann", "text": "Ann\nExperience\nJan 2024 - Dec 2024 Data Scientist\nPython pandas scikit-learn sql machine learning git"},
    {"name": "Bob", "text": "Bob\nSkills\npython pandas scikit-learn sql machine learning git\nExperience\n2024 - 2025 Barista"},
    {"name": "Cy", "text": "Cy\nExperience\n2024 - 2025 Chef\nKitchen management"},
]


async def main():
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "demo"}}
    out = await graph.ainvoke(initial_state(JD, RESUMES, top_k=2), cfg)
    if "__interrupt__" in out:
        print("PAUSED for human review:")
        for flag in out["__interrupt__"][0].value["flags"]:
            print(" -", flag["kind"], flag.get("candidate", ""), flag["detail"])
        out = await graph.ainvoke(Command(resume={"approve": True, "exclude": ["Bob"]}), cfg)
    print("status:", out["status"], "| shortlist:", [c["name"] for c in out["shortlist"]])
    print("\n".join(out["trace"]))


asyncio.run(main())
